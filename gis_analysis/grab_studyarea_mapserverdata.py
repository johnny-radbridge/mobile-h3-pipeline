#!/usr/bin/env python3
"""
Download PAD-US Landforms data intersecting a study area.

Source service:
https://edits.nationalmap.gov/arcgis/rest/services/PAD-US/PAD_US_Landforms/MapServer/0

What it does:
1. Reads the study area GeoJSON
2. Dissolves to one geometry
3. Reprojects to the ArcGIS service CRS
4. Requests all intersecting feature OBJECTIDs
5. Downloads matching features in chunks
6. Saves outputs to the target GIS folder

Outputs:
- padus_landforms_upper_american_river_watershed.gpkg
- padus_landforms_upper_american_river_watershed.geojson
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

# ----------------------------
# User paths
# ----------------------------
STUDY_AREA_PATH = Path(
    "/Users/johnnymojica/Library/CloudStorage/OneDrive-SharedLibraries-Radbridge/"
    "Radbridge - Documents/Radbridge Incorporated/3. Consulting/EDWA 24.06.01/"
    "Phase 2/EDWA Capture Area.geojson"
)

OUTPUT_DIR = Path(
    "/Users/johnnymojica/Library/CloudStorage/OneDrive-SharedLibraries-Radbridge/"
    "Radbridge - Documents/Radbridge Incorporated/3. Consulting/EDWA 24.06.01/"
    "Phase 2/GIS"
)

# PAD-US Landforms layer 0
LAYER_URL = (
    "https://edits.nationalmap.gov/arcgis/rest/services/"
    "PAD-US/PAD_US_Landforms/MapServer/0"
)

OUTPUT_STEM = "padus_landforms_upper_american_river_watershed"

# Service max record count is 2000; keep batches smaller for reliability.
BATCH_SIZE = 500
REQUEST_TIMEOUT = 120


def get_json(url: str, params: dict) -> dict:
    """GET JSON with basic error handling."""
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS service error: {json.dumps(data['error'], indent=2)}")
    return data


def post_json(url: str, data: dict) -> dict:
    """POST JSON with basic error handling."""
    resp = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    out = resp.json()
    if "error" in out:
        raise RuntimeError(f"ArcGIS service error: {json.dumps(out['error'], indent=2)}")
    return out


def chunked(seq, size):
    """Yield successive chunks from a list."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def geo_interface_to_esri_polygon(geom_mapping: dict, wkid: int) -> dict:
    """
    Convert a GeoJSON-style mapping (__geo_interface__) to Esri polygon JSON.

    ArcGIS REST query rejects GeoJSON geometry strings; Esri JSON with rings +
    spatialReference is required.
    """
    sr = {"wkid": int(wkid)}
    gtype = geom_mapping["type"]
    if gtype == "Polygon":
        return {"rings": geom_mapping["coordinates"], "spatialReference": sr}
    if gtype == "MultiPolygon":
        rings = []
        for polygon in geom_mapping["coordinates"]:
            rings.extend(polygon)
        return {"rings": rings, "spatialReference": sr}
    raise ValueError(
        f"Expected Polygon or MultiPolygon for study area, got {gtype!r}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading study area...")
    study = gpd.read_file(STUDY_AREA_PATH)
    if study.empty:
        raise ValueError("Study area file is empty.")

    # Dissolve to one geometry
    study_union = study.dissolve()

    print("Fetching service metadata...")
    service_info = get_json(LAYER_URL, {"f": "json"})

    # Prefer the layer's native spatial reference
    wkid = None
    if "extent" in service_info and service_info["extent"].get("spatialReference"):
        wkid = service_info["extent"]["spatialReference"].get("latestWkid") or service_info["extent"]["spatialReference"].get("wkid")

    if wkid is None:
        # Fallback from parent service docs if needed; PAD-US Landforms is Web Mercator
        wkid = 3857

    object_id_field = service_info.get("objectIdField", "OBJECTID")
    print(f"Service WKID: {wkid}")
    print(f"Object ID field: {object_id_field}")

    print("Reprojecting study area to service CRS...")
    study_union = study_union.to_crs(epsg=wkid)

    geom_mapping = study_union.geometry.iloc[0].__geo_interface__
    esri_geom = geo_interface_to_esri_polygon(geom_mapping, wkid)

    # ArcGIS query endpoint
    query_url = f"{LAYER_URL}/query"

    print("Requesting intersecting OBJECTIDs...")
    oid_params = {
        "f": "json",
        "geometry": json.dumps(esri_geom),
        "geometryType": "esriGeometryPolygon",
        "inSR": wkid,
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "returnGeometry": "false",
        "where": "1=1",
    }

    oid_result = post_json(query_url, oid_params)
    object_ids = oid_result.get("objectIds", [])

    if not object_ids:
        print("No intersecting PAD-US Landforms features found.")
        return

    object_ids = sorted(object_ids)
    print(f"Found {len(object_ids):,} intersecting features.")

    features = []
    total_batches = math.ceil(len(object_ids) / BATCH_SIZE)

    for batch_num, oid_batch in enumerate(chunked(object_ids, BATCH_SIZE), start=1):
        print(f"Downloading batch {batch_num}/{total_batches} ({len(oid_batch)} features)...")

        batch_params = {
            "f": "geojson",
            "objectIds": ",".join(map(str, oid_batch)),
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": wkid,
        }

        resp = requests.post(query_url, data=batch_params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        gj = resp.json()

        if "error" in gj:
            raise RuntimeError(f"Batch query failed: {json.dumps(gj['error'], indent=2)}")

        batch_features = gj.get("features", [])
        features.extend(batch_features)

        # Be polite to the service
        time.sleep(0.2)

    print(f"Downloaded {len(features):,} features total.")

    if not features:
        print("No features returned after batching.")
        return

    print("Building GeoDataFrame...")
    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{wkid}")

    # Clip to study area to keep only intersecting geometry within the watershed/capture area
    print("Clipping to study area boundary...")
    clipped = gpd.clip(gdf, study_union)

    # Also save in WGS84 for portability if desired
    clipped_wgs84 = clipped.to_crs(epsg=4326)

    gpkg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.gpkg"
    geojson_path = OUTPUT_DIR / f"{OUTPUT_STEM}.geojson"

    print(f"Writing GeoPackage: {gpkg_path}")
    clipped.to_file(gpkg_path, driver="GPKG")

    print(f"Writing GeoJSON: {geojson_path}")
    clipped_wgs84.to_file(geojson_path, driver="GeoJSON")

    print("Done.")
    print(f"Saved {len(clipped):,} clipped features.")
    print(gpkg_path)
    print(geojson_path)


if __name__ == "__main__":
    main()
