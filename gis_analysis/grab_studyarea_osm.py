#!/usr/bin/env python3
"""
Clip Geofabrik “free” OSM shapefile layers to a study-area polygon.

Geofabrik’s shapefile extracts are OSM-derived GIS layers (roads, buildings, …),
not native .osm XML. This script reads those .shp files, clips each layer to your
study polygon, and writes GeoPackages (default) under GIS/OSM.

For native OSM PBF clipped to a polygon, download a regional .osm.pbf from
Geofabrik and use osmium-tool:
  osmium extract -p study_area.geojson region.osm.pbf -o clipped.osm.pbf

Optional --mode overpass can request OSM XML from a public Overpass instance
(often fails on large areas with HTTP 504).

Requirements:
  pip install geopandas shapely
  (requests only needed for --mode overpass)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ----------------------------
# Default paths (override with CLI or edit here)
# ----------------------------
STUDY_AREA_PATH = Path(
    "/Users/johnnymojica/Library/CloudStorage/OneDrive-SharedLibraries-Radbridge/"
    "Radbridge - Documents/Radbridge Incorporated/3. Consulting/EDWA 24.06.01/"
    "Phase 2/EDWA Capture Area.geojson"
)

OUTPUT_DIR = Path(
    "/Users/johnnymojica/Library/CloudStorage/OneDrive-SharedLibraries-Radbridge/"
    "Radbridge - Documents/Radbridge Incorporated/3. Consulting/EDWA 24.06.01/"
    "Phase 2/GIS/OSM"
)

# Folder containing gis_osm_*.shp from Geofabrik (NorCal free extract, etc.)
GEOFABRIK_SHP_DIR = Path(
    "/Users/johnnymojica/Downloads/norcal-260326-free.shp"
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = (
    "grab_studyarea_osm/1.0 (EDWA Phase 2 study; "
    "contact: via project maintainer)"
)
REQUEST_TIMEOUT_SEC = 900


def _sanitize_layer_name(name: str) -> str:
    """GeoPackage layer names: short, alphanumeric + underscore."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return s[:60] if len(s) > 60 else s


def load_study_union(study_path: Path) -> gpd.GeoDataFrame:
    study = gpd.read_file(study_path)
    if study.empty:
        raise ValueError("Study area file is empty.")
    dissolved = study.dissolve()
    if dissolved.crs is None:
        raise ValueError(
            "Study area has no CRS. Assign one (e.g. EPSG:4326) before clipping."
        )
    return dissolved


def iter_geofabrik_shapefiles(geofabrik_dir: Path) -> list[Path]:
    if not geofabrik_dir.is_dir():
        raise FileNotFoundError(f"Geofabrik shapefile directory not found: {geofabrik_dir}")
    paths = sorted(geofabrik_dir.glob("gis_osm_*.shp"))
    if not paths:
        paths = sorted(geofabrik_dir.glob("*.shp"))
    return paths


def clip_geofabrik_layers(
    study_path: Path,
    geofabrik_dir: Path,
    output_dir: Path,
    driver: str,
    output_crs: str,
) -> None:
    """
    Clip each Geofabrik OSM layer to the study polygon.

    Projection: the study polygon is reprojected to each layer's CRS before
    gpd.clip (required). After clipping, results are reprojected to output_crs
    when output_crs is set (default EPSG:4326) so outputs match WGS 84 unless
    you pass --output-crs source.

    Uses bbox-filtered reads when supported (pyogrio) to avoid loading all of
    NorCal into memory for each layer.
    """
    print(f"Reading study area: {study_path}")
    study_union = load_study_union(study_path)
    print(f"Study area CRS: {study_union.crs}")

    output_dir.mkdir(parents=True, exist_ok=True)

    shapefiles = iter_geofabrik_shapefiles(geofabrik_dir)
    if not shapefiles:
        raise FileNotFoundError(
            f"No .shp files found under {geofabrik_dir} (expected gis_osm_*.shp)."
        )

    print(f"Found {len(shapefiles)} shapefile layer(s) in {geofabrik_dir}")

    for shp in shapefiles:
        stem = shp.stem
        print(f"  {stem} ...", flush=True)

        head = gpd.read_file(shp, rows=1)
        if head.empty:
            print(f"    (empty source, skip)")
            continue
        layer_crs = head.crs
        if layer_crs is None:
            print(f"    (shapefile has no CRS — set .prj or skip)")
            continue

        study_in_layer_crs = study_union.to_crs(layer_crs)
        bbox = tuple(study_in_layer_crs.total_bounds)

        try:
            gdf = gpd.read_file(shp, bbox=bbox)
        except Exception:
            gdf = gpd.read_file(shp)

        if gdf.empty:
            print(f"    no features in study bbox, skip")
            continue

        clipped = gpd.clip(gdf, study_in_layer_crs)
        if clipped.empty:
            print(f"    no features intersect study polygon, skip")
            continue

        if output_crs is not None and output_crs.lower() != "source":
            clipped = clipped.to_crs(output_crs)
        out_crs_note = clipped.crs
        print(f"    clip in {layer_crs}; write CRS {out_crs_note}")

        layer_name = _sanitize_layer_name(stem)

        if driver == "GPKG":
            out_path = output_dir / f"{stem}_clipped.gpkg"
            clipped.to_file(out_path, driver="GPKG", layer=layer_name)
            print(f"    -> {out_path} ({len(clipped):,} features)")
        elif driver == "GeoJSON":
            out_path = output_dir / f"{stem}_clipped.geojson"
            clipped.to_file(out_path, driver="GeoJSON")
            print(f"    -> {out_path} ({len(clipped):,} features)")
        elif driver == "ESRI Shapefile":
            out_dir = output_dir / f"{stem}_clipped_shp"
            out_dir.mkdir(parents=True, exist_ok=True)
            clipped.to_file(out_dir / f"{stem}.shp", driver="ESRI Shapefile")
            print(f"    -> {out_dir}/ ({len(clipped):,} features)")
        else:
            raise ValueError(f"Unsupported driver: {driver}")


# --- Optional Overpass (OSM XML) ---


def _polygon_to_overpass_poly_string(poly: Polygon) -> str:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return " ".join(f"{lat} {lon}" for lon, lat in coords)


def _study_geometries_wgs84(study_union: gpd.GeoDataFrame) -> list[Polygon]:
    geom = unary_union(study_union.geometry)
    if geom.is_empty:
        raise ValueError("Study area geometry is empty.")
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    raise ValueError(f"Unsupported geometry type for study area: {geom.geom_type!r}")


def build_overpass_query(poly_strings: list[str], timeout_sec: int) -> str:
    lines = [f"[out:xml][timeout:{timeout_sec}];", "("]
    for ps in poly_strings:
        lines.append(f'  node(poly:"{ps}");')
        lines.append(f'  way(poly:"{ps}");')
        lines.append(f'  relation(poly:"{ps}");')
    lines.append(");")
    lines.append("(._;>;);")
    lines.append("out meta;")
    return "\n".join(lines)


def download_osm_overpass(
    study_path: Path,
    output_path: Path,
    overpass_url: str,
    timeout_sec: int,
) -> None:
    import requests

    print(f"Reading study area: {study_path}")
    study = gpd.read_file(study_path)
    if study.empty:
        raise ValueError("Study area file is empty.")

    study_union = study.dissolve()
    study_wgs = study_union.to_crs(epsg=4326)
    polys = _study_geometries_wgs84(study_wgs)
    poly_strings = [_polygon_to_overpass_poly_string(p) for p in polys]
    query = build_overpass_query(poly_strings, timeout_sec=timeout_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"POST {overpass_url} (timeout {timeout_sec}s)...")
    headers = {"User-Agent": USER_AGENT, "Content-Type": "text/plain; charset=utf-8"}
    t0 = time.perf_counter()
    resp = requests.post(
        overpass_url,
        data=query.encode("utf-8"),
        headers=headers,
        timeout=timeout_sec + 60,
    )
    elapsed = time.perf_counter() - t0
    print(f"Response HTTP {resp.status_code} in {elapsed:.1f}s")

    if resp.status_code == 429:
        raise RuntimeError(
            "Overpass rate-limited (HTTP 429). Wait and retry or use "
            "--mode geofabrik with a local Geofabrik extract."
        )
    if resp.status_code >= 500:
        raise RuntimeError(
            f"Overpass server error {resp.status_code}. Large areas often return 504; "
            f"use Geofabrik shapefiles (--mode geofabrik) or osmium on a .pbf."
        )
    resp.raise_for_status()

    body = resp.content
    if not body.strip().startswith(b"<?xml") and not body.strip().startswith(b"<osm"):
        preview = body[:800].decode("utf-8", errors="replace")
        raise RuntimeError(
            "Response does not look like OSM XML. First bytes:\n" + preview
        )

    output_path.write_bytes(body)
    print(f"Wrote {len(body):,} bytes -> {output_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Clip Geofabrik OSM shapefile layers to a study polygon, "
        "or optionally download OSM XML via Overpass."
    )
    p.add_argument(
        "--mode",
        choices=("geofabrik", "overpass"),
        default="geofabrik",
        help="geofabrik: clip local gis_osm_*.shp (default). overpass: OSM XML via API.",
    )
    p.add_argument(
        "--study",
        type=Path,
        default=STUDY_AREA_PATH,
        help="Study area vector file (GeoJSON, etc.).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for clipped layers or .osm file.",
    )
    p.add_argument(
        "--geofabrik-dir",
        type=Path,
        default=GEOFABRIK_SHP_DIR,
        help="Directory containing Geofabrik gis_osm_*.shp files.",
    )
    p.add_argument(
        "--format",
        choices=("gpkg", "geojson", "shp"),
        default="gpkg",
        help="Output format for --mode geofabrik (default: gpkg).",
    )
    p.add_argument(
        "--output-crs",
        default="EPSG:4326",
        metavar="CRS",
        help=(
            "CRS for written layers after clipping (default: EPSG:4326). "
            "Use 'source' to keep each layer's native CRS from the shapefile."
        ),
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="study_area.osm",
        help="Output .osm filename for --mode overpass only.",
    )
    p.add_argument(
        "--overpass-url",
        type=str,
        default=OVERPASS_URL,
        help="Overpass interpreter URL (overpass mode only).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT_SEC,
        help="Overpass timeout seconds (overpass mode only).",
    )
    args = p.parse_args(argv)

    driver_map = {"gpkg": "GPKG", "geojson": "GeoJSON", "shp": "ESRI Shapefile"}

    try:
        if args.mode == "geofabrik":
            clip_geofabrik_layers(
                study_path=args.study,
                geofabrik_dir=args.geofabrik_dir,
                output_dir=args.output_dir,
                driver=driver_map[args.format],
                output_crs=args.output_crs,
            )
            print("Done.")
        else:
            out_path = args.output_dir / args.output_name
            download_osm_overpass(
                study_path=args.study,
                output_path=out_path,
                overpass_url=args.overpass_url,
                timeout_sec=args.timeout,
            )
            print("Done.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
