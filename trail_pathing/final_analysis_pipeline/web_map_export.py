"""Shared Folium helpers for client-facing interactive HTML maps."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import branca.colormap as cm
import folium
import geopandas as gpd
from branca.element import MacroElement, Template
from folium import FeatureGroup, GeoJson, LayerControl, TileLayer
from folium.features import GeoJsonTooltip
from activity_ranking.constants import ALL_RANKED_ACTIVITY_KEYS
from h3_device_hours_core import (
    ALL_RECREATION_ACTIVITY_KEY,
    RECREATION_SEASON_ORDER,
    period_to_display_label,
)

_PIPELINE_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _PIPELINE_DIR.parent.parent

# Carto basemaps — no API key (works on GitHub Pages).
CARTO_VOYAGER_TILES = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png"
CARTO_VOYAGER_LABELS_TILES = (
    "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
)
CARTO_LIGHT_NOLABELS_TILES = (
    "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
)
CARTO_LIGHT_LABELS_TILES = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_DARK_MATTER_NOLABELS_TILES = (
    "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
)
# Dark + place/road/water labels (H3 heatmap default).
CARTO_DARK_MATTER_LABELS_TILES = (
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
)
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)

# Census cartographic counties clipped to study area (map_assets/build_counties_geojson.py).
DEFAULT_BOUNDARIES_GEOJSON = _PIPELINE_DIR / "map_assets" / "counties_study_area.geojson"
# Back-compat alias
DEFAULT_PLACES_GEOJSON = DEFAULT_BOUNDARIES_GEOJSON

# Always-on reference outlines (not listed in the layer control).
REFERENCE_BOUNDARY_STYLE: dict[str, Any] = {
    "fillColor": "#e8e4dc",
    "color": "#6e6e6e",
    "weight": 0.85,
    "fillOpacity": 0.05,
    "opacity": 0.9,
}
REFERENCE_BOUNDARY_HIGHLIGHT_STYLE: dict[str, Any] = {
    "fillColor": "#ddd8ce",
    "color": "#3d3d3d",
    "weight": 1.1,
    "fillOpacity": 0.1,
    "opacity": 1.0,
}

H3_TOOLTIP_FIELDS: tuple[tuple[str, str], ...] = (("included_activities", "Included activities"),)
# Back-compat alias
H3_COMPOSITION_TOOLTIP_FIELDS = H3_TOOLTIP_FIELDS

ACTIVITY_DISPLAY_NAMES: dict[str, str] = {
    ALL_RECREATION_ACTIVITY_KEY: "All recreation",
    "ski": "Ski",
    "trails": "Trails",
    "ohv": "OHV",
    "golf": "Golf",
    "camping": "Camping",
    "farm": "Farm visits",
    "lakes": "Lakes",
    "rivers_and_streams": "Rivers & streams",
    "whitewater": "Whitewater",
    "local_parks": "Local parks",
    "state_parks": "State parks",
    "blm_land": "BLM land",
    "usfs_land": "USFS land",
    "cdfw_land": "CDFW land",
    "private_conservation": "Private conservation",
}

# Distinct sequential ramps (ColorBrewer-style) per activity for county layers.
COUNTY_COLOR_RAMPS: dict[str, list[str]] = {
    ALL_RECREATION_ACTIVITY_KEY: ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3", "#3f007d"],
    "ski": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "trails": ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
    "ohv": ["#f7fcfd", "#9ecae1", "#3182bd", "#08519c", "#08306b"],
    "golf": ["#f7fcf0", "#addd8e", "#41ab5d", "#006837"],
    "camping": ["#fff7ec", "#fdd49e", "#fdbb84", "#e34a33", "#7f0000"],
    "farm": ["#fff7f3", "#fde0dd", "#fa9fb5", "#c51b8a", "#7a0177"],
    "lakes": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "rivers_and_streams": ["#f0f9e8", "#bae4bc", "#74c476", "#238b45", "#005a32"],
    "whitewater": ["#edf8fb", "#b2e2e2", "#66c2a4", "#238b45", "#005824"],
    "local_parks": ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"],
    "state_parks": ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "blm_land": ["#ffffe5", "#fff7bc", "#fec44f", "#d95f0e", "#993404"],
    "usfs_land": ["#f7fcf5", "#bae4b3", "#78c679", "#31a354", "#006d2c"],
    "cdfw_land": ["#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f", "#990000"],
    "private_conservation": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
}

H3_COLOR_RAMPS: dict[str, list[str]] = COUNTY_COLOR_RAMPS

# Blue (low) → red (high); first stops are light enough for dark Carto tiles.
H3_HEAT_RAMP_BLUE_RED: list[str] = [
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
    "#fdbb84",
    "#fd8d3c",
    "#e6550d",
    "#d62728",
    "#a50f15",
]
# Same spectrum for light basemap toggle (lighter lows, strong reds).
H3_HEAT_RAMP_BLUE_RED_LIGHT: list[str] = [
    "#eff3ff",
    "#c6dbef",
    "#6baed6",
    "#3182bd",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#de2d26",
    "#a50f15",
]
# Back-compat alias
H3_COMPREHENSIVE_HEAT_RAMP = H3_HEAT_RAMP_BLUE_RED

H3_DETAIL_MAP_PANE = "h3DetailPane"
H3_COMPREHENSIVE_MAP_PANE = "h3ComprehensivePane"
# Color scale: cap vmax at 92nd percentile — reserve red for top ping density.
H3_HEATMAP_VMAX_PERCENTILE = 0.92
# Device-hours at or below this value map into the blue band only.
H3_HEATMAP_BLUE_CAP = 3.0
# Share of the ramp used for vmin..blue_cap (device-hours ~1–3).
H3_HEATMAP_BLUE_SPAN = 0.30
H3_HEATMAP_COLOR_SCALE = "log1p"
# Minimum ramp position (visible blue on dark basemap for quiet cells).
H3_HEATMAP_COLOR_FLOOR = 0.32
H3_HEATMAP_COLOR_GAMMA = 0.65

# H3 choropleth styling — faint outlines so fills read when zoomed out.
H3_HEX_STROKE_COLOR = "#4a5568"
H3_HEX_STROKE_WEIGHT = 0.35
H3_HEX_STROKE_OPACITY = 0.1
H3_HEX_FILL_OPACITY = 0.88
H3_HEX_HIGHLIGHT_STROKE_OPACITY = 0.35
# Heatmap profile — minimal stroke, strong fill.
H3_HEX_HEATMAP_STROKE = {
    "color": "#1a1a2e",
    "weight": 0.15,
    "opacity": 0.12,
    "fillOpacity": 0.92,
}

DEFAULT_COUNTY_GPKG = (
    _PIPELINE_DIR
    / "outputs"
    / "07_activity_origin_county_device_days"
    / "activity_origin_county_device_days.gpkg"
)
DEFAULT_COUNTY_HTML = DEFAULT_COUNTY_GPKG.with_name("activity_origin_county_device_days_map.html")

DEFAULT_H3_SEASONS_DIR = _PIPELINE_DIR / "outputs" / "06_h3_activity_device_hours" / "seasons"
# Back-compat alias (older quarter outputs lived under ``seasonal/``).
DEFAULT_H3_SEASONAL_DIR = DEFAULT_H3_SEASONS_DIR
DEFAULT_H3_GPKG_PREFIX = "h3_device_hours"
DEFAULT_H3_HTML = DEFAULT_H3_SEASONS_DIR / "h3_device_hours_map.html"

DEFAULT_H3_FULL_DAY_SEASONAL_DIR = (
    _PIPELINE_DIR / "outputs" / "10_h3_recreation_full_day_device_hours" / "seasonal"
)
DEFAULT_H3_FULL_DAY_GPKG_PREFIX = "h3_recreation_full_day"
DEFAULT_H3_FULL_DAY_HTML = DEFAULT_H3_FULL_DAY_SEASONAL_DIR / "h3_recreation_full_day_map.html"

DEFAULT_UARW_STUDY_GPKG = _ANALYSIS_ROOT / "gis_analysis" / "uarw_study.gpkg"
H3_STUDY_LAYER_PREFIX = "h3_device_hours_"
COUNTY_STUDY_LAYER_PREFIX = "origin_county_"
RECREATION_LANDS_LAYER_PREFIX = "recreation_lands_pois_"
RECREATION_LANDS_DATA_DIR_NAME = "recreation_lands_map_data"
# Simplify dense linework (trails, OHV) for smaller sidecar GeoJSON.
RECREATION_LANDS_SIMPLIFY_LARGE = 0.00015
RECREATION_LANDS_SIMPLIFY_DEFAULT = 0.00008
RECREATION_LANDS_LARGE_SUFFIXES = frozenset({"trails", "ohv_routes"})

RECREATION_LANDS_ACTIVITY_ALIASES: dict[str, str] = {
    "campgrounds": "camping",
    "ohv_routes": "ohv",
    "golf_courses": "golf",
    "ski_areas": "ski",
    "farm_parcels": "farm",
}

RECREATION_LANDS_TOOLTIP_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("name", "Name"),
    ("map_label", "Label"),
    ("farm_name", "Farm"),
    ("lake_name", "Lake"),
    ("lake_name_gnis", "GNIS name"),
    ("trail_class", "Trail class"),
    ("manager_agency", "Manager"),
    ("county", "County"),
    ("source_fid", "Source ID"),
)

RECREATION_SEASON_GPKG_RE = re.compile(
    r"^h3(?:_device_hours|_recreation_full_day)_(\d{4})_(winter|spring|summer|fall)\.gpkg$",
    re.IGNORECASE,
)
QUARTER_GPKG_RE = re.compile(
    r"^h3(?:_device_hours|_recreation_full_day)_(\d{4})_(Q[1-4])\.gpkg$",
    re.IGNORECASE,
)
SEASON_GPKG_RE = RECREATION_SEASON_GPKG_RE


def activity_label(activity_key: str) -> str:
    return ACTIVITY_DISPLAY_NAMES.get(activity_key, activity_key.replace("_", " ").title())


def list_gpkg_layers(gpkg_path: Path) -> list[str]:
    layers_df = gpd.list_layers(gpkg_path)
    return [str(name) for name in layers_df["name"].tolist()]


def sort_activity_layers(layer_names: Iterable[str]) -> list[str]:
    order = {key: idx for idx, key in enumerate(ALL_RANKED_ACTIVITY_KEYS)}
    return sorted(layer_names, key=lambda name: order.get(name, 999))


def is_h3_overview_layer_name(layer_name: str) -> bool:
    """Skip legacy res-8 ``*_z8`` layers when reading older GeoPackages."""
    from h3_device_hours_core import is_overview_layer_name

    return is_overview_layer_name(layer_name)


def sort_h3_detail_layers(layer_names: Iterable[str]) -> list[str]:
    """Activity layers only (excludes ``all_recreation``)."""
    names = [n for n in layer_names if not is_h3_overview_layer_name(n)]
    return sort_activity_layers(n for n in names if n != ALL_RECREATION_ACTIVITY_KEY)


def load_study_h3_layers(
    gpkg_path: Path,
    *,
    value_field: str = "device_hours",
) -> dict[str, gpd.GeoDataFrame]:
    """Load ``h3_device_hours_*`` layers from ``uarw_study.gpkg`` keyed by activity."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name in list_gpkg_layers(gpkg_path):
        if not layer_name.startswith(H3_STUDY_LAYER_PREFIX):
            continue
        activity = layer_name[len(H3_STUDY_LAYER_PREFIX) :]
        gdf = read_gpkg_layer(gpkg_path, layer_name)
        if gdf.empty or value_field not in gdf.columns:
            continue
        layers[activity] = gdf
    return layers


def annotate_h3_included_activities(
    activity: str,
    gdf: gpd.GeoDataFrame,
    activity_layers: dict[str, gpd.GeoDataFrame],
    *,
    h3_field: str = "h3",
    value_field: str = "device_hours",
) -> gpd.GeoDataFrame:
    """Add ``included_activities`` text for H3 tooltips."""
    out = gdf.copy()
    if out.empty or h3_field not in out.columns:
        return out

    if activity != ALL_RECREATION_ACTIVITY_KEY:
        out["included_activities"] = activity_label(activity)
        return out

    contributors: dict[str, list[str]] = {}
    for other_activity, other_gdf in activity_layers.items():
        if other_activity == ALL_RECREATION_ACTIVITY_KEY:
            continue
        if other_gdf.empty or h3_field not in other_gdf.columns:
            continue
        active = other_gdf
        if value_field in active.columns:
            active = active[active[value_field].fillna(0).astype(float) > 0]
        label = activity_label(other_activity)
        for h3_cell in active[h3_field].dropna().astype(str).unique().tolist():
            contributors.setdefault(h3_cell, []).append(label)

    out[h3_field] = out[h3_field].astype(str)
    out["included_activities"] = out[h3_field].map(
        lambda h3_cell: ", ".join(contributors.get(h3_cell, [])) or activity_label(activity)
    )
    return out


def load_study_county_layers(gpkg_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Load ``origin_county_*`` layers from ``uarw_study.gpkg`` keyed by activity."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name in list_gpkg_layers(gpkg_path):
        if not layer_name.startswith(COUNTY_STUDY_LAYER_PREFIX):
            continue
        activity = layer_name[len(COUNTY_STUDY_LAYER_PREFIX) :]
        gdf = read_gpkg_layer(gpkg_path, layer_name)
        if not gdf.empty:
            layers[activity] = gdf
    if layers and ALL_RECREATION_ACTIVITY_KEY not in layers:
        aggregate = aggregate_county_layers(layers)
        if aggregate is not None and not aggregate.empty:
            layers[ALL_RECREATION_ACTIVITY_KEY] = aggregate
    return layers


def aggregate_county_layers(
    activity_layers: dict[str, gpd.GeoDataFrame],
    *,
    value_field: str = "unique_device_localdate_id_count",
) -> gpd.GeoDataFrame | None:
    """Build an ``All recreation`` county layer by summing counts across activity layers."""
    frames: list[gpd.GeoDataFrame] = []
    for gdf in activity_layers.values():
        if gdf.empty or value_field not in gdf.columns:
            continue
        geoid_col = "GEOID" if "GEOID" in gdf.columns else "origin_county_geoid"
        county_name_col = "NAMELSAD" if "NAMELSAD" in gdf.columns else None
        keep = [geoid_col, value_field, "geometry"]
        if county_name_col:
            keep.insert(1, county_name_col)
        frames.append(gdf[keep].copy())
    if not frames:
        return None

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    geoid_col = "GEOID" if "GEOID" in combined.columns else "origin_county_geoid"
    group_cols = [geoid_col]
    if "NAMELSAD" in combined.columns:
        group_cols.append("NAMELSAD")
    totals = combined.groupby(group_cols, as_index=False)[value_field].sum()
    geom = combined.drop_duplicates(subset=[geoid_col])[[geoid_col, "geometry"]]
    out = totals.merge(geom, on=geoid_col, how="left")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=combined.crs)


def recreation_lands_activity_key(layer_suffix: str) -> str:
    """Map ``recreation_lands_pois_*`` suffix to activity ramp key."""
    return RECREATION_LANDS_ACTIVITY_ALIASES.get(layer_suffix, layer_suffix)


def recreation_lands_tooltip_fields(gdf: gpd.GeoDataFrame) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for field, label in RECREATION_LANDS_TOOLTIP_CANDIDATES:
        if field in gdf.columns:
            fields.append((field, label))
    return fields[:5]


def recreation_lands_style_for_activity(activity: str) -> dict[str, Any]:
    ramp = COUNTY_COLOR_RAMPS.get(activity, COUNTY_COLOR_RAMPS["trails"])
    fill_color = ramp[min(len(ramp) // 2, len(ramp) - 1)]
    stroke_color = ramp[-1]
    return {
        "fillColor": fill_color,
        "color": stroke_color,
        "weight": 1.0,
        "opacity": 0.85,
        "fillOpacity": 0.35,
    }


def load_study_recreation_lands_layers(gpkg_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Load ``recreation_lands_pois_*`` layers keyed by activity."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name in list_gpkg_layers(gpkg_path):
        if not layer_name.startswith(RECREATION_LANDS_LAYER_PREFIX):
            continue
        suffix = layer_name[len(RECREATION_LANDS_LAYER_PREFIX) :]
        activity = recreation_lands_activity_key(suffix)
        gdf = read_gpkg_layer(gpkg_path, layer_name)
        if not gdf.empty:
            layers[activity] = gdf
    return layers


def county_tooltip_fields(gdf: gpd.GeoDataFrame, value_field: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if "NAMELSAD" in gdf.columns:
        fields.append(("NAMELSAD", "County"))
    geoid_col = "GEOID" if "GEOID" in gdf.columns else "origin_county_geoid"
    if geoid_col in gdf.columns:
        fields.append((geoid_col, "GEOID"))
    if value_field in gdf.columns:
        fields.append((value_field, "Unique device-days"))
    return fields


def partition_h3_export_layers(layer_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return ``(comprehensive, detail)`` for map export ordering."""
    names = [n for n in layer_names if not is_h3_overview_layer_name(n)]
    comprehensive = [n for n in names if n == ALL_RECREATION_ACTIVITY_KEY]
    detail = sort_activity_layers(n for n in names if n != ALL_RECREATION_ACTIVITY_KEY)
    return comprehensive, detail


def aggregate_h3_device_hours_across_gpkgs(
    gpkg_paths: Sequence[Path],
    *,
    value_field: str = "device_hours",
) -> dict[str, gpd.GeoDataFrame]:
    """Sum ``device_hours`` by H3 cell across all period GeoPackages per activity layer."""
    activity_names: set[str] = set()
    for path in gpkg_paths:
        activity_names.update(list_gpkg_layers(path))
    activity_names = {
        name for name in activity_names if not is_h3_overview_layer_name(name)
    }

    aggregated: dict[str, gpd.GeoDataFrame] = {}
    for activity in sort_activity_layers(activity_names):
        frames: list[gpd.GeoDataFrame] = []
        for path in gpkg_paths:
            if activity not in list_gpkg_layers(path):
                continue
            gdf = read_gpkg_layer(path, activity)
            if gdf.empty or value_field not in gdf.columns or "h3" not in gdf.columns:
                continue
            frames.append(gdf[["h3", value_field, "geometry"]].copy())
        if not frames:
            continue
        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            crs=frames[0].crs,
        )
        totals = (
            combined.groupby("h3", as_index=False)[value_field]
            .sum()
            .rename(columns={value_field: value_field})
        )
        geom = combined.drop_duplicates(subset=["h3"])[["h3", "geometry"]]
        out = geom.merge(totals, on="h3", how="inner")
        aggregated[activity] = gpd.GeoDataFrame(out, geometry="geometry", crs=combined.crs)
    return aggregated


def h3_colors_for_activity(activity: str, *, heatmap: bool, light: bool = False) -> list[str]:
    if heatmap:
        if light:
            return list(H3_HEAT_RAMP_BLUE_RED_LIGHT)
        return list(H3_HEAT_RAMP_BLUE_RED)
    return list(H3_COLOR_RAMPS.get(activity, H3_COLOR_RAMPS["trails"]))


def is_comprehensive_h3_activity(activity: str) -> bool:
    return activity == ALL_RECREATION_ACTIVITY_KEY


def discover_seasonal_gpkgs(seasonal_dir: Path, *, gpkg_prefix: str) -> list[Path]:
    """All recreation-season ``{prefix}_YYYY_{season}.gpkg`` files, sorted chronologically."""
    if not seasonal_dir.is_dir():
        return []
    season_rank = {name: idx for idx, name in enumerate(RECREATION_SEASON_ORDER)}
    paths = [
        p
        for p in seasonal_dir.glob(f"{gpkg_prefix}_*.gpkg")
        if RECREATION_SEASON_GPKG_RE.match(p.name)
    ]
    return sorted(
        paths,
        key=lambda p: (
            int(RECREATION_SEASON_GPKG_RE.match(p.name).group(1)),  # type: ignore[union-attr]
            season_rank.get(
                RECREATION_SEASON_GPKG_RE.match(p.name).group(2).lower(),  # type: ignore[union-attr]
                99,
            ),
        ),
    )


def period_key_from_gpkg_path(gpkg_path: Path, *, gpkg_prefix: str) -> str:
    """``h3_device_hours_2024_winter.gpkg`` → ``2024_winter``."""
    stem = gpkg_path.stem
    prefix = f"{gpkg_prefix}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def period_key_from_gpkg_path(gpkg_path: Path, *, gpkg_prefix: str) -> str:
    """``h3_device_hours_2024_winter.gpkg`` → ``2024_winter``."""
    stem = gpkg_path.stem
    prefix = f"{gpkg_prefix}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def period_label_from_gpkg_path(gpkg_path: Path) -> str:
    """Human label from ``h3_device_hours_2024_winter.gpkg`` → ``Winter 2024``."""
    match = RECREATION_SEASON_GPKG_RE.match(gpkg_path.name)
    if match:
        return period_to_display_label(f"{match.group(1)}_{match.group(2).lower()}")
    match = QUARTER_GPKG_RE.match(gpkg_path.name)
    if match:
        return period_to_display_label(f"{match.group(1)}_Q{match.group(2)}")
    return gpkg_path.stem.replace("_", " ").title()


def linear_colormap(
    values: Sequence[float],
    colors: Sequence[str],
    *,
    caption: str,
) -> cm.LinearColormap:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(clean), max(clean)
        if vmin == vmax:
            vmax = vmin + 1.0
    return cm.LinearColormap(colors=list(colors), vmin=vmin, vmax=vmax, caption=caption)


def value_range(values: Sequence[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return 0.0, 1.0
    vmin, vmax = min(clean), max(clean)
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile; ``p`` in [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    p = max(0.0, min(1.0, p))
    k = (n - 1) * p
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def value_range_for_heatmap(
    values: Sequence[float],
    *,
    vmax_percentile: float = H3_HEATMAP_VMAX_PERCENTILE,
) -> tuple[float, float]:
    """Min and display max for choropleth (max is a percentile, not the spike)."""
    clean = sorted(
        float(v) for v in values if v is not None and math.isfinite(float(v))
    )
    if not clean:
        return 0.0, 1.0
    vmin = clean[0]
    vmax = percentile(clean, vmax_percentile)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _interpolate_hex_color(colors: Sequence[str], t: float) -> str:
    t = max(0.0, min(1.0, t))
    if len(colors) == 1:
        return colors[0]
    idx = t * (len(colors) - 1)
    lo = int(idx)
    hi = min(len(colors) - 1, lo + 1)
    frac = idx - lo
    c0, c1 = _hex_to_rgb(colors[lo]), _hex_to_rgb(colors[hi])
    return "#{:02x}{:02x}{:02x}".format(
        int(c0[0] + (c1[0] - c0[0]) * frac),
        int(c0[1] + (c1[1] - c0[1]) * frac),
        int(c0[2] + (c1[2] - c0[2]) * frac),
    )


def make_h3_style_function(
    vmin: float,
    vmax: float,
    colors: Sequence[str],
    value_field: str,
):
    """Python style callback (do not use branca colormap — it bloats the HTML)."""
    ramp = list(colors)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        raw = feature.get("properties", {}).get(value_field)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        return {
            "fillColor": _interpolate_hex_color(ramp, t),
            "color": H3_HEX_STROKE_COLOR,
            "weight": H3_HEX_STROKE_WEIGHT,
            "opacity": H3_HEX_STROKE_OPACITY,
            "fillOpacity": H3_HEX_FILL_OPACITY,
        }

    return style_function


def _h3_highlight_style(_feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "weight": 1.2,
        "color": "#1a202c",
        "opacity": H3_HEX_HIGHLIGHT_STROKE_OPACITY,
        "fillOpacity": 0.92,
    }


def bounds_with_padding(
    gdfs: Sequence[gpd.GeoDataFrame],
    *,
    padding_ratio: float = 0.04,
) -> list[list[float]]:
    frames = [g for g in gdfs if g is not None and not g.empty]
    if not frames:
        return [[39.0, -120.5], [38.5, -121.0]]
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    minx, miny, maxx, maxy = merged.total_bounds
    pad_x = (maxx - minx) * padding_ratio or 0.05
    pad_y = (maxy - miny) * padding_ratio or 0.05
    return [
        [miny - pad_y, minx - pad_x],
        [maxy + pad_y, maxx + pad_x],
    ]


def add_info_panel(m: folium.Map, *, title: str, subtitle: str, instructions: str) -> None:
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <div id="map-info-panel" style="
            position: fixed;
            bottom: 28px;
            left: 12px;
            z-index: 9999;
            max-width: 340px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid #d0d7de;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
            padding: 12px 14px;
            font: 13px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            color: #24292f;
        ">
            <div style="font-size: 15px; font-weight: 600; margin-bottom: 4px;">{{ this.title }}</div>
            <div style="color: #57606a; margin-bottom: 8px;">{{ this.subtitle }}</div>
            <div>{{ this.instructions }}</div>
        </div>
        {% endmacro %}
        """
    )

    panel = MacroElement()
    panel._template = template
    panel.title = title
    panel.subtitle = subtitle
    panel.instructions = instructions
    m.get_root().add_child(panel)


def resolve_boundaries_geojson(path: Path | None = None) -> Path | None:
    candidate = (path or DEFAULT_BOUNDARIES_GEOJSON).resolve()
    return candidate if candidate.is_file() else None


resolve_places_geojson = resolve_boundaries_geojson


def add_reference_boundary_layer(m: folium.Map, boundaries_geojson: Path) -> None:
    """Always-on county outlines (hidden from layer control)."""
    gdf = gpd.read_file(boundaries_geojson)
    if gdf.empty:
        return
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    keep = [c for c in ("GEOID", "NAME", "STATEFP") if c in gdf.columns] + ["geometry"]
    gdf = gdf[keep].copy()

    tip_fields = [c for c in ("NAME", "GEOID") if c in gdf.columns]
    tip_aliases = ["County", "GEOID"][: len(tip_fields)]

    GeoJson(
        data=gdf_to_geojson_dict(gdf, geometry_precision=1e-5),
        name="County boundaries",
        style_function=lambda _feature: dict(REFERENCE_BOUNDARY_STYLE),
        highlight_function=lambda _feature: dict(REFERENCE_BOUNDARY_HIGHLIGHT_STYLE),
        overlay=True,
        control=False,
        show=True,
        tooltip=GeoJsonTooltip(fields=tip_fields, aliases=tip_aliases, sticky=True)
        if tip_fields
        else None,
    ).add_to(m)


add_place_boundary_layer = add_reference_boundary_layer


def add_reference_overlays(
    m: folium.Map,
    *,
    boundaries_geojson: Path | None = None,
    places_geojson: Path | None = None,
) -> None:
    """Default county boundaries (always visible, not in layer control)."""
    path = resolve_boundaries_geojson(boundaries_geojson or places_geojson)
    if path is not None:
        add_reference_boundary_layer(m, path)


def make_base_map(
    bounds: list[list[float]],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    instructions: str | None = None,
    show_info_panel: bool = False,
    places_geojson: Path | None = None,
    boundaries_geojson: Path | None = None,
    show_county_boundaries: bool = False,
    show_reference_overlays: bool | None = None,
    basemap_tiles: str | None = None,
    return_tile_layer: bool = False,
    fit_bounds_max_zoom: int | None = None,
    initial_zoom: int | None = None,
    initial_center: tuple[float, float] | None = None,
    prefer_canvas: bool = False,
) -> folium.Map | tuple[folium.Map, str]:
    if show_reference_overlays is not None:
        show_county_boundaries = show_reference_overlays
    if initial_center is not None:
        center_lat, center_lon = initial_center
    else:
        center_lat = (bounds[0][0] + bounds[1][0]) / 2
        center_lon = (bounds[0][1] + bounds[1][1]) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        tiles=None,
        zoom_start=initial_zoom if initial_zoom is not None else 8,
        control_scale=True,
        prefer_canvas=prefer_canvas,
    )
    tile = TileLayer(
        tiles=basemap_tiles or CARTO_VOYAGER_TILES,
        attr=CARTO_ATTRIBUTION,
        name="Basemap",
        overlay=False,
        control=False,
    )
    tile.add_to(m)
    m.get_root().header.add_child(
        folium.Element(
            """
            <style>
                .leaflet-container .leaflet-interactive:focus {
                    outline: none;
                }
            </style>
            """
        )
    )
    if initial_zoom is None:
        fit_kwargs: dict[str, Any] = {}
        if fit_bounds_max_zoom is not None:
            fit_kwargs["max_zoom"] = fit_bounds_max_zoom
        m.fit_bounds(bounds, **fit_kwargs)
    if show_county_boundaries:
        add_reference_overlays(
            m,
            boundaries_geojson=boundaries_geojson,
            places_geojson=places_geojson,
        )
    if show_info_panel and title and subtitle and instructions:
        add_info_panel(m, title=title, subtitle=subtitle, instructions=instructions)
    if return_tile_layer:
        return m, tile.get_name()
    return m


def add_choropleth_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    *,
    layer_name: str,
    value_field: str,
    colors: Sequence[str],
    caption: str,
    tooltip_fields: Sequence[tuple[str, str]],
    show: bool = False,
    simplify_tolerance: float | None = None,
) -> FeatureGroup | None:
    if gdf.empty:
        return None

    keep_columns = list(dict.fromkeys([value_field, *[field for field, _ in tooltip_fields]]))
    gdf = slim_gdf_for_web(
        gdf,
        keep_columns=keep_columns,
        simplify_tolerance=simplify_tolerance,
    )
    vmin, vmax = value_range(gdf[value_field].astype(float).tolist())
    style_function = make_h3_style_function(vmin, vmax, colors, value_field)

    tip_aliases = [label for _, label in tooltip_fields]
    tip_fields = [field for field, _ in tooltip_fields]

    group = FeatureGroup(name=layer_name, show=show)
    GeoJson(
        data=gdf_to_geojson_dict(gdf),
        name=layer_name,
        style_function=style_function,
        highlight_function=_h3_highlight_style,
        tooltip=GeoJsonTooltip(
            fields=tip_fields,
            aliases=tip_aliases,
            localize=True,
            sticky=True,
        ),
    ).add_to(group)
    group.add_to(m)
    return group


class LazyH3MapLoader(MacroElement):
    """Load res-10 H3 GeoJSON from sidecar files when a layer is turned on (keeps HTML small)."""

    def __init__(
        self,
        layers: list[dict[str, Any]],
        map_var: str,
        *,
        heatmap: bool = False,
    ) -> None:
        super().__init__()
        self._name = "LazyH3MapLoader"
        self.map_var = map_var
        self.layers_json = json.dumps(layers)
        default_stroke = H3_HEX_HEATMAP_STROKE if heatmap else {
            "color": H3_HEX_STROKE_COLOR,
            "weight": H3_HEX_STROKE_WEIGHT,
            "opacity": H3_HEX_STROKE_OPACITY,
            "fillOpacity": H3_HEX_FILL_OPACITY,
        }
        self._template = Template(
            """
        {% macro script(this, kwargs) %}
        (function() {
            var mapRef = {{ this.map_var }};
            var layers = {{ this.layers_json }};
            var loaded = {};
            var geoLayers = {};
            var STROKE = {{ this.stroke_json }};

            function hexToRgb(hex) {
                var h = hex.replace("#", "");
                return [
                    parseInt(h.substring(0, 2), 16),
                    parseInt(h.substring(2, 4), 16),
                    parseInt(h.substring(4, 6), 16),
                ];
            }
            function mixHex(colors, t) {
                t = Math.max(0, Math.min(1, t));
                var idx = t * (colors.length - 1);
                var lo = Math.floor(idx);
                var hi = Math.min(colors.length - 1, lo + 1);
                var frac = idx - lo;
                var c0 = hexToRgb(colors[lo]);
                var c1 = hexToRgb(colors[hi]);
                return "#" + [0, 1, 2].map(function(i) {
                    var v = Math.round(c0[i] + (c1[i] - c0[i]) * frac);
                    var s = v.toString(16);
                    return s.length === 1 ? "0" + s : s;
                }).join("");
            }
            function paletteFor(meta) {
                var light = window.edwaH3ColorMode === "light";
                if (light && meta.colorsLight) {
                    return {
                        colors: meta.colorsLight,
                        gamma: meta.colorGammaLight != null ? meta.colorGammaLight : 1.0,
                        stroke: meta.strokeLight || STROKE,
                    };
                }
                return {
                    colors: meta.colorsHeat || meta.colors,
                    gamma: meta.colorGammaHeat != null ? meta.colorGammaHeat : 1.0,
                    stroke: meta.strokeHeat || STROKE,
                };
            }
            function normalizedT(meta, value) {
                if (!isFinite(value)) {
                    value = meta.vmin != null ? meta.vmin : 0;
                }
                var lo = meta.vmin, hi = meta.vmax;
                var floor = meta.colorFloor != null ? meta.colorFloor : 0;
                var blueCap = meta.blueCap;
                var blueSpan = meta.blueSpan != null ? meta.blueSpan : 0.3;
                var t = 0;
                if (hi <= lo) {
                    return Math.max(0, Math.min(1, floor));
                }
                if (blueCap != null && blueCap > lo && value <= blueCap) {
                    t = ((value - lo) / (blueCap - lo)) * blueSpan;
                } else {
                    var base = blueCap != null && blueCap > lo ? blueCap : lo;
                    var spanStart = blueCap != null && blueCap > lo ? blueSpan : 0;
                    var x = Math.max(0, value - base);
                    var denom = hi - base;
                    var tHi = 0;
                    if (denom > 0) {
                        if (meta.scale === "log1p") {
                            tHi = Math.log1p(x) / Math.log1p(denom);
                        } else {
                            tHi = x / denom;
                        }
                    }
                    t = spanStart + (1 - spanStart) * Math.max(0, Math.min(1, tHi));
                }
                t = Math.max(0, Math.min(1, t));
                if (floor > 0) {
                    t = floor + (1 - floor) * t;
                }
                return t;
            }
            function styleFor(meta) {
                return function(feature) {
                    var value = Number(feature.properties[meta.valueField]);
                    var t = normalizedT(meta, value);
                    var pal = paletteFor(meta);
                    var gamma = pal.gamma;
                    if (gamma > 0 && gamma !== 1.0) {
                        t = Math.pow(Math.max(0, Math.min(1, t)), gamma);
                    }
                    var stroke = pal.stroke;
                    return {
                        fillColor: mixHex(pal.colors, t),
                        color: stroke.color,
                        weight: stroke.weight,
                        opacity: stroke.opacity,
                        fillOpacity: stroke.fillOpacity,
                    };
                };
            }
            function restyleLoadedLayers() {
                Object.keys(geoLayers).forEach(function(id) {
                    var meta = layers.find(function(m) { return m.id === id; });
                    var gj = geoLayers[id];
                    if (!meta || !gj) return;
                    gj.setStyle(styleFor(meta));
                });
            }
            window.edwaH3RestyleAll = restyleLoadedLayers;
            function ensureH3Panes() {
                if (!mapRef.getPane("{{ this.detail_pane }}")) {
                    mapRef.createPane("{{ this.detail_pane }}");
                    mapRef.getPane("{{ this.detail_pane }}").style.zIndex = 420;
                }
                if (!mapRef.getPane("{{ this.comprehensive_pane }}")) {
                    mapRef.createPane("{{ this.comprehensive_pane }}");
                    mapRef.getPane("{{ this.comprehensive_pane }}").style.zIndex = 640;
                }
            }
            function tooltipFor(meta) {
                return function(feature, layer) {
                    var p = feature.properties || {};
                    var lines = [];
                    meta.tooltipFields.forEach(function(field, i) {
                        var label = meta.tooltipAliases[i] || field;
                        var val = p[field];
                        if (val !== undefined && val !== null && String(val).trim() !== "") {
                            lines.push("<b>" + label + ":</b> " + val);
                        }
                    });
                    if (lines.length) {
                        layer.bindTooltip(lines.join("<br>"), {
                            sticky: true,
                            direction: "top",
                            opacity: 0.96,
                        });
                        layer.on("mouseover", function() {
                            layer.openTooltip();
                        });
                        layer.on("mouseout", function() {
                            layer.closeTooltip();
                        });
                    }
                };
            }
            function loadLayer(meta) {
                if (loaded[meta.id]) {
                    var cached = geoLayers[meta.id];
                    if (cached && meta.groupLayer && !meta.groupLayer.hasLayer(cached)) {
                        cached.addTo(meta.groupLayer);
                        requestAnimationFrame(function() {
                            mapRef.invalidateSize({pan: false});
                        });
                    }
                    return Promise.resolve();
                }
                var group = meta.groupLayer;
                return fetch(meta.geojsonUrl).then(function(r) {
                    if (!r.ok) throw new Error("Failed to load " + meta.geojsonUrl);
                    return r.json();
                }).then(function(data) {
                    ensureH3Panes();
                    var gjOpts = {
                        style: styleFor(meta),
                        onEachFeature: tooltipFor(meta),
                    };
                    if (meta.mapPane) {
                        gjOpts.pane = meta.mapPane;
                    }
                    var gj = L.geoJSON(data, gjOpts);
                    geoLayers[meta.id] = gj;
                    loaded[meta.id] = true;
                    if (group && mapRef.hasLayer(group)) {
                        gj.addTo(group);
                        requestAnimationFrame(function() {
                            mapRef.invalidateSize({pan: false});
                        });
                    }
                });
            }
            layers.forEach(function(meta) {
                meta.groupLayer = window[meta.groupVar];
            });
            mapRef.on("overlayadd", function(e) {
                var meta = layers.find(function(m) { return m.groupLayer === e.layer; });
                if (!meta) return;
                loadLayer(meta).catch(function(err) {
                    console.error("H3 layer load failed:", meta.id, err);
                });
            });
            mapRef.on("overlayremove", function(e) {
                var meta = layers.find(function(m) { return m.groupLayer === e.layer; });
                if (!meta) return;
                var gj = geoLayers[meta.id];
                if (gj && meta.groupLayer && meta.groupLayer.hasLayer(gj)) {
                    meta.groupLayer.removeLayer(gj);
                }
            });
            mapRef.whenReady(function() {
                window.edwaH3ColorMode = window.edwaH3ColorMode || "dark";
                ensureH3Panes();
                layers.forEach(function(meta) {
                    if (meta.groupLayer && mapRef.hasLayer(meta.groupLayer)) {
                        loadLayer(meta).catch(function(err) {
                            console.error("H3 layer load failed:", meta.id, err);
                        });
                    }
                });
            });
        })();
        {% endmacro %}
        """
        )
        self.detail_pane = H3_DETAIL_MAP_PANE
        self.comprehensive_pane = H3_COMPREHENSIVE_MAP_PANE
        self.stroke_json = json.dumps(default_stroke)


class LazyRecreationLandsLoader(MacroElement):
    """Load recreation land/POI GeoJSON sidecars when a layer is turned on."""

    def __init__(self, layers: list[dict[str, Any]], map_var: str) -> None:
        super().__init__()
        self._name = "LazyRecreationLandsLoader"
        self.map_var = map_var
        self.layers_json = json.dumps(layers)
        self._template = Template(
            """
        {% macro script(this, kwargs) %}
        (function() {
            var mapRef = {{ this.map_var }};
            var layers = {{ this.layers_json }};
            var loaded = {};
            var geoLayers = {};

            function styleFor(meta) {
                return function() {
                    return meta.style;
                };
            }
            function tooltipFor(meta) {
                return function(feature, layer) {
                    var p = feature.properties || {};
                    var lines = [];
                    meta.tooltipFields.forEach(function(field, i) {
                        var label = meta.tooltipAliases[i] || field;
                        var val = p[field];
                        if (val !== undefined && val !== null && String(val).trim() !== "") {
                            lines.push("<b>" + label + ":</b> " + val);
                        }
                    });
                    if (lines.length) {
                        layer.bindTooltip(lines.join("<br>"), {sticky: true});
                    }
                };
            }
            function loadLayer(meta) {
                if (loaded[meta.id]) {
                    var cached = geoLayers[meta.id];
                    if (cached && meta.groupLayer && !meta.groupLayer.hasLayer(cached)) {
                        cached.addTo(meta.groupLayer);
                    }
                    return Promise.resolve();
                }
                var group = meta.groupLayer;
                return fetch(meta.geojsonUrl).then(function(r) {
                    if (!r.ok) throw new Error("Failed to load " + meta.geojsonUrl);
                    return r.json();
                }).then(function(data) {
                    var gj = L.geoJSON(data, {
                        style: styleFor(meta),
                        onEachFeature: tooltipFor(meta),
                    });
                    geoLayers[meta.id] = gj;
                    loaded[meta.id] = true;
                    if (group && mapRef.hasLayer(group)) {
                        gj.addTo(group);
                    }
                });
            }
            layers.forEach(function(meta) {
                meta.groupLayer = window[meta.groupVar];
            });
            mapRef.on("overlayadd", function(e) {
                var meta = layers.find(function(m) { return m.groupLayer === e.layer; });
                if (!meta) return;
                loadLayer(meta).catch(function(err) {
                    console.error("Recreation lands layer load failed:", meta.id, err);
                });
            });
            mapRef.on("overlayremove", function(e) {
                var meta = layers.find(function(m) { return m.groupLayer === e.layer; });
                if (!meta) return;
                var gj = geoLayers[meta.id];
                if (gj && meta.groupLayer && meta.groupLayer.hasLayer(gj)) {
                    meta.groupLayer.removeLayer(gj);
                }
            });
            mapRef.whenReady(function() {
                layers.forEach(function(meta) {
                    if (meta.groupLayer && mapRef.hasLayer(meta.groupLayer)) {
                        loadLayer(meta).catch(function(err) {
                            console.error("Recreation lands layer load failed:", meta.id, err);
                        });
                    }
                });
            });
        })();
        {% endmacro %}
        """
        )


def add_overlay_bulk_toggle_controls(m: folium.Map, *, label: str = "All layers") -> None:
    """Add a master checkbox above flat overlay layers (standard LayerControl)."""
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
            .edwa-select-all-label {
                display: block;
                padding-bottom: 4px;
                margin-bottom: 2px;
                border-bottom: 1px solid #ddd;
                font-weight: 600;
            }
            .edwa-select-all-label span span {
                font-weight: 600;
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        function edwaLeafLayerInputs(root) {
            return Array.prototype.slice.call(
                root.querySelectorAll("input.leaflet-control-layers-selector")
            ).filter(function(input) {
                return input.layerId !== undefined;
            });
        }
        function edwaOverlayInputs(section) {
            return edwaLeafLayerInputs(section);
        }
        function edwaInstallMasterLayerCheckbox() {
            if (document.getElementById("edwa-layer-select-all")) return;
            var list = document.querySelector(".leaflet-control-layers-list");
            var section = document.querySelector(".leaflet-control-layers-overlays");
            if (!list || !section) return;

            var row = document.createElement("label");
            row.className = "edwa-select-all-label";
            row.innerHTML = '<span><input type="checkbox" id="edwa-layer-select-all" '
                + 'class="edwa-master-layer-toggle" />'
                + '<span>{{ this.label }}</span></span>';
            list.insertBefore(row, section);

            var master = document.getElementById("edwa-layer-select-all");
            var syncing = false;

            function syncMasterFromChildren() {
                if (syncing) return;
                var inputs = edwaOverlayInputs(section);
                if (!inputs.length) return;
                var checked = inputs.filter(function(i) { return i.checked; }).length;
                syncing = true;
                master.indeterminate = checked > 0 && checked < inputs.length;
                master.checked = checked === inputs.length;
                syncing = false;
            }

            master.addEventListener("change", function() {
                if (syncing) return;
                var on = master.checked;
                syncing = true;
                master.indeterminate = false;
                edwaOverlayInputs(section).forEach(function(input) {
                    if (input.checked !== on) input.click();
                });
                syncing = false;
            });

            section.addEventListener("change", syncMasterFromChildren);
            syncMasterFromChildren();
        }
        document.addEventListener("DOMContentLoaded", function() {
            edwaInstallMasterLayerCheckbox();
            setTimeout(edwaInstallMasterLayerCheckbox, 300);
        });
        {% endmacro %}
        """
    )
    bulk = MacroElement()
    bulk._template = template
    bulk.label = label
    m.get_root().add_child(bulk)


def add_layer_search_filter(
    m: folium.Map,
    *,
    search_placeholder: str = "Filter layers",
) -> None:
    """Text filter for overlay entries in the layer control."""
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
            .edwa-layer-search-wrap {
                display: block;
                padding: 6px 8px 8px;
                margin-bottom: 4px;
                border-bottom: 1px solid #ddd;
            }
            .edwa-layer-search {
                width: 100%;
                box-sizing: border-box;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                padding: 5px 7px;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
            }
            .edwa-layer-search:focus {
                outline: none;
                border-color: #6e6e6e;
            }
            .edwa-layer-bulk-actions {
                display: flex;
                gap: 10px;
                margin-top: 6px;
            }
            .edwa-layer-bulk-btn {
                background: none;
                border: none;
                padding: 0;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                color: #57606a;
                cursor: pointer;
                text-decoration: underline;
                text-underline-offset: 2px;
            }
            .edwa-layer-bulk-btn:hover {
                color: #24292f;
            }
            .leaflet-control-layers-expanded .leaflet-control-layers-list {
                max-height: min(70vh, 520px);
                overflow-y: auto;
                overflow-x: hidden;
            }
            .leaflet-control-layers label {
                display: block;
                white-space: normal;
                word-break: break-word;
            }
            label.edwa-layer-hidden {
                display: none !important;
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        function edwaOverlayInputs(section) {
            return Array.prototype.slice.call(
                section.querySelectorAll("input.leaflet-control-layers-selector")
            ).filter(function(input) {
                return input.layerId !== undefined;
            });
        }
        function edwaLayerLabel(input) {
            var row = input.closest("label") || input.parentElement;
            return row ? (row.textContent || "").trim() : "";
        }
        function edwaInstallLayerSearchFilter() {
            var list = document.querySelector(".leaflet-control-layers-list");
            var section = document.querySelector(".leaflet-control-layers-overlays");
            if (!list || !section || document.getElementById("edwa-layer-search")) return;

            var wrap = document.createElement("div");
            wrap.className = "edwa-layer-search-wrap";
            wrap.innerHTML = '<input type="search" id="edwa-layer-search" class="edwa-layer-search" '
                + 'placeholder="{{ this.search_placeholder }}" autocomplete="off" />'
                + '<div class="edwa-layer-bulk-actions">'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-select-all">Select all</button>'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-clear-all">Clear all</button>'
                + '</div>';
            list.insertBefore(wrap, section);

            var search = document.getElementById("edwa-layer-search");
            var selectAllBtn = document.getElementById("edwa-layer-select-all");
            var clearAllBtn = document.getElementById("edwa-layer-clear-all");
            var syncing = false;

            function isVisibleInput(input) {
                var row = input.closest("label");
                return row && !row.classList.contains("edwa-layer-hidden");
            }

            function visibleInputs() {
                return edwaOverlayInputs(section).filter(isVisibleInput);
            }

            function setInputsChecked(inputs, on) {
                syncing = true;
                inputs.forEach(function(input) {
                    if (input.checked !== on) input.click();
                });
                syncing = false;
            }

            function applySearchFilter() {
                var q = (search.value || "").trim().toLowerCase();
                edwaOverlayInputs(section).forEach(function(input) {
                    var row = input.closest("label");
                    if (!row) return;
                    var text = edwaLayerLabel(input).toLowerCase();
                    if (!q || text.indexOf(q) !== -1) {
                        row.classList.remove("edwa-layer-hidden");
                    } else {
                        row.classList.add("edwa-layer-hidden");
                    }
                });
            }

            search.addEventListener("input", applySearchFilter);

            selectAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                setInputsChecked(visibleInputs(), true);
            });

            clearAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                setInputsChecked(edwaOverlayInputs(section).filter(function(i) { return i.checked; }), false);
            });
        }
        document.addEventListener("DOMContentLoaded", function() {
            edwaInstallLayerSearchFilter();
            setTimeout(edwaInstallLayerSearchFilter, 300);
            setTimeout(edwaInstallLayerSearchFilter, 800);
        });
        {% endmacro %}
        """
    )
    picker = MacroElement()
    picker._template = template
    picker.search_placeholder = search_placeholder
    m.get_root().add_child(picker)


add_layer_search_and_exclusive_select = add_layer_search_filter


def add_featured_overlay_layer_panel(
    m: folium.Map,
    *,
    search_placeholder: str = "Filter layers",
    featured_matcher: str = r"all recreation",
    featured_title: str = "All recreation (deduplicated)",
    section_title: str = "By activity / land",
) -> None:
    """Split flat overlay list into a featured top block plus searchable activity list."""
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
            .edwa-layer-search-wrap {
                display: block;
                padding: 6px 8px 8px;
                margin-bottom: 4px;
                border-bottom: 1px solid #ddd;
            }
            .edwa-panel-section-title {
                display: block;
                margin: 8px 0 4px;
                font: 11px/1.35 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                font-weight: 600;
                color: #57606a;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            .edwa-featured-layers label {
                display: block;
                font-weight: 600;
                margin: 2px 0;
                white-space: normal;
                word-break: break-word;
            }
            .edwa-layer-search {
                width: 100%;
                box-sizing: border-box;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                padding: 5px 7px;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
            }
            .edwa-layer-bulk-actions {
                display: flex;
                gap: 10px;
                margin-top: 6px;
            }
            .edwa-layer-bulk-btn {
                background: none;
                border: none;
                padding: 0;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                color: #57606a;
                cursor: pointer;
                text-decoration: underline;
                text-underline-offset: 2px;
            }
            .edwa-layer-bulk-btn:hover {
                color: #24292f;
            }
            .leaflet-control-layers-expanded .leaflet-control-layers-list {
                max-height: min(70vh, 520px);
                overflow-y: auto;
                overflow-x: hidden;
            }
            .leaflet-control-layers label {
                display: block;
                white-space: normal;
                word-break: break-word;
            }
            label.edwa-layer-hidden {
                display: none !important;
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        function edwaFeaturedOverlayInputs(section) {
            return Array.prototype.slice.call(
                section.querySelectorAll("input.leaflet-control-layers-selector")
            ).filter(function(input) {
                return input.layerId !== undefined;
            });
        }
        function edwaFeaturedLayerLabel(input) {
            var row = input.closest("label") || input.parentElement;
            return row ? (row.textContent || "").trim() : "";
        }
        function edwaInstallFeaturedOverlayPanel() {
            var list = document.querySelector(".leaflet-control-layers-list");
            var section = document.querySelector(".leaflet-control-layers-overlays");
            if (!list || !section || document.getElementById("edwa-layer-search")) return;

            var wrap = document.createElement("div");
            wrap.className = "edwa-layer-search-wrap";
            wrap.innerHTML = ''
                + '<span class="edwa-panel-section-title">{{ this.featured_title }}</span>'
                + '<div class="edwa-featured-layers" id="edwa-featured-layers"></div>'
                + '<span class="edwa-panel-section-title">{{ this.section_title }}</span>'
                + '<input type="search" id="edwa-layer-search" class="edwa-layer-search" '
                + 'placeholder="{{ this.search_placeholder }}" autocomplete="off" />'
                + '<div class="edwa-layer-bulk-actions">'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-select-all">Select all</button>'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-clear-all">Clear all</button>'
                + '</div>';
            list.insertBefore(wrap, section);

            var featuredHost = document.getElementById("edwa-featured-layers");
            var featuredPattern = new RegExp({{ this.featured_matcher | tojson }}, "i");
            Array.prototype.slice.call(section.querySelectorAll("label")).forEach(function(label) {
                if (featuredPattern.test((label.textContent || "").trim())) {
                    featuredHost.appendChild(label);
                }
            });

            var search = document.getElementById("edwa-layer-search");
            var selectAllBtn = document.getElementById("edwa-layer-select-all");
            var clearAllBtn = document.getElementById("edwa-layer-clear-all");
            var syncing = false;

            function activityInputs() {
                return edwaFeaturedOverlayInputs(section);
            }

            function featuredInputs() {
                return Array.prototype.slice.call(
                    featuredHost.querySelectorAll("input.leaflet-control-layers-selector")
                );
            }

            function allInputs() {
                return featuredInputs().concat(activityInputs());
            }

            function isVisibleActivityInput(input) {
                var row = input.closest("label");
                return row && !row.classList.contains("edwa-layer-hidden");
            }

            function setInputsChecked(inputs, on) {
                syncing = true;
                inputs.forEach(function(input) {
                    if (input.checked !== on) input.click();
                });
                syncing = false;
            }

            function applySearchFilter() {
                var q = (search.value || "").trim().toLowerCase();
                activityInputs().forEach(function(input) {
                    var row = input.closest("label");
                    if (!row) return;
                    var text = edwaFeaturedLayerLabel(input).toLowerCase();
                    if (!q || text.indexOf(q) !== -1) {
                        row.classList.remove("edwa-layer-hidden");
                    } else {
                        row.classList.add("edwa-layer-hidden");
                    }
                });
            }

            search.addEventListener("input", applySearchFilter);
            selectAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                var targets = activityInputs().filter(isVisibleActivityInput).concat(featuredInputs());
                setInputsChecked(targets, true);
            });
            clearAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                setInputsChecked(allInputs().filter(function(i) { return i.checked; }), false);
            });
        }
        document.addEventListener("DOMContentLoaded", function() {
            edwaInstallFeaturedOverlayPanel();
            setTimeout(edwaInstallFeaturedOverlayPanel, 300);
            setTimeout(edwaInstallFeaturedOverlayPanel, 800);
        });
        {% endmacro %}
        """
    )
    panel = MacroElement()
    panel._template = template
    panel.search_placeholder = search_placeholder
    panel.featured_matcher = featured_matcher
    panel.featured_title = featured_title
    panel.section_title = section_title
    m.get_root().add_child(panel)


def add_h3_map_layer_controls(
    m: folium.Map,
    *,
    map_var: str,
    tile_var: str,
    search_placeholder: str = "Filter layers",
    basemap_dark_url: str = CARTO_DARK_MATTER_LABELS_TILES,
    basemap_light_url: str = CARTO_VOYAGER_LABELS_TILES,
) -> None:
    """H3 layer panel: search, basemap toggle, all_recreation block, then activity layers."""
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
            .edwa-layer-search-wrap {
                display: block;
                padding: 6px 8px 8px;
                margin-bottom: 4px;
                border-bottom: 1px solid #ddd;
            }
            .edwa-layer-search {
                width: 100%;
                box-sizing: border-box;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                padding: 5px 7px;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
            }
            .edwa-basemap-toggle {
                display: flex;
                gap: 4px;
                margin: 0 0 8px;
            }
            .edwa-layer-search {
                margin-top: 4px;
            }
            .edwa-basemap-btn {
                flex: 1;
                font: 11px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                padding: 5px 6px;
                border: 1px solid #c9d1d9;
                border-radius: 4px;
                background: #f6f8fa;
                color: #24292f;
                cursor: pointer;
            }
            .edwa-basemap-btn.edwa-active {
                background: #24292f;
                color: #fff;
                border-color: #24292f;
            }
            .edwa-panel-section-title {
                display: block;
                margin: 8px 0 4px;
                font: 11px/1.35 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                font-weight: 600;
                color: #57606a;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            .edwa-all-recreation-layers label {
                display: block;
                font-weight: 600;
                margin: 2px 0;
            }
            .edwa-layer-bulk-actions {
                display: flex;
                gap: 10px;
                margin-top: 6px;
            }
            .edwa-layer-bulk-btn {
                background: none;
                border: none;
                padding: 0;
                font: 12px/1.3 -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                color: #57606a;
                cursor: pointer;
                text-decoration: underline;
            }
            .leaflet-control-layers-expanded .leaflet-control-layers-list {
                max-height: min(70vh, 520px);
                overflow-y: auto;
                overflow-x: hidden;
            }
            .leaflet-control-layers label {
                display: block;
                white-space: normal;
                word-break: break-word;
            }
            label.edwa-layer-hidden {
                display: none !important;
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        function edwaH3SetBasemap(mode) {
            window.edwaH3ColorMode = mode;
            var tile = window["{{ this.tile_var }}"];
            if (tile && tile.setUrl) {
                tile.setUrl(mode === "light" ? "{{ this.basemap_light_url }}" : "{{ this.basemap_dark_url }}");
            }
            var darkBtn = document.getElementById("edwa-basemap-dark");
            var lightBtn = document.getElementById("edwa-basemap-light");
            if (darkBtn && lightBtn) {
                darkBtn.classList.toggle("edwa-active", mode === "dark");
                lightBtn.classList.toggle("edwa-active", mode === "light");
            }
            if (window.edwaH3RestyleAll) window.edwaH3RestyleAll();
        }
        function edwaOverlayInputs(section) {
            return Array.prototype.slice.call(
                section.querySelectorAll("input.leaflet-control-layers-selector")
            ).filter(function(input) {
                return input.layerId !== undefined;
            });
        }
        function edwaLayerLabel(input) {
            var row = input.closest("label") || input.parentElement;
            return row ? (row.textContent || "").trim() : "";
        }
        function edwaInstallH3LayerPanel() {
            var list = document.querySelector(".leaflet-control-layers-list");
            var section = document.querySelector(".leaflet-control-layers-overlays");
            if (!list || !section || document.getElementById("edwa-layer-search")) return;

            var wrap = document.createElement("div");
            wrap.className = "edwa-layer-search-wrap";
            wrap.innerHTML = ''
                + '<div class="edwa-basemap-toggle">'
                + '<button type="button" class="edwa-basemap-btn edwa-active" id="edwa-basemap-dark">Dark heatmap</button>'
                + '<button type="button" class="edwa-basemap-btn" id="edwa-basemap-light">Light map</button>'
                + '</div>'
                + '<span class="edwa-panel-section-title">All recreation (deduplicated)</span>'
                + '<div class="edwa-all-recreation-layers" id="edwa-all-recreation-layers"></div>'
                + '<span class="edwa-panel-section-title">By activity / land</span>'
                + '<input type="search" id="edwa-layer-search" class="edwa-layer-search" '
                + 'placeholder="{{ this.search_placeholder }}" autocomplete="off" />'
                + '<div class="edwa-layer-bulk-actions">'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-select-all">Select all</button>'
                + '<button type="button" class="edwa-layer-bulk-btn" id="edwa-layer-clear-all">Clear all</button>'
                + '</div>';
            list.insertBefore(wrap, section);

            var allRecHost = document.getElementById("edwa-all-recreation-layers");
            Array.prototype.slice.call(section.querySelectorAll("label")).forEach(function(label) {
                if (/all recreation/i.test((label.textContent || "").trim())) {
                    allRecHost.appendChild(label);
                }
            });

            window.edwaH3ColorMode = "dark";
            document.getElementById("edwa-basemap-dark").addEventListener("click", function() {
                edwaH3SetBasemap("dark");
            });
            document.getElementById("edwa-basemap-light").addEventListener("click", function() {
                edwaH3SetBasemap("light");
            });

            var search = document.getElementById("edwa-layer-search");
            var selectAllBtn = document.getElementById("edwa-layer-select-all");
            var clearAllBtn = document.getElementById("edwa-layer-clear-all");
            var syncing = false;

            function activityInputs() {
                return edwaOverlayInputs(section);
            }

            function allRecreationInputs() {
                return Array.prototype.slice.call(
                    allRecHost.querySelectorAll("input.leaflet-control-layers-selector")
                );
            }

            function allToggleInputs() {
                return allRecreationInputs().concat(activityInputs());
            }

            function isVisibleActivityInput(input) {
                var row = input.closest("label");
                return row && !row.classList.contains("edwa-layer-hidden");
            }

            function setInputsChecked(inputs, on) {
                syncing = true;
                inputs.forEach(function(input) {
                    if (input.checked !== on) input.click();
                });
                syncing = false;
            }

            function applySearchFilter() {
                var q = (search.value || "").trim().toLowerCase();
                activityInputs().forEach(function(input) {
                    var row = input.closest("label");
                    if (!row) return;
                    var text = edwaLayerLabel(input).toLowerCase();
                    if (!q || text.indexOf(q) !== -1) {
                        row.classList.remove("edwa-layer-hidden");
                    } else {
                        row.classList.add("edwa-layer-hidden");
                    }
                });
            }

            search.addEventListener("input", applySearchFilter);
            selectAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                var targets = activityInputs().filter(isVisibleActivityInput).concat(allRecreationInputs());
                setInputsChecked(targets, true);
            });
            clearAllBtn.addEventListener("click", function(ev) {
                ev.preventDefault();
                setInputsChecked(allToggleInputs().filter(function(i) { return i.checked; }), false);
            });
        }
        document.addEventListener("DOMContentLoaded", function() {
            edwaInstallH3LayerPanel();
            setTimeout(edwaInstallH3LayerPanel, 300);
            setTimeout(edwaInstallH3LayerPanel, 800);
        });
        {% endmacro %}
        """
    )
    panel = MacroElement()
    panel._template = template
    panel.search_placeholder = search_placeholder
    panel.tile_var = tile_var
    panel.basemap_dark_url = basemap_dark_url
    panel.basemap_light_url = basemap_light_url
    m.get_root().add_child(panel)


def add_grouped_layer_toggle_controls(m: folium.Map, *, label: str = "All layers") -> None:
    """Master + per-quarter checkboxes for GroupedLayerControl (groupCheckboxes=True)."""
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>
            .edwa-select-all-label {
                display: block;
                padding-bottom: 4px;
                margin-bottom: 4px;
                border-bottom: 1px solid #ddd;
                font-weight: 600;
            }
            .edwa-select-all-label span span,
            .leaflet-control-layers-group-label span.leaflet-control-layers-group-name {
                font-weight: 600;
            }
            .leaflet-control-layers-group {
                padding-top: 4px;
                margin-top: 2px;
                border-top: 1px solid #eee;
            }
            .leaflet-control-layers-group:first-child {
                border-top: none;
                margin-top: 0;
                padding-top: 0;
            }
            .leaflet-control-layers-group.edwa-quarter-collapsed > label:not(.leaflet-control-layers-group-label) {
                display: none;
            }
            .leaflet-control-layers-group-label {
                cursor: default;
            }
            .leaflet-control-layers-group-name.edwa-quarter-toggle {
                cursor: pointer;
                user-select: none;
            }
            .leaflet-control-layers-group-name.edwa-quarter-toggle::before {
                display: inline-block;
                width: 1.1em;
                color: #57606a;
            }
            .leaflet-control-layers-group.edwa-quarter-collapsed .edwa-quarter-toggle::before {
                content: "▸";
            }
            .leaflet-control-layers-group:not(.edwa-quarter-collapsed) .edwa-quarter-toggle::before {
                content: "▾";
            }
            .leaflet-control-layers-expanded .leaflet-control-layers-list {
                max-height: min(70vh, 520px);
                overflow-y: auto;
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        function edwaLeafLayerInputs(root) {
            return Array.prototype.slice.call(
                root.querySelectorAll("input.leaflet-control-layers-selector")
            ).filter(function(input) {
                return input.layerId !== undefined;
            });
        }
        function edwaGroupedLeafInputs(groupEl) {
            return edwaLeafLayerInputs(groupEl);
        }
        function edwaFlatLeafInputs(section) {
            return edwaLeafLayerInputs(section);
        }
        function edwaSyncGroupMaster(groupEl) {
            var master = groupEl.querySelector("input.leaflet-control-layers-group-selector");
            if (!master) return;
            var inputs = edwaGroupedLeafInputs(groupEl);
            if (!inputs.length) return;
            var checked = inputs.filter(function(i) { return i.checked; }).length;
            master.indeterminate = checked > 0 && checked < inputs.length;
            master.checked = checked === inputs.length;
        }
        function edwaInitQuarterCollapsibles(section) {
            var groups = section.querySelectorAll(".leaflet-control-layers-group");
            Array.prototype.forEach.call(groups, function(group) {
                if (group.getAttribute("data-edwa-collapse-ready")) return;
                group.setAttribute("data-edwa-collapse-ready", "1");
                group.classList.add("edwa-quarter-collapsed");
                var nameSpan = group.querySelector(".leaflet-control-layers-group-name");
                if (!nameSpan) return;
                nameSpan.classList.add("edwa-quarter-toggle");
                nameSpan.setAttribute("role", "button");
                nameSpan.setAttribute("tabindex", "0");
                nameSpan.setAttribute("aria-expanded", "false");
                nameSpan.title = "Click to expand or collapse activities";
                function toggleQuarter(ev) {
                    if (ev) {
                        ev.preventDefault();
                        ev.stopPropagation();
                    }
                    var collapsed = group.classList.toggle("edwa-quarter-collapsed");
                    nameSpan.setAttribute("aria-expanded", collapsed ? "false" : "true");
                }
                nameSpan.addEventListener("click", toggleQuarter);
                nameSpan.addEventListener("keydown", function(ev) {
                    if (ev.key === "Enter" || ev.key === " ") toggleQuarter(ev);
                });
            });
        }
        function edwaInstallGroupedLayerToggles() {
            var list = document.querySelector(".leaflet-control-layers-list");
            var section = document.querySelector(".leaflet-control-layers-overlays");
            if (!list || !section) return;

            if (!document.getElementById("edwa-layer-select-all")) {
                var row = document.createElement("label");
                row.className = "edwa-select-all-label";
                row.innerHTML = '<span><input type="checkbox" id="edwa-layer-select-all" '
                    + 'class="edwa-master-layer-toggle" />'
                    + '<span>{{ this.label }}</span></span>';
                list.insertBefore(row, section);
            }

            var master = document.getElementById("edwa-layer-select-all");
            var syncing = false;
            var groups = section.querySelectorAll(".leaflet-control-layers-group");

            function allLeafInputs() {
                if (groups.length) {
                    var out = [];
                    Array.prototype.forEach.call(groups, function(g) {
                        out = out.concat(edwaGroupedLeafInputs(g));
                    });
                    return out;
                }
                return edwaFlatLeafInputs(section);
            }

            function syncMasterFromLeaves() {
                if (syncing) return;
                var inputs = allLeafInputs();
                if (!inputs.length) return;
                var checked = inputs.filter(function(i) { return i.checked; }).length;
                syncing = true;
                master.indeterminate = checked > 0 && checked < inputs.length;
                master.checked = checked === inputs.length;
                syncing = false;
                Array.prototype.forEach.call(groups, edwaSyncGroupMaster);
            }

            master.addEventListener("change", function() {
                if (syncing) return;
                var on = master.checked;
                syncing = true;
                master.indeterminate = false;
                allLeafInputs().forEach(function(input) {
                    if (input.checked !== on) input.click();
                });
                Array.prototype.forEach.call(groups, function(g) {
                    var gm = g.querySelector("input.leaflet-control-layers-group-selector");
                    if (gm) {
                        gm.indeterminate = false;
                        gm.checked = on;
                    }
                });
                syncing = false;
            });

            section.addEventListener("change", function(ev) {
                if (syncing) return;
                if (ev.target === master) return;
                if (ev.target.classList.contains("leaflet-control-layers-group-selector")) {
                    syncing = true;
                    setTimeout(function() {
                        syncing = false;
                        syncMasterFromLeaves();
                    }, 0);
                    return;
                }
                syncMasterFromLeaves();
            });

            edwaInitQuarterCollapsibles(section);
            syncMasterFromLeaves();
        }
        document.addEventListener("DOMContentLoaded", function() {
            edwaInstallGroupedLayerToggles();
            setTimeout(edwaInstallGroupedLayerToggles, 300);
            setTimeout(edwaInstallGroupedLayerToggles, 800);
        });
        {% endmacro %}
        """
    )
    bulk = MacroElement()
    bulk._template = template
    bulk.label = label
    m.get_root().add_child(bulk)


def slim_gdf_for_web(
    gdf: gpd.GeoDataFrame,
    *,
    keep_columns: Sequence[str],
    simplify_tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """Keep only tooltip fields and optionally simplify linework for smaller HTML."""
    cols = [c for c in keep_columns if c in gdf.columns] + ["geometry"]
    export = gdf[cols].copy()
    if simplify_tolerance is not None and simplify_tolerance > 0:
        export["geometry"] = export.geometry.simplify(
            simplify_tolerance,
            preserve_topology=True,
        )
    return export


def slugify_h3_layer_id(*parts: str) -> str:
    """Filesystem-safe id for sidecar GeoJSON (relative to map HTML)."""
    raw = "_".join(parts)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return slug or "layer"


def write_geojson_sidecar(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gdf_to_geojson_dict(gdf)), encoding="utf-8")


def register_h3_lazy_layer(
    m: folium.Map,
    data_dir: Path,
    gdf: gpd.GeoDataFrame,
    *,
    layer_id: str,
    layer_name: str,
    value_field: str,
    activity: str,
    tooltip_fields: Sequence[tuple[str, str]],
    show: bool = False,
    data_url_prefix: str | None = None,
    comprehensive: bool = False,
    heatmap: bool = False,
) -> dict[str, Any] | None:
    """Write res-10 sidecar GeoJSON and register an empty overlay group for lazy loading."""
    if gdf.empty:
        return None

    keep_columns = list(dict.fromkeys([value_field, *[field for field, _ in tooltip_fields]]))
    export_gdf = slim_gdf_for_web(gdf, keep_columns=keep_columns)
    values = export_gdf[value_field].astype(float).tolist()
    if heatmap:
        vmin, vmax = value_range_for_heatmap(values)
    else:
        vmin, vmax = value_range(values)

    prefix = data_url_prefix or data_dir.name
    geojson_name = f"{layer_id}.json"
    write_geojson_sidecar(export_gdf, data_dir / geojson_name)

    group = FeatureGroup(name=layer_name, show=show)
    group.add_to(m)

    stroke_heat = H3_HEX_HEATMAP_STROKE
    stroke_light = {
        "color": H3_HEX_STROKE_COLOR,
        "weight": H3_HEX_STROKE_WEIGHT,
        "opacity": H3_HEX_STROKE_OPACITY,
        "fillOpacity": H3_HEX_FILL_OPACITY,
    }
    colors_heat = h3_colors_for_activity(activity, heatmap=True)
    colors_light = (
        h3_colors_for_activity(activity, heatmap=True, light=True)
        if heatmap
        else h3_colors_for_activity(activity, heatmap=False)
    )
    meta: dict[str, Any] = {
        "id": layer_id,
        "groupVar": group.get_name(),
        "geojsonUrl": f"{prefix}/{geojson_name}",
        "vmin": vmin,
        "vmax": vmax,
        "colors": colors_heat,
        "colorsHeat": colors_heat,
        "colorsLight": colors_light,
        "valueField": value_field,
        "tooltipFields": [field for field, _ in tooltip_fields],
        "tooltipAliases": [label for _, label in tooltip_fields],
        "comprehensive": comprehensive,
        "strokeHeat": stroke_heat,
        "strokeLight": stroke_light,
    }
    if heatmap:
        meta["colorGammaHeat"] = H3_HEATMAP_COLOR_GAMMA
        meta["colorGammaLight"] = 1.0
        meta["scale"] = H3_HEATMAP_COLOR_SCALE
        meta["colorFloor"] = H3_HEATMAP_COLOR_FLOOR
        meta["blueCap"] = H3_HEATMAP_BLUE_CAP
        meta["blueSpan"] = H3_HEATMAP_BLUE_SPAN
        meta["mapPane"] = H3_COMPREHENSIVE_MAP_PANE if comprehensive else H3_DETAIL_MAP_PANE
    return meta


def register_recreation_lands_lazy_layer(
    m: folium.Map,
    data_dir: Path,
    gdf: gpd.GeoDataFrame,
    *,
    layer_id: str,
    layer_name: str,
    activity: str,
    layer_suffix: str,
    show: bool = False,
    data_url_prefix: str | None = None,
) -> dict[str, Any] | None:
    if gdf.empty:
        return None

    tooltip_fields = recreation_lands_tooltip_fields(gdf)
    keep_columns = [field for field, _ in tooltip_fields]
    simplify = (
        RECREATION_LANDS_SIMPLIFY_LARGE
        if layer_suffix in RECREATION_LANDS_LARGE_SUFFIXES
        else RECREATION_LANDS_SIMPLIFY_DEFAULT
    )
    export_gdf = slim_gdf_for_web(
        gdf,
        keep_columns=keep_columns,
        simplify_tolerance=simplify,
    )

    prefix = data_url_prefix or data_dir.name
    geojson_name = f"{layer_id}.json"
    write_geojson_sidecar(export_gdf, data_dir / geojson_name)

    group = FeatureGroup(name=layer_name, show=show)
    group.add_to(m)

    return {
        "id": layer_id,
        "groupVar": group.get_name(),
        "geojsonUrl": f"{prefix}/{geojson_name}",
        "style": recreation_lands_style_for_activity(activity),
        "tooltipFields": [field for field, _ in tooltip_fields],
        "tooltipAliases": [label for _, label in tooltip_fields],
    }


def gdf_to_geojson_dict(
    gdf: gpd.GeoDataFrame,
    *,
    geometry_precision: float | None = 1e-5,
) -> dict[str, Any]:
    export = gdf.copy()
    for col in export.columns:
        if col == "geometry":
            continue
        if export[col].dtype.name.startswith("datetime"):
            export[col] = export[col].astype(str)
    if geometry_precision is not None:
        export["geometry"] = export.geometry.set_precision(geometry_precision)
    return json.loads(export.to_json())


def save_map(m: folium.Map, output_html: Path) -> Path:
    output_html = output_html.resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    return output_html


def read_gpkg_layer(gpkg_path: Path, layer_name: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def study_area_bounds_from_h3_gpkgs(gpkg_paths: Sequence[Path]) -> list[list[float]]:
    """Bounds around all H3 cells in the provided seasonal GPKGs."""
    frames: list[gpd.GeoDataFrame] = []
    for path in gpkg_paths:
        for layer in list_gpkg_layers(path):
            if is_h3_overview_layer_name(layer):
                continue
            gdf = read_gpkg_layer(path, layer)
            if not gdf.empty:
                frames.append(gdf)
    if not frames:
        return [[39.0, -120.5], [38.5, -121.0]]
    return bounds_with_padding(frames, padding_ratio=0.06)
