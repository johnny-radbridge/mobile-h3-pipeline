"""
Export interactive HTML maps for GitHub Pages (or any static host).

County visitor-origin map::

    source ~/venvs/etl/bin/activate
    cd trail_pathing/final_analysis_pipeline
    python export_web_maps.py county

H3 stint recreation device-hours (default: all season GPKGs in ``outputs/06.../seasons/``,
summed across periods; layer names are activity-only, e.g. ``Trails``)::

    python export_web_maps.py h3

H3 full-day recreation device-hours (all seasonal GPKGs in outputs/10.../seasonal/)::

    python export_web_maps.py h3-full-day

Study delivery GeoPackage (``gis_analysis/uarw_study.gpkg``) — client maps::

    python export_web_maps.py study-h3 --show-on-load
    python export_web_maps.py study-county
    python export_web_maps.py study-recreation

GitHub Pages (all three from ``uarw_study.gpkg``)::

    python export_web_maps.py study-pages --show-on-load

Legacy pipeline preview (seasonal GPKGs under outputs/06)::

    python export_web_maps.py h3-pages --show-on-load

Commit ``docs/`` (or each ``.html`` plus its ``*_data/`` folder) and enable GitHub Pages
from the ``/docs`` folder on ``main``. H3 maps load hex layers when you enable a layer.
Tiles and scripts use public CDNs (no API keys).
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import geopandas as gpd
from folium import LayerControl

from h3_device_hours_core import ALL_RECREATION_ACTIVITY_KEY
from web_map_export import (
    COUNTY_COLOR_RAMPS,
    CARTO_DARK_MATTER_LABELS_TILES,
    CARTO_DARK_MATTER_NOLABELS_TILES,
    CARTO_LIGHT_NOLABELS_TILES,
    CARTO_VOYAGER_TILES,
    DEFAULT_COUNTY_GPKG,
    DEFAULT_COUNTY_HTML,
    DEFAULT_H3_FULL_DAY_GPKG_PREFIX,
    DEFAULT_H3_FULL_DAY_HTML,
    DEFAULT_H3_FULL_DAY_SEASONAL_DIR,
    DEFAULT_H3_GPKG_PREFIX,
    DEFAULT_H3_HTML,
    DEFAULT_H3_SEASONS_DIR,
    DEFAULT_H3_SEASONAL_DIR,
    COUNTY_STUDY_LAYER_PREFIX,
    DEFAULT_UARW_STUDY_GPKG,
    H3_STUDY_LAYER_PREFIX,
    county_tooltip_fields,
    load_study_county_layers,
    load_study_h3_layers,
    load_study_recreation_lands_layers,
    recreation_lands_activity_key,
    register_recreation_lands_lazy_layer,
    period_key_from_gpkg_path,
    period_label_from_gpkg_path,
    H3_TOOLTIP_FIELDS,
    discover_seasonal_gpkgs,
    aggregate_h3_device_hours_across_gpkgs,
    activity_label,
    LazyH3MapLoader,
    LazyRecreationLandsLoader,
    RECREATION_LANDS_DATA_DIR_NAME,
    RECREATION_LANDS_LAYER_PREFIX,
    add_choropleth_layer,
    add_h3_map_layer_controls,
    add_featured_overlay_layer_panel,
    add_layer_search_filter,
    is_comprehensive_h3_activity,
    partition_h3_export_layers,
    register_h3_lazy_layer,
    slugify_h3_layer_id,
    bounds_with_padding,
    list_gpkg_layers,
    make_base_map,
    read_gpkg_layer,
    save_map,
    sort_activity_layers,
    study_area_bounds_from_h3_gpkgs,
)

_PIPELINE_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _PIPELINE_DIR.parent.parent
DOCS_H3_PREVIEW_DIR = _ANALYSIS_ROOT / "docs" / "edwa_h3_preview"
DOCS_COUNTY_PREVIEW_DIR = _ANALYSIS_ROOT / "docs" / "edwa_county_preview"
DOCS_RECREATION_LANDS_DIR = _ANALYSIS_ROOT / "docs" / "edwa_recreation_lands_preview"
H3_PAGES_DATA_DIR_NAME = "h3_device_hours_map_data"
DEFAULT_UARW_STUDY_H3_HTML = DEFAULT_UARW_STUDY_GPKG.with_name("h3_visitation_map.html")
DEFAULT_UARW_STUDY_COUNTY_HTML = DEFAULT_UARW_STUDY_GPKG.with_name("origin_county_map.html")
DEFAULT_UARW_STUDY_RECREATION_HTML = DEFAULT_UARW_STUDY_GPKG.with_name("recreation_lands_map.html")
# Fixed zoom for H3 heatmap (~3 mi on Leaflet scale bar at study-area latitude).
H3_MAP_INITIAL_ZOOM = 12
H3_AGGREGATE_PERIOD_KEY = "all"

COUNTY_VALUE_FIELD = "unique_device_localdate_id_count"
H3_VALUE_FIELD = "device_hours"

def season_label_from_path(gpkg_path: Path) -> str:
    return period_label_from_gpkg_path(gpkg_path)


def default_h3_preview_gpkgs() -> list[Path]:
    """All recreation-season H3 GeoPackages under ``outputs/06.../seasons/``."""
    return discover_seasonal_gpkgs(DEFAULT_H3_SEASONS_DIR, gpkg_prefix=DEFAULT_H3_GPKG_PREFIX)


def h3_layer_visible(
    activity: str,
    default_layer: str | None,
    *,
    show_on_load: bool,
    season: str | None = None,
    primary_season: str | None = None,
) -> bool:
    if not show_on_load:
        return False
    if default_layer == "__all__":
        return True
    label = activity_label(activity)
    if season is None:
        if default_layer is None:
            return activity == ALL_RECREATION_ACTIVITY_KEY
        if " — " in default_layer:
            return default_layer.endswith(f" — {label}")
        return default_layer == label
    full_name = f"{season} — {label}"
    if default_layer is None:
        return season == primary_season and activity == ALL_RECREATION_ACTIVITY_KEY
    if " — " in default_layer:
        return default_layer == full_name
    return default_layer == label and season == primary_season


def build_county_origin_map(
    gpkg_path: Path | None,
    output_html: Path,
    *,
    default_activity: str | None,
    show_layers_on_load: bool = False,
    boundaries_geojson: Path | None = None,
    places_geojson: Path | None = None,
    activity_layers: dict[str, gpd.GeoDataFrame] | None = None,
) -> Path:
    if activity_layers is None:
        if gpkg_path is None:
            raise SystemExit("gpkg_path is required when activity_layers is not provided.")
        layer_names = sort_activity_layers(list_gpkg_layers(gpkg_path))
        if not layer_names:
            raise SystemExit(f"No layers found in {gpkg_path}")
        activity_layers = {
            layer: read_gpkg_layer(gpkg_path, layer)
            for layer in layer_names
        }
    activity_layers = {
        activity: gdf for activity, gdf in activity_layers.items() if not gdf.empty
    }
    if not activity_layers:
        raise SystemExit("No county origin layers with data to export.")

    layer_names = sort_activity_layers(activity_layers.keys())
    bounds = bounds_with_padding(list(activity_layers.values()), padding_ratio=0.02)
    m = make_base_map(bounds, show_info_panel=False, show_county_boundaries=False)

    if default_activity is None:
        if "state_parks" in layer_names:
            default_activity = "state_parks"
        elif "trails" in layer_names:
            default_activity = "trails"
        else:
            default_activity = layer_names[0]

    for activity in layer_names:
        gdf = activity_layers[activity]
        label = activity_label(activity)
        colors = COUNTY_COLOR_RAMPS.get(activity, COUNTY_COLOR_RAMPS["trails"])
        add_choropleth_layer(
            m,
            gdf,
            layer_name=label,
            value_field=COUNTY_VALUE_FIELD,
            colors=colors,
            caption=f"{label}: device-days",
            tooltip_fields=county_tooltip_fields(gdf, COUNTY_VALUE_FIELD),
            show=show_layers_on_load or (activity == default_activity),
            simplify_tolerance=0.02,
        )

    LayerControl(collapsed=False).add_to(m)
    if ALL_RECREATION_ACTIVITY_KEY in layer_names:
        add_featured_overlay_layer_panel(
            m,
            search_placeholder="Filter activities",
            featured_title="All recreation (deduplicated)",
            section_title="By activity / land",
        )
    else:
        add_layer_search_filter(m, search_placeholder="Filter activities")
    return save_map(m, output_html)


def build_uarw_study_county_map(
    gpkg_path: Path,
    output_html: Path,
    *,
    default_activity: str | None = None,
    show_layers_on_load: bool = False,
) -> Path:
    activity_layers = load_study_county_layers(gpkg_path)
    if not activity_layers:
        raise SystemExit(
            f"No {COUNTY_STUDY_LAYER_PREFIX}* layers found in {gpkg_path}. "
            "Expected layers like origin_county_trails."
        )
    return build_county_origin_map(
        None,
        output_html,
        default_activity=default_activity,
        show_layers_on_load=show_layers_on_load,
        activity_layers=activity_layers,
    )


def build_h3_device_hours_map(
    gpkg_paths: list[Path] | None,
    output_html: Path,
    *,
    default_layer: str | None,
    show_layers_on_load: bool = False,
    show_county_boundaries: bool = False,
    boundaries_geojson: Path | None = None,
    places_geojson: Path | None = None,
    basemap_tiles: str | None = None,
    heatmap: bool = True,
    data_dir: Path | None = None,
    aggregate_periods: bool = True,
    initial_zoom: int | None = H3_MAP_INITIAL_ZOOM,
    preloaded_layers: dict[str, gpd.GeoDataFrame] | None = None,
) -> Path:
    if preloaded_layers is None and not gpkg_paths:
        raise SystemExit("At least one --gpkg path is required for h3 export.")

    combined: dict[str, gpd.GeoDataFrame] | None = None
    if preloaded_layers is not None:
        combined = {
            activity: gdf
            for activity, gdf in preloaded_layers.items()
            if not gdf.empty
        }
        aggregate_periods = True
    elif aggregate_periods:
        combined = aggregate_h3_device_hours_across_gpkgs(
            gpkg_paths or [],
            value_field=H3_VALUE_FIELD,
        )

    if combined is not None:
        focus = combined.get(ALL_RECREATION_ACTIVITY_KEY)
        if focus is not None and not focus.empty:
            bounds = bounds_with_padding([focus], padding_ratio=0.05)
        elif combined:
            bounds = bounds_with_padding(list(combined.values()), padding_ratio=0.05)
        else:
            bounds = study_area_bounds_from_h3_gpkgs(gpkg_paths or [])
    else:
        bounds = study_area_bounds_from_h3_gpkgs(gpkg_paths or [])

    boundary_path = boundaries_geojson or places_geojson if show_county_boundaries else None
    if basemap_tiles is None:
        basemap_tiles = CARTO_DARK_MATTER_LABELS_TILES if heatmap else CARTO_VOYAGER_TILES
    if heatmap:
        m, tile_var = make_base_map(
            bounds,
            show_info_panel=False,
            show_county_boundaries=show_county_boundaries,
            boundaries_geojson=boundary_path,
            basemap_tiles=basemap_tiles,
            return_tile_layer=True,
            initial_zoom=initial_zoom,
            prefer_canvas=True,
        )
    else:
        m = make_base_map(
            bounds,
            show_info_panel=False,
            show_county_boundaries=show_county_boundaries,
            boundaries_geojson=boundary_path,
            basemap_tiles=basemap_tiles,
        )
        tile_var = ""

    primary_season = (
        season_label_from_path(gpkg_paths[0])
        if gpkg_paths and not aggregate_periods
        else None
    )
    layer_count = 0
    if data_dir is None:
        data_dir = output_html.parent / f"{output_html.stem}_data"
    else:
        data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.json"):
        stale.unlink(missing_ok=True)
    lazy_layers: list[dict] = []

    def export_activity_layer(
        activity: str,
        gdf: gpd.GeoDataFrame,
        *,
        season: str | None,
        period_key: str,
    ) -> None:
        nonlocal layer_count
        if gdf.empty:
            return
        activity_label_text = activity_label(activity)
        layer_name = (
            activity_label_text
            if season is None
            else f"{season} — {activity_label_text}"
        )
        layer_id = slugify_h3_layer_id(period_key, activity)
        tooltip_fields = [
            field for field in H3_TOOLTIP_FIELDS if field[0] in gdf.columns
        ]
        show_layer = h3_layer_visible(
            activity,
            default_layer,
            show_on_load=show_layers_on_load,
            season=season,
            primary_season=primary_season,
        )
        cfg = register_h3_lazy_layer(
            m,
            data_dir,
            gdf,
            layer_id=layer_id,
            layer_name=layer_name,
            value_field=H3_VALUE_FIELD,
            activity=activity,
            tooltip_fields=tooltip_fields,
            show=show_layer,
            comprehensive=is_comprehensive_h3_activity(activity),
            heatmap=heatmap,
        )
        if cfg is not None:
            lazy_layers.append(cfg)
            layer_count += 1

    if aggregate_periods:
        if combined is None:
            combined = aggregate_h3_device_hours_across_gpkgs(
                gpkg_paths or [],
                value_field=H3_VALUE_FIELD,
            )
        comprehensive, detail = partition_h3_export_layers(combined.keys())
        for activity in comprehensive + detail:
            export_activity_layer(
                activity,
                combined[activity],
                season=None,
                period_key=H3_AGGREGATE_PERIOD_KEY,
            )
    elif gpkg_paths:
        for gpkg_path in gpkg_paths:
            season = season_label_from_path(gpkg_path)
            comprehensive, detail = partition_h3_export_layers(list_gpkg_layers(gpkg_path))
            period_key = period_key_from_gpkg_path(
                gpkg_path,
                gpkg_prefix=DEFAULT_H3_GPKG_PREFIX,
            )
            for activity in comprehensive + detail:
                gdf = read_gpkg_layer(gpkg_path, activity)
                export_activity_layer(
                    activity,
                    gdf,
                    season=season,
                    period_key=period_key,
                )
    else:
        raise SystemExit("No H3 GeoPackage paths or preloaded layers to export.")

    if layer_count == 0:
        raise SystemExit("No H3 layers with data to export.")

    LayerControl(collapsed=False).add_to(m)
    if heatmap:
        add_h3_map_layer_controls(
            m,
            map_var=m.get_name(),
            tile_var=tile_var,
        )
    else:
        add_layer_search_filter(m, search_placeholder="Filter layers")
    if lazy_layers:
        LazyH3MapLoader(lazy_layers, m.get_name(), heatmap=heatmap).add_to(m)
    return save_map(m, output_html)


def build_uarw_study_recreation_lands_map(
    gpkg_path: Path,
    output_html: Path,
    *,
    default_activity: str | None = "state_parks",
    show_layers_on_load: bool = False,
    data_dir: Path | None = None,
) -> Path:
    activity_layers = load_study_recreation_lands_layers(gpkg_path)
    if not activity_layers:
        raise SystemExit(
            f"No {RECREATION_LANDS_LAYER_PREFIX}* layers found in {gpkg_path}."
        )

    layer_names = sort_activity_layers(activity_layers.keys())
    bounds = bounds_with_padding(list(activity_layers.values()), padding_ratio=0.04)
    m = make_base_map(
        bounds,
        show_info_panel=False,
        show_county_boundaries=False,
        basemap_tiles=CARTO_VOYAGER_TILES,
    )

    if default_activity is None:
        default_activity = "state_parks" if "state_parks" in layer_names else layer_names[0]

    if data_dir is None:
        data_dir = output_html.parent / f"{output_html.stem}_data"
    else:
        data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.json"):
        stale.unlink(missing_ok=True)

    lazy_layers: list[dict] = []
    for layer_name in list_gpkg_layers(gpkg_path):
        if not layer_name.startswith(RECREATION_LANDS_LAYER_PREFIX):
            continue
        suffix = layer_name[len(RECREATION_LANDS_LAYER_PREFIX) :]
        activity = recreation_lands_activity_key(suffix)
        gdf = activity_layers.get(activity)
        if gdf is None or gdf.empty:
            continue
        label = activity_label(activity)
        layer_id = slugify_h3_layer_id("lands", activity)
        meta = register_recreation_lands_lazy_layer(
            m,
            data_dir,
            gdf,
            layer_id=layer_id,
            layer_name=label,
            activity=activity,
            layer_suffix=suffix,
            show=show_layers_on_load,
        )
        if meta is not None:
            lazy_layers.append(meta)

    if not lazy_layers:
        raise SystemExit("No recreation lands layers with data to export.")

    LayerControl(collapsed=False).add_to(m)
    add_layer_search_filter(m, search_placeholder="Filter activities")
    LazyRecreationLandsLoader(lazy_layers, m.get_name()).add_to(m)
    return save_map(m, output_html)


def build_uarw_study_h3_map(
    gpkg_path: Path,
    output_html: Path,
    *,
    default_layer: str | None = None,
    show_layers_on_load: bool = False,
    heatmap: bool = True,
    data_dir: Path | None = None,
) -> Path:
    preloaded = load_study_h3_layers(gpkg_path, value_field=H3_VALUE_FIELD)
    if not preloaded:
        raise SystemExit(
            f"No {H3_STUDY_LAYER_PREFIX}* layers found in {gpkg_path}. "
            "Expected layers like h3_device_hours_trails."
        )
    return build_h3_device_hours_map(
        None,
        output_html,
        default_layer=default_layer,
        show_layers_on_load=show_layers_on_load,
        heatmap=heatmap,
        data_dir=data_dir,
        preloaded_layers=preloaded,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export interactive Folium HTML maps for GitHub Pages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    county = sub.add_parser("county", help="Visitor origin county choropleth (multi-layer GPKG).")
    county.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_COUNTY_GPKG,
        help=f"Input GeoPackage (default: {DEFAULT_COUNTY_GPKG.name})",
    )
    county.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COUNTY_HTML,
        help="Output HTML path",
    )
    county.add_argument(
        "--default-activity",
        default=None,
        help="Activity layer key shown on load (e.g. state_parks). Default: state_parks if present.",
    )

    h3 = sub.add_parser("h3", help="H3 device-hour hex layers from one or more seasonal GPKGs.")
    h3.add_argument(
        "--gpkg",
        type=Path,
        action="append",
        dest="gpkg_paths",
        help="Seasonal H3 GeoPackage (repeat for multiple seasons).",
    )
    h3.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_H3_HTML,
        help="Output HTML path",
    )
    h3.add_argument(
        "--default-layer",
        default=None,
        help='Layer to show on load (e.g. "All recreation"). Default: none.',
    )
    h3.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show the default layer on map load (default: all_recreation for primary quarter).",
    )
    h3.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Use Voyager basemap and legacy color ramps instead of dark heatmap styling.",
    )

    h3_full = sub.add_parser(
        "h3-full-day",
        help="H3 full-day recreation device-hours (grouped by quarter).",
    )
    h3_full.add_argument(
        "--gpkg",
        type=Path,
        action="append",
        dest="gpkg_paths",
        help="Seasonal full-day H3 GeoPackage (repeat for multiple seasons).",
    )
    h3_full.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_H3_FULL_DAY_HTML,
        help="Output HTML path",
    )
    h3_full.add_argument(
        "--default-layer",
        default=None,
        help='Layer to show on load. Default: none (quarters collapsed).',
    )
    h3_full.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show the default layer on map load.",
    )

    h3_pages = sub.add_parser(
        "h3-pages",
        help=(
            "Export H3 preview to docs/edwa_h3_preview/ for GitHub Pages "
            f"(index.html + {H3_PAGES_DATA_DIR_NAME}/)."
        ),
    )
    h3_pages.add_argument(
        "--gpkg",
        type=Path,
        action="append",
        dest="gpkg_paths",
        help="Seasonal H3 GeoPackage (repeat for multiple seasons).",
    )
    h3_pages.add_argument(
        "--default-layer",
        default=None,
        help='Layer shown on load (e.g. "Fall 2024 — All recreation").',
    )
    h3_pages.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show the default layer when the map opens.",
    )

    county.add_argument(
        "--boundaries-geojson",
        "--places-geojson",
        type=Path,
        default=None,
        dest="boundaries_geojson",
        help="County boundary GeoJSON (default: map_assets/counties_study_area.geojson).",
    )

    study_h3 = sub.add_parser(
        "study-h3",
        help="H3 visitation map from gis_analysis/uarw_study.gpkg (h3_device_hours_* layers).",
    )
    study_h3.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_UARW_STUDY_GPKG,
        help=f"Study GeoPackage (default: {DEFAULT_UARW_STUDY_GPKG.name})",
    )
    study_h3.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_UARW_STUDY_H3_HTML,
        help="Output HTML path",
    )
    study_h3.add_argument(
        "--default-layer",
        default="All recreation",
        help='Layer shown on load when --show-on-load is set (default: "All recreation").',
    )
    study_h3.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show the default layer when the map opens.",
    )
    study_h3.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Use Voyager basemap instead of dark heatmap styling.",
    )

    study_county = sub.add_parser(
        "study-county",
        help="County-of-origin map from gis_analysis/uarw_study.gpkg (origin_county_* layers).",
    )
    study_county.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_UARW_STUDY_GPKG,
        help=f"Study GeoPackage (default: {DEFAULT_UARW_STUDY_GPKG.name})",
    )
    study_county.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_UARW_STUDY_COUNTY_HTML,
        help="Output HTML path",
    )
    study_county.add_argument(
        "--default-activity",
        default="trails",
        help="Activity layer key shown on load (default: trails).",
    )

    study_pages = sub.add_parser(
        "study-pages",
        help=(
            "Export H3, county, and recreation-lands maps from uarw_study.gpkg to docs/ "
            "for GitHub Pages."
        ),
    )
    study_pages.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_UARW_STUDY_GPKG,
        help=f"Study GeoPackage (default: {DEFAULT_UARW_STUDY_GPKG.name})",
    )
    study_pages.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show default layers when each map opens.",
    )

    study_recreation = sub.add_parser(
        "study-recreation",
        help="Recreation lands/POI map from uarw_study.gpkg (recreation_lands_pois_* layers).",
    )
    study_recreation.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_UARW_STUDY_GPKG,
        help=f"Study GeoPackage (default: {DEFAULT_UARW_STUDY_GPKG.name})",
    )
    study_recreation.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_UARW_STUDY_RECREATION_HTML,
        help="Output HTML path",
    )
    study_recreation.add_argument(
        "--default-activity",
        default="state_parks",
        help="Activity layer shown on load with --show-on-load (default: state_parks).",
    )
    study_recreation.add_argument(
        "--show-on-load",
        action="store_true",
        help="Show the default activity layer when the map opens.",
    )

    return parser.parse_args()


def export_study_pages(
    gpkg_path: Path,
    *,
    show_on_load: bool = False,
) -> tuple[Path, Path, Path]:
    gpkg_path = gpkg_path.resolve()
    if not gpkg_path.is_file():
        raise SystemExit(f"GPKG not found: {gpkg_path}")

    h3_dir = DOCS_H3_PREVIEW_DIR.resolve()
    county_dir = DOCS_COUNTY_PREVIEW_DIR.resolve()
    recreation_dir = DOCS_RECREATION_LANDS_DIR.resolve()
    h3_dir.mkdir(parents=True, exist_ok=True)
    county_dir.mkdir(parents=True, exist_ok=True)
    recreation_dir.mkdir(parents=True, exist_ok=True)

    stale_h3_data = h3_dir / "index_data"
    if stale_h3_data.is_dir():
        shutil.rmtree(stale_h3_data)
    for stale in h3_dir.glob("*.json"):
        stale.unlink(missing_ok=True)

    h3_out = build_uarw_study_h3_map(
        gpkg_path,
        h3_dir / "index.html",
        default_layer="__all__" if show_on_load else "All recreation",
        show_layers_on_load=show_on_load,
        heatmap=True,
        data_dir=h3_dir / H3_PAGES_DATA_DIR_NAME,
    )
    county_out = build_uarw_study_county_map(
        gpkg_path,
        county_dir / "index.html",
        default_activity=None,
        show_layers_on_load=show_on_load,
    )
    recreation_data = recreation_dir / RECREATION_LANDS_DATA_DIR_NAME
    recreation_data.mkdir(parents=True, exist_ok=True)
    for stale in recreation_data.glob("*.json"):
        stale.unlink(missing_ok=True)
    recreation_out = build_uarw_study_recreation_lands_map(
        gpkg_path,
        recreation_dir / "index.html",
        default_activity=None,
        show_layers_on_load=show_on_load,
        data_dir=recreation_data,
    )
    return h3_out, county_out, recreation_out


def write_docs_landing_page() -> Path:
    index_path = (_ANALYSIS_ROOT / "docs" / "index.html").resolve()
    index_path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EDWA recreation maps</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
           max-width: 36rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.5; }
    h1 { font-size: 1.35rem; }
    ul { padding-left: 1.25rem; }
    a { color: #0969da; }
  </style>
</head>
<body>
  <h1>Upper American River Watershed recreation maps</h1>
  <ul>
    <li><a href="edwa_h3_preview/index.html">H3 visitation (device-hours)</a></li>
    <li><a href="edwa_county_preview/index.html">Visitor county of origin</a></li>
    <li><a href="edwa_recreation_lands_preview/index.html">Recreation lands &amp; POIs</a></li>
  </ul>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    if args.command == "county":
        gpkg = args.gpkg.resolve()
        if not gpkg.is_file():
            raise SystemExit(f"GPKG not found: {gpkg}")
        out = build_county_origin_map(
            gpkg,
            args.output.resolve(),
            default_activity=args.default_activity,
            boundaries_geojson=args.boundaries_geojson,
        )
        print(f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
        return

    if args.command == "study-h3":
        gpkg = args.gpkg.resolve()
        if not gpkg.is_file():
            raise SystemExit(f"GPKG not found: {gpkg}")
        out = build_uarw_study_h3_map(
            gpkg,
            args.output.resolve(),
            default_layer=args.default_layer,
            show_layers_on_load=args.show_on_load,
            heatmap=not args.no_heatmap,
        )
        data_dir = out.parent / f"{out.stem}_data"
        data_mb = sum(f.stat().st_size for f in data_dir.glob("*.json")) / 1_048_576
        print(
            f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {data_dir.name}/ ({data_mb:.1f} MB GeoJSON)"
        )
        return

    if args.command == "study-county":
        gpkg = args.gpkg.resolve()
        if not gpkg.is_file():
            raise SystemExit(f"GPKG not found: {gpkg}")
        out = build_uarw_study_county_map(
            gpkg,
            args.output.resolve(),
            default_activity=args.default_activity,
        )
        print(f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
        return

    if args.command == "study-recreation":
        gpkg = args.gpkg.resolve()
        if not gpkg.is_file():
            raise SystemExit(f"GPKG not found: {gpkg}")
        out = build_uarw_study_recreation_lands_map(
            gpkg,
            args.output.resolve(),
            default_activity=args.default_activity,
            show_layers_on_load=args.show_on_load,
        )
        data_dir = out.parent / f"{out.stem}_data"
        data_mb = sum(f.stat().st_size for f in data_dir.glob("*.json")) / 1_048_576
        print(
            f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {data_dir.name}/ ({data_mb:.1f} MB GeoJSON)"
        )
        return

    if args.command == "study-pages":
        gpkg = args.gpkg.resolve()
        h3_out, county_out, recreation_out = export_study_pages(
            gpkg, show_on_load=args.show_on_load
        )
        landing = write_docs_landing_page()
        h3_data = h3_out.parent / H3_PAGES_DATA_DIR_NAME
        h3_data_mb = sum(f.stat().st_size for f in h3_data.glob("*.json")) / 1_048_576
        rec_data = recreation_out.parent / RECREATION_LANDS_DATA_DIR_NAME
        rec_data_mb = sum(f.stat().st_size for f in rec_data.glob("*.json")) / 1_048_576
        print(
            f"Wrote {h3_out} ({h3_out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {h3_data.name}/ ({h3_data_mb:.1f} MB GeoJSON)"
        )
        print(f"Wrote {county_out} ({county_out.stat().st_size / 1_048_576:.1f} MB)")
        print(
            f"Wrote {recreation_out} ({recreation_out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {rec_data.name}/ ({rec_data_mb:.1f} MB GeoJSON)"
        )
        print(f"Wrote {landing}")
        print("GitHub Pages: docs/index.html (enable Pages from /docs on main)")
        return

    if args.command == "h3-pages":
        if args.gpkg_paths:
            resolved = [p.resolve() for p in args.gpkg_paths]
        else:
            resolved = default_h3_preview_gpkgs()
        if not resolved:
            raise SystemExit(
                f"No season GPKGs found in {DEFAULT_H3_SEASONS_DIR} "
                f"({DEFAULT_H3_GPKG_PREFIX}_YYYY_{{winter|spring|summer|fall}}.gpkg)"
            )
        for p in resolved:
            if not p.is_file():
                raise SystemExit(f"GPKG not found: {p}")
        preview_dir = DOCS_H3_PREVIEW_DIR.resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        stale_data = preview_dir / "index_data"
        if stale_data.is_dir():
            shutil.rmtree(stale_data)
        print(
            f"Aggregating {len(resolved)} period GeoPackage(s): "
            f"{', '.join(p.name for p in resolved)}"
        )
        out = build_h3_device_hours_map(
            resolved,
            preview_dir / "index.html",
            default_layer=args.default_layer,
            show_layers_on_load=args.show_on_load,
            show_county_boundaries=False,
            heatmap=True,
            data_dir=preview_dir / H3_PAGES_DATA_DIR_NAME,
            aggregate_periods=True,
        )
        data_dir = preview_dir / H3_PAGES_DATA_DIR_NAME
        data_mb = sum(f.stat().st_size for f in data_dir.glob("*.json")) / 1_048_576
        print(
            f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {data_dir.name}/ ({data_mb:.1f} MB GeoJSON)"
        )
        print(f"GitHub Pages path: docs/edwa_h3_preview/index.html")
        return

    if args.command in ("h3", "h3-full-day"):
        if args.command == "h3":
            seasonal_dir = DEFAULT_H3_SEASONS_DIR
            gpkg_prefix = DEFAULT_H3_GPKG_PREFIX
        else:
            seasonal_dir = DEFAULT_H3_FULL_DAY_SEASONAL_DIR
            gpkg_prefix = DEFAULT_H3_FULL_DAY_GPKG_PREFIX
        if args.gpkg_paths:
            resolved = [p.resolve() for p in args.gpkg_paths]
        elif args.command == "h3":
            resolved = default_h3_preview_gpkgs()
        else:
            resolved = discover_seasonal_gpkgs(seasonal_dir, gpkg_prefix=gpkg_prefix)
        if not resolved:
            raise SystemExit(
                f"No season GPKGs found in {seasonal_dir} ({gpkg_prefix}_YYYY_{{winter|spring|summer|fall}}.gpkg)"
            )
        for p in resolved:
            if not p.is_file():
                raise SystemExit(f"GPKG not found: {p}")
        is_stint_map = args.command == "h3"
        if is_stint_map:
            print(
                f"Aggregating {len(resolved)} period GeoPackage(s): "
                f"{', '.join(p.name for p in resolved)}"
            )
        else:
            print(
                f"Including {len(resolved)} season(s): "
                f"{', '.join(season_label_from_path(p) for p in resolved)}"
            )
        heatmap = is_stint_map and not getattr(args, "no_heatmap", False)
        out = build_h3_device_hours_map(
            resolved,
            args.output.resolve(),
            default_layer=args.default_layer,
            show_layers_on_load=args.show_on_load,
            show_county_boundaries=False,
            basemap_tiles=CARTO_LIGHT_NOLABELS_TILES if not is_stint_map else None,
            heatmap=heatmap,
            aggregate_periods=is_stint_map,
        )
        data_dir = out.parent / f"{out.stem}_data"
        data_mb = sum(f.stat().st_size for f in data_dir.glob("*.json")) / 1_048_576
        print(
            f"Wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB) "
            f"+ {data_dir.name}/ ({data_mb:.1f} MB GeoJSON, loaded when layers are enabled)"
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
