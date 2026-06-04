"""Tests for H3 web map export helpers."""

from export_web_maps import h3_layer_visible

from h3_device_hours_core import period_to_display_label, recreation_season_period
from web_map_export import (
    ALL_RECREATION_ACTIVITY_KEY,
    aggregate_h3_device_hours_across_gpkgs,
    h3_colors_for_activity,
    is_comprehensive_h3_activity,
    partition_h3_export_layers,
    value_range,
    value_range_for_heatmap,
)
from web_map_export import DEFAULT_H3_GPKG_PREFIX, DEFAULT_H3_SEASONS_DIR, discover_seasonal_gpkgs


def test_partition_h3_export_layers() -> None:
    comprehensive, detail = partition_h3_export_layers(
        ["trails", ALL_RECREATION_ACTIVITY_KEY, "camping", "all_recreation_z8"]
    )
    assert comprehensive == [ALL_RECREATION_ACTIVITY_KEY]
    assert "trails" in detail
    assert "camping" in detail
    assert "all_recreation_z8" not in detail


def test_all_recreation_is_comprehensive() -> None:
    assert is_comprehensive_h3_activity(ALL_RECREATION_ACTIVITY_KEY)
    assert not is_comprehensive_h3_activity("trails")


def test_recreation_season_period_winter_year() -> None:
    assert recreation_season_period("2024-12-15") == "2024_winter"
    assert recreation_season_period("2025-01-20") == "2024_winter"
    assert recreation_season_period("2025-03-31") == "2024_winter"
    assert recreation_season_period("2024-07-04") == "2024_summer"
    assert period_to_display_label("2024_winter") == "Winter 2024"
    assert period_to_display_label("2024_fall") == "Fall 2024"


def test_heatmap_colors_are_distinct_from_legacy() -> None:
    heat = h3_colors_for_activity("trails", heatmap=True)
    legacy = h3_colors_for_activity("trails", heatmap=False)
    assert heat != legacy
    assert len(heat) >= 4
    assert heat == h3_colors_for_activity("camping", heatmap=True)
    assert heat[0].startswith("#")
    assert heat[-1].startswith("#")


def test_heatmap_blue_red_ramp_orders_low_to_high() -> None:
    from web_map_export import H3_HEAT_RAMP_BLUE_RED

    assert H3_HEAT_RAMP_BLUE_RED[0] != H3_HEAT_RAMP_BLUE_RED[-1]


def test_aggregate_h3_sums_device_hours_by_cell() -> None:
    paths = discover_seasonal_gpkgs(DEFAULT_H3_SEASONS_DIR, gpkg_prefix=DEFAULT_H3_GPKG_PREFIX)
    if len(paths) < 2:
        return
    combined = aggregate_h3_device_hours_across_gpkgs(paths[:2])
    assert "trails" in combined
    assert len(combined["trails"]) > 0
    assert combined["trails"]["device_hours"].sum() >= combined["trails"]["device_hours"].max()


def test_heatmap_value_range_uses_percentile_not_spike() -> None:
    values = [1.0] * 100 + [500.0]
    vmin_lin, vmax_lin = value_range(values)
    vmin_hm, vmax_hm = value_range_for_heatmap(values, vmax_percentile=0.92)
    assert vmin_lin == vmin_hm == 1.0
    assert vmax_lin == 500.0
    assert vmax_hm < vmax_lin


def test_h3_layer_visible_supports_all_layers_sentinel() -> None:
    assert h3_layer_visible(
        "trails",
        "__all__",
        show_on_load=True,
    )
    assert h3_layer_visible(
        ALL_RECREATION_ACTIVITY_KEY,
        "__all__",
        show_on_load=True,
    )
