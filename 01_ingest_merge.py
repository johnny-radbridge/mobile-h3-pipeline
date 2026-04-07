"""
STEP 1: INGEST, MERGE & H3 ASSIGNMENT (FINAL PIPELINE)

Inputs:
    - Raw ping/pathing folders specified in parameters.json
Outputs:
    - combined_pings.parquet (ping-level data with H3 13–6, device_localdate_id)
    - device_localdate_id_count_by_visit_date.csv (per visit_date: record_count,
      unique_device_localdate_id_count; Brattleboro CEL/CDL columns added/overwritten in Step 2)

Key tasks:
    - Canonical schema: device_id, unix_ts, lat, lon, source_file, source_type, pin_num
    - Derive visit_* calendar fields from unix_ts (LOCAL_TIMEZONE); remove vendor time fields
    - Deduplicate on (device_id, unix_ts), preferring ping over pathing when duplicates exist
    - Assign globally unique pin_num after sort
    - Assign H3_13 from lat/lon, derive H3_12..H3_6 via parent relationships
    - Include print statements for QA/monitoring
"""

from pathlib import Path
import json
import tempfile

import polars as pl

from pipeline_logger import log_step, DEFAULT_LOG_PATH
from tqdm import tqdm
import h3

# =============================================================================
# CONFIGURATION
# =============================================================================

# Root project directory (where parameters.json lives)
ROOT_DIR = Path(__file__).resolve().parents[1]

# This analysis run's base directory
ANALYSIS_DIR = ROOT_DIR

# Load parameters from root parameters.json
PARAMS_FILE = ROOT_DIR / "parameters.json"
with open(PARAMS_FILE, "r") as f:
    PARAMS = json.load(f)

# Folder paths (relative paths in params are anchored at ROOT_DIR)
PING_FOLDERS = [ROOT_DIR / f for f in PARAMS["ping_data"]]
PATHING_FOLDERS = [ROOT_DIR / f for f in PARAMS["pathing_files"]]

LOCAL_TIMEZONE = PARAMS["local_timezone"]  # e.g. "America/Chicago"

# H3 configuration for this step: base at res 13, parents down to 6
H3_BASE_RES = 13
H3_PARENT_RESOLUTIONS = [12, 11, 10, 9, 8, 7, 6]

# Output locations for this final analysis run
OUTPUT_DIR = ANALYSIS_DIR / "intermediate" / "01_ingest_merge"
OUTPUT_FILE = OUTPUT_DIR / "combined_pings.parquet"
OUTPUT_SAMPLE = OUTPUT_DIR / "combined_pings_sample.csv"
OUTPUT_DEVICE_LOCALDATE_BY_DATE = OUTPUT_DIR / "device_localdate_id_count_by_visit_date.csv"

# Canonical schema: ONLY these columns in final output (order preserved)
CANONICAL_COLUMNS = [
    "device_id",
    "unix_ts",
    "lat",
    "lon",
    "source_file",
    "source_type",
    "pin_num",
    "visit_year",
    "visit_month",
    "visit_day",
    "visit_date",
    "visit_dow_num",
    "visit_dow",
    "visit_hour_of_day",
    "visit_time_of_day",
    "visit_time_zone",
    "device_localdate_id",
    "geom",
    "h3_13",
    "h3_12",
    "h3_11",
    "h3_10",
    "h3_9",
    "h3_8",
    "h3_7",
    "h3_6",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tsv_files_from_folders(paths: list[Path]) -> list[Path]:
    """
    Collect all TSV files (including .gz) from the specified paths.

    Each entry in `paths` may be either:
      - a directory (we collect all *.tsv / *.tsv.gz inside), or
      - a specific file path (we include it directly if it exists).
    """
    all_files: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"⚠️  Warning: Path not found: {p}")
            continue

        if p.is_dir():
            # Get all TSV files (both .tsv and .tsv.gz) within the directory
            tsv_files = list(p.glob("*.tsv")) + list(p.glob("*.tsv.gz"))
            all_files.extend(tsv_files)
        else:
            # Treat as explicit file path
            if p.suffix in {".tsv", ".gz"}:
                all_files.append(p)

    return sorted(all_files)


def load_ping_data(files: list[Path]) -> pl.DataFrame:
    """
    Load and normalize ping data from TSV files.

    - Normalizes column names to snake_case
    - Maps to canonical schema: device_id, unix_ts, lat, lon, source_file, source_type
    - Drops vendor date/time columns
    """
    dfs: list[pl.DataFrame] = []
    for f in tqdm(files, desc="Loading ping files"):
        if not f.exists():
            print(f"⚠️  Warning: File not found: {f}")
            continue

        df = pl.read_csv(f, separator="\t", infer_schema_length=1000)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No ping files found!")

    df = pl.concat(dfs, how="diagonal")

    # Drop vendor date/time columns
    drop_cols = {"Date", "Time of Day", "Day of Week", "Time Zone"}
    df = df.drop([c for c in df.columns if c in drop_cols])

    # Normalize column names to snake_case
    column_mapping = {
        col: col.lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    df = df.rename(column_mapping)

    # Handle alternative device ID column
    if "hashed_ubermedia_id" in df.columns and "hashed_device_id" not in df.columns:
        df = df.rename({"hashed_ubermedia_id": "hashed_device_id"})

    # Require raw columns; map to canonical names
    if "hashed_device_id" not in df.columns:
        raise ValueError("Missing required ping column: hashed_device_id")
    if "unix_timestamp_of_visit" not in df.columns:
        raise ValueError("Missing required ping column: unix_timestamp_of_visit")
    if "lat_of_visit" not in df.columns:
        raise ValueError("Missing required ping column: lat_of_visit")
    if "lon_of_visit" not in df.columns:
        raise ValueError("Missing required ping column: lon_of_visit")
    if "polygon_id" not in df.columns:
        raise ValueError("Missing required ping column: polygon_id (Polygon ID)")

    # Canonical schema: device_id, unix_ts, lat, lon, source_file, source_type
    df = df.select(
        [
            pl.col("hashed_device_id").alias("device_id"),
            pl.col("unix_timestamp_of_visit").cast(pl.Int64, strict=False).alias("unix_ts"),
            pl.col("lat_of_visit").cast(pl.Float64).alias("lat"),
            pl.col("lon_of_visit").cast(pl.Float64).alias("lon"),
            pl.col("polygon_id").cast(pl.Utf8).alias("source_file"),
            pl.lit("ping").alias("source_type"),
        ]
    )

    return df


def load_pathing_data(files: list[Path]) -> pl.DataFrame | None:
    """
    Load and normalize pathing data from TSV files.

    - Normalizes column names to snake_case
    - Maps observation columns to canonical fields (same as ping)
    - Outputs canonical schema: device_id, unix_ts, lat, lon, source_file, source_type
    """
    dfs: list[pl.DataFrame] = []
    for f in tqdm(files, desc="Loading pathing files"):
        if not f.exists():
            print(f"⚠️  Warning: File not found: {f}")
            continue

        df = pl.read_csv(f, separator="\t", infer_schema_length=1000)
        dfs.append(df)

    if not dfs:
        print("⚠️  No pathing files found - proceeding with ping data only.")
        return None

    df = pl.concat(dfs, how="diagonal")

    # Normalize columns
    column_mapping = {
        col: col.lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    df = df.rename(column_mapping)

    # Map observation-oriented columns to canonical equivalents
    rename_map: dict[str, str] = {}
    if "hashed_ubermedia_id" in df.columns and "hashed_device_id" not in df.columns:
        rename_map["hashed_ubermedia_id"] = "hashed_device_id"
    if "lat_of_observation_point" in df.columns and "lat_of_visit" not in df.columns:
        rename_map["lat_of_observation_point"] = "lat_of_visit"
    if "lon_of_observation_point" in df.columns and "lon_of_visit" not in df.columns:
        rename_map["lon_of_observation_point"] = "lon_of_visit"
    if "unix_timestamp_of_observation_point" in df.columns and "unix_timestamp_of_visit" not in df.columns:
        rename_map["unix_timestamp_of_observation_point"] = "unix_timestamp_of_visit"

    if rename_map:
        df = df.rename(rename_map)

    # Require raw columns; map to canonical schema
    if "hashed_device_id" not in df.columns:
        raise ValueError("Missing required pathing column: hashed_device_id")
    if "unix_timestamp_of_visit" not in df.columns:
        raise ValueError("Missing required pathing column: unix_timestamp_of_visit")
    if "lat_of_visit" not in df.columns:
        raise ValueError("Missing required pathing column: lat_of_visit")
    if "lon_of_visit" not in df.columns:
        raise ValueError("Missing required pathing column: lon_of_visit")
    if "polygon_id" not in df.columns:
        raise ValueError("Missing required pathing column: polygon_id (Polygon ID)")

    # Canonical schema: device_id, unix_ts, lat, lon, source_file, source_type
    df = df.select(
        [
            pl.col("hashed_device_id").alias("device_id"),
            pl.col("unix_timestamp_of_visit").cast(pl.Int64, strict=False).alias("unix_ts"),
            pl.col("lat_of_visit").cast(pl.Float64).alias("lat"),
            pl.col("lon_of_visit").cast(pl.Float64).alias("lon"),
            pl.col("polygon_id").cast(pl.Utf8).alias("source_file"),
            pl.lit("pathing").alias("source_type"),
        ]
    )

    return df


def add_partition_date(df: pl.DataFrame, epoch_unit: str = "s") -> pl.DataFrame:
    """
    Add _partition_date from unix_ts using LOCAL timezone.
    Use this for chunked processing; aligns with visit_date/device_localdate_id.
    (Unix epoch is UTC; we convert to local time then extract date.)
    """
    ts = pl.col("unix_ts").cast(pl.Int64, strict=False)
    df = df.with_columns(
        pl.from_epoch(ts, time_unit=epoch_unit)
        .dt.convert_time_zone(LOCAL_TIMEZONE)
        .dt.date()
        .alias("_partition_date")
    )
    return df


def process_timestamps(
    df: pl.DataFrame, epoch_unit: str | None = None
) -> pl.DataFrame:
    """
    Derive canonical visit_* calendar fields from unix_ts using LOCAL_TIMEZONE.
    Do NOT create datetime_utc or datetime_local; derive only the required fields.
    """
    ts_col = "unix_ts"

    # Ensure timestamp is numeric
    df = df.with_columns(pl.col(ts_col).cast(pl.Int64, strict=False))

    # Detect epoch unit if not provided (seconds vs milliseconds)
    if epoch_unit is None:
        max_val = df[ts_col].max()
        epoch_unit = "ms" if max_val is not None and max_val > 1e11 else "s"

    # Local datetime from epoch (naive epoch treated as UTC; convert to LOCAL_TIMEZONE)
    dt_local = pl.from_epoch(pl.col(ts_col), time_unit=epoch_unit).dt.convert_time_zone(LOCAL_TIMEZONE)

    # Canonical visit_* fields only
    df = df.with_columns(
        [
            dt_local.dt.date().alias("visit_date"),
            dt_local.dt.year().alias("visit_year"),
            dt_local.dt.month().alias("visit_month"),
            dt_local.dt.day().alias("visit_day"),
            dt_local.dt.weekday().alias("visit_dow_num"),  # ISO: Mon=1 .. Sun=7
            dt_local.dt.strftime("%A").alias("visit_dow"),
            dt_local.dt.hour().alias("visit_hour_of_day"),
            dt_local.dt.strftime("%H:%M:%S").alias("visit_time_of_day"),
            pl.lit(LOCAL_TIMEZONE).alias("visit_time_zone"),
        ]
    )

    return df


def create_device_localdate_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct device_localdate_id = device_id + "_" + YYYYMMDD (local date).
    """
    df = df.with_columns(
        (
            pl.col("device_id").fill_null("")  # guard against null IDs
            + "_"
            + pl.col("visit_date").cast(pl.Utf8).str.replace_all("-", "")
        ).alias("device_localdate_id")
    )
    return df


def latlon_to_h3(lat: float | None, lon: float | None, resolution: int) -> str | None:
    """Safe lat/lon → H3 cell conversion."""
    if lat is None or lon is None:
        return None
    try:
        return h3.latlng_to_cell(lat, lon, resolution)
    except Exception:
        return None


def make_h3_converter(resolution: int):
    """Create an H3 converter closure for Polars struct.map_elements."""

    def converter(row: dict) -> str | None:
        return latlon_to_h3(row.get("lat"), row.get("lon"), resolution)

    return converter


def make_h3_parent_converter(parent_resolution: int):
    """
    Create a converter that derives a parent H3 cell from an h3_13 child.
    """

    def converter(row: dict) -> str | None:
        cell = row.get("h3_13")
        if cell is None:
            return None
        try:
            return h3.cell_to_parent(cell, parent_resolution)
        except Exception:
            return None

    return converter


def assign_h3_indices(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Assign H3 indices:
        - h3_13 from (lat, lon)
        - h3_12 .. h3_6 via parent relationships from h3_13
    When verbose=False, suppresses per-chunk logging (for chunked processing).
    """
    if verbose:
        print("\n🔷 Assigning H3 indices (13 → 6)...")
        print(f"   Computing h3_{H3_BASE_RES} from lat/lon...")

    h3_converter = make_h3_converter(H3_BASE_RES)
    df = df.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(h3_converter, return_dtype=pl.Utf8)
        .alias(f"h3_{H3_BASE_RES}")
    )

    total_rows = len(df)
    if verbose:
        non_null_base = df[f"h3_{H3_BASE_RES}"].is_not_null().sum()
        pct_base = (non_null_base / total_rows * 100) if total_rows > 0 else 0
        print(
            f"      ✅ h3_{H3_BASE_RES}: {non_null_base:,} non-null "
            f"({pct_base:.1f}% of {total_rows:,})"
        )

    # Parent resolutions derived from h3_13
    for res in H3_PARENT_RESOLUTIONS:
        col_name = f"h3_{res}"
        if verbose:
            print(f"   Deriving {col_name} from h3_13...")
        parent_converter = make_h3_parent_converter(res)
        df = df.with_columns(
            pl.struct([f"h3_{H3_BASE_RES}"])
            .map_elements(parent_converter, return_dtype=pl.Utf8)
            .alias(col_name)
        )
        if verbose:
            non_null = df[col_name].is_not_null().sum()
            pct = (non_null / total_rows * 100) if total_rows > 0 else 0
            print(f"      ✅ {col_name}: {non_null:,} non-null ({pct:.1f}%)")

    return df


def create_geometry_wkt(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create a simple WKT POINT geometry column ("geom") from lon/lat.
    """
    df = df.with_columns(
        pl.when(
            pl.col("lon").is_not_null()
            & pl.col("lat").is_not_null()
            & pl.col("lon").is_finite()
            & pl.col("lat").is_finite()
        )
        .then(
            pl.concat_str(
                [
                    pl.lit("POINT("),
                    pl.col("lon").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("lat").cast(pl.Utf8),
                    pl.lit(")"),
                ]
            )
        )
        .otherwise(None)
        .alias("geom")
    )
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> pl.DataFrame:
    print("=" * 60)
    print("STEP 1 (FINAL): INGEST, MERGE & H3")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load ping data (primary source)
    # -------------------------------------------------------------------------
    print("\n📂 Loading ping data (primary source)...")
    ping_files = get_tsv_files_from_folders(PING_FOLDERS)
    print(f"   Found {len(ping_files)} ping files in folders")
    df_pings = load_ping_data(ping_files)
    print(f"   ✅ Loaded {len(df_pings):,} ping records")

    # -------------------------------------------------------------------------
    # 2. Load pathing data (secondary source)
    # -------------------------------------------------------------------------
    print("\n📂 Loading pathing data (secondary source)...")
    pathing_files = get_tsv_files_from_folders(PATHING_FOLDERS)
    print(f"   Found {len(pathing_files)} pathing files in folders")
    df_pathing = load_pathing_data(pathing_files)

    # -------------------------------------------------------------------------
    # 3. Combine sources (identical canonical schema from both loaders)
    # -------------------------------------------------------------------------
    print("\n🔧 Combining sources...")

    if df_pathing is not None:
        df_combined = pl.concat([df_pings, df_pathing], how="diagonal")
    else:
        df_combined = df_pings

    # Explicit ping preference: 0=ping, 1=pathing; sort before dedup so ping wins
    df_combined = df_combined.with_columns(
        pl.when(pl.col("source_type") == "ping").then(0).otherwise(1).alias("_source_priority")
    )

    print(f"   ✅ Combined dataset: {len(df_combined):,} total records (ping + pathing)")

    # -------------------------------------------------------------------------
    # 4–7. Process by LOCAL date chunk: dedup (ping preferred), timestamps, geometry, sort, pin_num, H3
    # -------------------------------------------------------------------------
    max_ts = df_combined["unix_ts"].max()
    epoch_unit = "ms" if max_ts is not None and max_ts > 1e11 else "s"
    df_combined = add_partition_date(df_combined, epoch_unit)
    partition_dates = df_combined["_partition_date"].unique().sort().to_list()
    before_dedup = len(df_combined)

    print(
        f"\n🔄 Processing by local date ({LOCAL_TIMEZONE}): {len(partition_dates):,} days"
    )
    print("   Per day: sort (device_id, unix_ts, _source_priority) → dedup (ping preferred) → timestamps → geometry → sort → pin_num → H3")

    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        tmp_path = Path(tmpdir)
        pin_offset = 0
        for d in tqdm(partition_dates, desc="   Chunk"):
            part = df_combined.filter(pl.col("_partition_date") == d)
            part = part.sort(["device_id", "unix_ts", "_source_priority"])
            part = part.unique(subset=["device_id", "unix_ts"], keep="first")
            part = part.drop("_source_priority")
            part = process_timestamps(part, epoch_unit)
            part = create_device_localdate_id(part)
            part = create_geometry_wkt(part)
            part = part.sort(["device_id", "unix_ts"])
            part = part.with_columns(
                pl.arange(pin_offset + 1, pin_offset + 1 + len(part)).alias("pin_num")
            )
            pin_offset += len(part)
            part = assign_h3_indices(part, verbose=False)
            part = part.drop("_partition_date")
            part.write_parquet(tmp_path / f"part_{d}.parquet")

        del df_combined
        df_combined = pl.concat(
            pl.read_parquet(f) for f in sorted(tmp_path.glob("part_*.parquet"))
        )

    duplicates_removed = before_dedup - len(df_combined)
    print(f"   ✅ Removed {duplicates_removed:,} duplicates; processed all {len(partition_dates):,} days")

    # -------------------------------------------------------------------------
    # 8. Enforce canonical schema and save outputs
    # -------------------------------------------------------------------------
    df_combined = df_combined.select(
        [c for c in CANONICAL_COLUMNS if c in df_combined.columns]
    )

    print("\n💾 Saving outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_combined.write_parquet(OUTPUT_FILE)
    print(f"   ✅ Saved parquet: {OUTPUT_FILE}")

    df_combined.head(1000).write_csv(OUTPUT_SAMPLE)
    print(f"   ✅ Saved sample CSV (first 1,000 rows): {OUTPUT_SAMPLE}")

    df_device_localdate_by_date = (
        df_combined.group_by("visit_date")
        .agg(
            pl.len().alias("record_count"),
            pl.col("device_localdate_id").n_unique().alias("unique_device_localdate_id_count"),
        )
        .sort("visit_date")
        .select("visit_date", "record_count", "unique_device_localdate_id_count")
    )
    df_device_localdate_by_date.write_csv(OUTPUT_DEVICE_LOCALDATE_BY_DATE)
    print(
        f"   ✅ Saved daily ping counts: {OUTPUT_DEVICE_LOCALDATE_BY_DATE} "
        "(CEL/CDL Brattleboro columns added when Step 2 runs)"
    )

    # -------------------------------------------------------------------------
    # 9. QA Summary (required metrics)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n = len(df_combined)
    print(f"Total records: {n:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Unique devices: {df_combined['device_id'].n_unique():,}")
    print(f"Unique device_localdate_id: {df_combined['device_localdate_id'].n_unique():,}")
    print(
        f"Date range: {df_combined['visit_date'].min()} "
        f"to {df_combined['visit_date'].max()}"
    )
    h3_13_non_null = df_combined["h3_13"].is_not_null().sum()
    pct_h3_13 = (h3_13_non_null / n * 100) if n > 0 else 0
    print(f"% non-null h3_13: {pct_h3_13:.2f}%")

    # Pipeline log
    n_ids = df_combined["device_localdate_id"].n_unique()
    log_step(
        step_id=1,
        pings_start=before_dedup,
        pings_end=len(df_combined),
        device_localdate_ids_start=0,
        device_localdate_ids_end=n_ids,
        filter_details=[
            {"name": "Deduplication (device_id, unix_ts)", "pings_removed": duplicates_removed},
        ],
    )
    print(f"\n📋 Pipeline log updated: {DEFAULT_LOG_PATH}")

    print("\n✅ Step 1 (final) complete.")
    return df_combined


if __name__ == "__main__":
    main()

