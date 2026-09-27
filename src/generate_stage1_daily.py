"""
generate_stage1_daily.py
========================
Stage 1 of the Decoupled Two-Stage Architecture.

Aggregates hourly multimodal CPCB+Weather data to daily frequency,
fetches daily Sentinel-5P Absorbing Aerosol Index (AAI) from Google
Earth Engine with qa_value >= 0.8 pixel masking, and merges them into
Stage 1 Daily Spatial training matrices.

Run with:
    python -m src.generate_stage1_daily --project YOUR_GEE_PROJECT_ID
"""

import argparse
import logging
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import yaml

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("stage1_daily")

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DIR = ROOT_DIR / "data" / "final_multimodal"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "stage1_daily"
CONFIG_PATH = ROOT_DIR / "configs" / "stations.yaml"
REPORT_PATH = ROOT_DIR / "data" / "processed" / "quality_report.csv"


# ═════════════════════════════════════════════════════════════
# STEP 1 — Hourly-to-Daily Aggregation
# ═════════════════════════════════════════════════════════════
def aggregate_hourly_to_daily(filepath: Path) -> pd.DataFrame:
    """
    Load an hourly multimodal CSV, convert UTC → IST,
    drop hourly-only cyclic features, and resample to daily mean.
    """
    df = pd.read_csv(filepath, index_col=0)

    # Convert UTC timestamps to IST for correct daily boundaries
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")

    # Drop hourly-only cyclic features (average to ~0 over a full day)
    hourly_only_cols = ["hour_sin", "hour_cos"]
    df.drop(columns=[c for c in hourly_only_cols if c in df.columns], inplace=True)

    # Resample to daily mean
    daily = df.resample("D").mean()

    # Drop incomplete edge days where PM2.5 target is NaN
    daily.dropna(subset=["pm25"], inplace=True)

    log.info("    Aggregated %d hourly rows → %d daily rows", len(df), len(daily))
    return daily


# ═════════════════════════════════════════════════════════════
# STEP 2 — GEE Sentinel-5P AAI Extraction
# ═════════════════════════════════════════════════════════════
def extract_s5p_aai(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Query COPERNICUS/S5P/OFFL/L3_AER_AI from GEE for the given
    coordinates and date range. Applies qa_value >= 0.8 pixel masking
    and a 5km buffer spatial reduction at scale=3500m.

    Returns a DataFrame with Date index and AER_AI column.
    """
    point = ee.Geometry.Point(lon, lat)
    buffer = point.buffer(5000)  # 5km radius

    collection = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI")
        .filterDate(start_date, end_date)
        .filterBounds(buffer)
    )

    n_images = collection.size().getInfo()
    log.info("    GEE: Found %d S5P images for %s → %s", n_images, start_date, end_date)

    if n_images == 0:
        return pd.DataFrame(columns=["AER_AI"])

    # Extract daily AAI using server-side mapping
    def reduce_image(img):
        result = img.select("absorbing_aerosol_index").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=3500,
            bestEffort=True,
        )
        return ee.Feature(None, {
            'date': img.date().format("YYYY-MM-dd"),
            'AER_AI': result.get("absorbing_aerosol_index")
        })

    # Map the function over the collection (runs entirely on Google's servers)
    reduced_features = collection.map(reduce_image).getInfo()

    # Parse the server response back into a pandas DataFrame
    records = []
    for feature in reduced_features['features']:
        props = feature['properties']
        # GEE returns None for AER_AI if the buffer was completely masked/empty
        if props.get('AER_AI') is not None:
            records.append(props)

    if not records:
        return pd.DataFrame(columns=["AER_AI"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    
    # GEE often has multiple swaths per day; aggregate them to a single daily mean
    df = df.groupby("date").mean()

    # Localize to IST to match the daily-aggregated CPCB data
    df.index = df.index.tz_localize("Asia/Kolkata")

    return df


# ═════════════════════════════════════════════════════════════
# COORDINATE LOOKUP
# ═════════════════════════════════════════════════════════════
def get_station_coords(station_name: str, config: dict):
    """Look up lat/lon from stations.yaml by station name."""
    for st in config["stations"]:
        if st["name"].lower() == station_name.lower():
            return st.get("latitude"), st.get("longitude")
    return None, None


# ═════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 Daily Spatial Datasets"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GEE Cloud Project ID (e.g., ee-yourproject)",
    )
    args = parser.parse_args()

    # ── Initialize GEE ───────────────────────────────────────
    log.info("Initializing Google Earth Engine (project: %s) ...", args.project)
    ee.Initialize(project=args.project)
    log.info("GEE initialized successfully.")

    # ── Load config ──────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # ── Load quality report to identify KEPT chunks ──────────
    report_df = pd.read_csv(REPORT_PATH)
    kept_chunks = report_df[report_df["status"] == "KEPT"]
    log.info("Found %d KEPT chunks to process.", len(kept_chunks))

    # ── Create output directory ──────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Verification report accumulator ──────────────────────
    verification = []

    for _, row in kept_chunks.iterrows():
        station_name = row["City/Station"]
        chunk_id = row["Chunk_ID"]
        start_date = row["Chunk_Start_Date"]
        end_date = row["Chunk_End_Date"]

        log.info("━━━ Processing: %s ━━━", chunk_id)

        # ── Step 1: Hourly → Daily ───────────────────────────
        multimodal_file = MULTIMODAL_DIR / f"{chunk_id}_multimodal.csv"
        if not multimodal_file.exists():
            log.warning("  ✗ Multimodal file not found: %s", multimodal_file.name)
            continue

        daily_df = aggregate_hourly_to_daily(multimodal_file)
        agg_rows = len(daily_df)

        # ── Step 2: GEE Sentinel-5P ─────────────────────────
        lat, lon = get_station_coords(station_name, config)
        if lat is None or lon is None:
            log.error("  ✗ Coordinates not found for: %s", station_name)
            continue

        try:
            gee_df = extract_s5p_aai(lat, lon, start_date, end_date)
        except Exception as e:
            log.error("  ✗ GEE extraction failed for %s: %s", chunk_id, e)
            continue

        gee_rows = len(gee_df)

        # ── Step 3: Inner Join ───────────────────────────────
        # Normalize both indexes to date-only for clean join
        daily_df.index = daily_df.index.normalize()
        gee_df.index = gee_df.index.normalize()

        merged = daily_df.join(gee_df, how="inner")

        # Drop any rows where AER_AI is NaN (cloud-blocked days)
        merged.dropna(subset=["AER_AI"], inplace=True)
        final_rows = len(merged)

        # ── Step 4: Save ─────────────────────────────────────
        out_path = OUTPUT_DIR / f"{chunk_id}_stage1.csv"
        merged.to_csv(out_path)
        log.info(
            "  ✓ Saved %s — Agg: %d days | GEE: %d days | Merged: %d days",
            out_path.name,
            agg_rows,
            gee_rows,
            final_rows,
        )

        verification.append(
            {
                "Chunk_ID": chunk_id,
                "Aggregated_Daily_Rows": agg_rows,
                "GEE_Rows": gee_rows,
                "Final_Merged_Rows": final_rows,
            }
        )

        # Rate limit between chunks
        time.sleep(1.5)

    # ── Verification Report ──────────────────────────────────
    log.info("═" * 60)
    log.info("STAGE 1 DAILY VERIFICATION REPORT")
    log.info("═" * 60)
    if verification:
        vdf = pd.DataFrame(verification)
        log.info("\n%s", vdf.to_string(index=False))
    else:
        log.warning("No chunks were processed successfully.")


if __name__ == "__main__":
    main()
