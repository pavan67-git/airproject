"""
fetch_cpcb.py
=============
Downloads historical CPCB air quality data from the OpenAQ public
S3 archive (s3://openaq-data-archive).

Data flow:
  1. Read station list + date range from configs/stations.yaml
  2. For each station × year × month, download CSV.GZ files from S3
  3. Pivot from OpenAQ narrow format to wide-format hourly columns
  4. Save one CSV per station into data/raw/

Usage:
    python -m src.fetch_cpcb
"""

import io
import gzip
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import concurrent.futures

import boto3
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.config import Config

# ── Constants ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STATION_CFG = ROOT / "configs" / "stations.yaml"
RAW_DIR = ROOT / "data" / "raw"
BUCKET = "openaq-data-archive"
PREFIX_TEMPLATE = "records/csv.gz/locationid={loc_id}/year={year}/month={month}/"

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("fetch_cpcb")


def _load_config() -> dict:
    """Load station registry from YAML."""
    with open(STATION_CFG, "r") as f:
        return yaml.safe_load(f)


def _build_month_list(start: str, end: str) -> List[Tuple[int, int]]:
    """Return a list of (year, month) tuples spanning the date range."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months = []
    current = s.replace(day=1)
    while current <= e:
        months.append((current.year, current.month))
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def _download_station(
    s3_client,
    location_id: int,
    months: List[Tuple[int, int]],
    target_params: List[str],
) -> pd.DataFrame:
    """
    Download all CSV.GZ files for a station across the given months,
    parse in-memory, and return a single wide-format DataFrame.

    OpenAQ narrow format columns (typical):
        location_id, sensors_id, location, datetime, lat, lon, parameter, units, value
    """
    # 1. Collect all object keys across all months first
    keys_to_download = []
    for year, month in months:
        prefix = PREFIX_TEMPLATE.format(
            loc_id=location_id, year=year, month=str(month).zfill(2)
        )
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            page_iter = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
            for page in page_iter:
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".csv.gz"):
                        keys_to_download.append(obj["Key"])
        except Exception as exc:
            log.warning("S3 list failed for %s: %s", prefix, exc)
            
    if not keys_to_download:
        return pd.DataFrame()

    # 2. Define a worker function to download and parse one file
    def _fetch_one_key(key: str) -> pd.DataFrame:
        try:
            # Create a thread-local client for thread safety
            thread_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            resp = thread_s3.get_object(Bucket=BUCKET, Key=key)
            raw_bytes = resp["Body"].read()
            csv_text = gzip.decompress(raw_bytes).decode("utf-8")
            return pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:
            log.warning("Failed to read %s: %s", key, exc)
            return pd.DataFrame()

    # 3. Download concurrently
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_key = {executor.submit(_fetch_one_key, key): key for key in keys_to_download}
        for future in concurrent.futures.as_completed(future_to_key):
            df_part = future.result()
            if not df_part.empty:
                frames.append(df_part)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # --- Standardize column names (OpenAQ may vary slightly) ---
    col_map = {}
    for col in combined.columns:
        lc = col.lower().strip()
        if "datetime" in lc or lc == "date":
            col_map[col] = "datetime"
        elif lc == "parameter":
            col_map[col] = "parameter"
        elif lc == "value":
            col_map[col] = "value"
    combined.rename(columns=col_map, inplace=True)

    # Ensure we have the essential columns
    for req in ("datetime", "parameter", "value"):
        if req not in combined.columns:
            log.error(
                "Station %s: missing column '%s'. Columns found: %s",
                location_id,
                req,
                list(combined.columns),
            )
            return pd.DataFrame()

    # Filter to target pollutant parameters only
    combined["parameter"] = combined["parameter"].str.lower().str.strip()
    combined = combined[combined["parameter"].isin(target_params)]

    if combined.empty:
        return pd.DataFrame()

    # --- Parse datetime → IST (Layer 1 requirement) ---
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True)
    combined["datetime"] = combined["datetime"].dt.tz_convert(
        "Asia/Kolkata"
    )
    # Floor to the hour to align all readings on clean hourly boundaries
    combined["datetime"] = combined["datetime"].dt.floor("h")

    # Coerce values to numeric (handles stray strings)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")

    # --- Pivot narrow → wide ---
    # Average duplicates within the same hour+parameter bucket
    wide = combined.pivot_table(
        index="datetime",
        columns="parameter",
        values="value",
        aggfunc="mean",
    )
    wide.index.name = "datetime"
    wide.columns.name = None  # flatten MultiIndex name

    return wide


def main() -> None:
    """Orchestrate the full download for all configured stations."""
    cfg = _load_config()
    months = _build_month_list(
        cfg["date_range"]["start"], cfg["date_range"]["end"]
    )
    target_params = [p.lower().strip() for p in cfg["parameters"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    log.info(
        "Fetching %d stations × %d months from s3://%s",
        len(cfg["stations"]),
        len(months),
        BUCKET,
    )

    for station in cfg["stations"]:
        loc_id = station["openaq_location_id"]
        name = station["name"]
        log.info("▶ Station: %s  (locationId=%d)", name, loc_id)

        df = _download_station(s3, loc_id, months, target_params)

        if df.empty:
            log.warning("  ⚠ No data returned — skipping.")
            continue

        # Save raw wide-format CSV
        safe_name = (
            name.lower()
            .replace(",", "")
            .replace(" - ", "_")
            .replace(" ", "_")
        )
        out_path = RAW_DIR / f"{safe_name}_raw.csv"
        df.to_csv(out_path)
        log.info(
            "  ✓ Saved %s  (%d rows × %d cols)",
            out_path.name,
            len(df),
            df.shape[1],
        )

    log.info("Done. Raw files are in %s", RAW_DIR)


if __name__ == "__main__":
    main()
