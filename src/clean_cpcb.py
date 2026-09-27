"""
clean_cpcb.py
=============
Implements the 6-layer cleaning pipeline for CPCB ground data,
as defined in the implementation plan.

Layers:
  1. IST Timezone Standardization & Continuous DateTime Index
  2. Aggressive Artifact Pre-Filtering (≤0, upper-bound spikes)
  3. Quiet Sensor Drift Detection & Cross-Pollutant Inversion Check
  4. Quality Assessment & Station Filtering
  5. Safe Micro-Imputation (≤3-hour gaps only)
  6. Smart Sequence Windowing

Usage:
    python -m src.clean_cpcb
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import yaml

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
STATION_CFG = ROOT / "configs" / "stations.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("clean_cpcb")

# ── Pollutant column name mapping ────────────────────────────
# OpenAQ uses lowercase; we keep them consistent internally.
POLLUTANT_COLS = ["pm25", "pm10", "no2"]


def _load_pipeline_cfg() -> dict:
    with open(PIPELINE_CFG) as f:
        return yaml.safe_load(f)


def _load_station_cfg() -> dict:
    with open(STATION_CFG) as f:
        return yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════
# LAYER 1 — IST Timezone Standardization & Reindexing
# ═════════════════════════════════════════════════════════════
def layer1_standardize_index(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Parse the datetime index as IST (UTC+05:30), floor to the hour,
    and reindex onto a gapless hourly grid spanning [start, end].

    This guarantees a continuous time chain so that later layers can
    safely reason about "contiguous gaps" using simple index arithmetic.
    """
    log.info("  L1 │ Standardising datetime index to IST …")

    # If the index is not already a DatetimeIndex, parse it
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Convert to IST if timezone-aware, else localize as IST
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_localize("Asia/Kolkata")

    # Floor to the hour (handles 15-min resolution data)
    df.index = df.index.floor("h")

    # Drop duplicate timestamps (keep mean of duplicates)
    df = df.groupby(df.index).mean()

    # Build the continuous hourly grid based on the stations true lifespan
    start_ts = df.index.min()
    end_ts = df.index.max()
    full_range = pd.date_range(start=start_ts, end=end_ts, freq="h")
    df = df.reindex(full_range)
    df.index.name = "datetime"

    log.info("  L1 │ Continuous grid: %s → %s  (%d hours)", df.index[0], df.index[-1], len(df))
    return df


# ═════════════════════════════════════════════════════════════
# LAYER 2 — Aggressive Artifact Pre-Filtering
# ═════════════════════════════════════════════════════════════
def layer2_artifact_filter(df: pd.DataFrame, cfg: dict, report: dict) -> pd.DataFrame:
    """
    Purge negative values, extreme spikes, and apply Adaptive Baseline Filtering.
    """
    log.info("  L2 │ Artifact pre-filtering & Adaptive Baseline …")
    upper = cfg["artifact_filter"]["upper_bounds"]

    for col in POLLUTANT_COLS:
        if col not in df.columns:
            continue
        before_nan = df[col].isna().sum()

        # 1. Purge negatives, zeros, and known error codes FIRST
        df.loc[df[col] <= 0, col] = np.nan

        # 2. Purge upper-bound voltage spikes
        ceiling = upper.get(col, float("inf"))
        df.loc[df[col] > ceiling, col] = np.nan

        # 3. Adaptive Baseline Filter (specifically for pm25)
        if col == "pm25":
            station_mean = df[col].mean()
            if pd.isna(station_mean):
                report["Station_Mean_PM25"] = None
                report["Calculated_Dynamic_Threshold"] = None
            else:
                dynamic_threshold = max(station_mean * 0.05, 3.0)
                report["Station_Mean_PM25"] = round(station_mean, 2)
                report["Calculated_Dynamic_Threshold"] = round(dynamic_threshold, 2)
                
                # 12-hour rolling std with min_periods=2
                rolling_std = df[col].rolling(window=12, min_periods=2).std()
                
                # Execution Rule: PM2.5 < Dynamic Threshold AND rolling std < 1.0
                noise_mask = (df[col] < dynamic_threshold) & (rolling_std < 1.0)
                df.loc[noise_mask, col] = np.nan

        after_nan = df[col].isna().sum()
        purged = after_nan - before_nan
        if purged > 0:
            log.info("  L2 │   %s: purged %d artifact/noise readings", col, purged)

    return df


# ═════════════════════════════════════════════════════════════
# LAYER 3 — Sensor Drift Detection & Cross-Pollutant Checks
# ═════════════════════════════════════════════════════════════
def layer3_drift_and_inversion(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Detect and nullify:
      (a) "Quiet" sensor drift — identical readings for ≥N consecutive hours
      (b) Cross-pollutant physical inversions — PM2.5 > PM10 (beyond tolerance)
    """
    log.info("  L3 │ Drift detection & cross-pollutant checks …")
    flatline_hrs = cfg["drift_detection"]["flatline_hours"]
    tolerance = cfg["drift_detection"]["pm25_pm10_tolerance"]

    # --- (a) Flatline / zero-variance detection ---
    for col in POLLUTANT_COLS:
        if col not in df.columns:
            continue

        series = df[col]
        # Find runs of identical values
        is_same = series.eq(series.shift(1))
        # Group consecutive identical values
        block_id = (~is_same).cumsum()
        block_sizes = is_same.groupby(block_id).transform("sum") + 1

        # Flag blocks that are ≥ flatline_hrs AND not already NaN
        flatline_mask = (block_sizes >= flatline_hrs) & series.notna()
        n_flagged = flatline_mask.sum()
        if n_flagged > 0:
            df.loc[flatline_mask, col] = np.nan
            log.info("  L3 │   %s: flagged %d flatline readings", col, n_flagged)

    # --- (b) Cross-pollutant inversion: PM2.5 > PM10 ---
    if "pm25" in df.columns and "pm10" in df.columns:
        both_valid = df["pm25"].notna() & df["pm10"].notna()
        # PM2.5 exceeds PM10 beyond the tolerance margin
        inverted = both_valid & (df["pm25"] > df["pm10"] * (1 + tolerance))
        n_inv = inverted.sum()
        if n_inv > 0:
            df.loc[inverted, ["pm25", "pm10"]] = np.nan
            log.info("  L3 │   Cross-pollutant inversion: nullified %d rows", n_inv)

    return df


# ═════════════════════════════════════════════════════════════
# LAYER 4 — Quality Assessment & Station Filtering
# ═════════════════════════════════════════════════════════════
def layer4_chunk_splitting(
    df: pd.DataFrame, cfg: dict, station_name: str, base_report: dict
) -> List[Tuple[Optional[pd.DataFrame], dict]]:
    """
    Split the station dataframe into isolated chunks at gaps > max_gap.
    Evaluate each chunk against local completeness and min_chunk_hours.
    Returns a list of chunks and their metadata reports.
    """
    log.info("  L4 │ Chunk Splitting & Quality Assessment …")
    
    min_comp = cfg["quality_filter"]["min_completeness"]
    max_gap = cfg["quality_filter"]["max_contiguous_gap_hours"]
    min_len = cfg["quality_filter"]["min_chunk_hours"]
    target_features = cfg.get("windowing", {}).get("target_features", ["pm25"])
    
    primary = target_features[0]
    if primary not in df.columns:
        log.warning("  L4 │   ✗ Missing primary target %s — dropping %s", primary, station_name)
        return []

    is_na = df[primary].isna()
    blocks = (is_na != is_na.shift()).cumsum()
    block_sizes = df.groupby(blocks)[primary].transform('size')
    
    # Fracture gap if it is a missing block and length > max_gap
    is_fracture_gap = is_na & (block_sizes > max_gap)
    
    # We want to identify contiguous blocks of non-fracture data.
    chunk_mask = ~is_fracture_gap
    if not chunk_mask.any():
        return []

    df_valid = df[chunk_mask].copy()
    time_diff = df_valid.index.to_series().diff()
    chunk_ids = (time_diff > pd.Timedelta(hours=max_gap)).cumsum()
    
    valid_chunks = []
    chunk_idx = 1
    
    for _, chunk in df_valid.groupby(chunk_ids):
        chunk_len = len(chunk)
        if chunk_len == 0:
            continue
            
        start_ts = chunk.index.min()
        end_ts = chunk.index.max()
        
        target_cols = [c for c in target_features if c in chunk.columns]
        if not target_cols:
            continue
            
        local_comps = []
        for c in target_cols:
            c_comp = chunk[c].notna().sum() / chunk_len
            local_comps.append(c_comp)
        local_completeness = np.mean(local_comps)
        
        report = base_report.copy()
        report.update({
            "Chunk_ID": f"{station_name}_C{chunk_idx}",
            "Chunk_Start_Date": start_ts.strftime("%Y-%m-%d"),
            "Chunk_End_Date": end_ts.strftime("%Y-%m-%d"),
            "Local_Completeness": round(local_completeness, 4),
            "Total_Valid_Continuous_Hours_Extracted": 0,
        })
        
        if chunk_len < min_len:
            report["status"] = f"DROPPED (len {chunk_len}h < {min_len}h)"
            valid_chunks.append((None, report))
        elif local_completeness < min_comp:
            report["status"] = f"DROPPED (comp {local_completeness:.1%} < {min_comp:.1%})"
            valid_chunks.append((None, report))
        else:
            report["status"] = "KEPT"
            # Reindex to its own continuous boundary to ensure contiguous grid for L5/L6
            full_chunk_idx = pd.date_range(start=start_ts, end=end_ts, freq="h")
            chunk = chunk.reindex(full_chunk_idx)
            valid_chunks.append((chunk, report))
            
            log.info(
                "  L4 │   ✓ %s kept (len: %dh, comp: %.1f%%)",
                report["Chunk_ID"], chunk_len, local_completeness * 100
            )
            
        chunk_idx += 1

    return valid_chunks


# ═════════════════════════════════════════════════════════════
# LAYER 5 — Safe Micro-Imputation
# ═════════════════════════════════════════════════════════════
def layer5_micro_imputation(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Linear-interpolate ONLY contiguous gaps of ≤ max_gap_hours.
    Gaps longer than the threshold are left as NaN (handled by Layer 6).

    Interpolation is applied per-column (per-station is guaranteed by the
    outer loop that calls this function station-by-station).
    """
    max_gap = cfg["imputation"]["max_gap_hours"]
    log.info("  L5 │ Micro-imputation (≤%dh gaps) …", max_gap)

    for col in POLLUTANT_COLS:
        if col not in df.columns:
            continue

        series = df[col].copy()
        is_nan = series.isna()

        if not is_nan.any():
            continue

        # Identify contiguous NaN blocks and their lengths
        block_id = (~is_nan).cumsum()
        # Only look at NaN positions
        nan_blocks = is_nan[is_nan].groupby(block_id[is_nan])
        gap_sizes = nan_blocks.transform("size")

        # Build mask: positions that are NaN AND belong to a short-enough gap
        eligible = pd.Series(False, index=df.index)
        eligible.loc[gap_sizes.index] = gap_sizes <= max_gap

        # Interpolate the full column, then only keep short-gap fills
        interpolated = series.interpolate(method="time")
        filled = series.copy()
        filled.loc[eligible] = interpolated.loc[eligible]

        n_filled = int((series.isna() & filled.notna()).sum())
        df[col] = filled
        if n_filled > 0:
            log.info("  L5 │   %s: interpolated %d values", col, n_filled)

    return df


# ═════════════════════════════════════════════════════════════
# LAYER 6 — Smart Sequence Windowing
# ═════════════════════════════════════════════════════════════
def layer6_sequence_windowing(
    df: pd.DataFrame, cfg: dict
) -> List[pd.DataFrame]:
    """
    Slice the cleaned dataframe into rolling (lookback + horizon) windows.
    Drop any window that still contains a NaN in the target pollutant columns.

    Returns a list of DataFrames, each representing one valid training window.
    """
    lookback = cfg["windowing"]["lookback_hours"]
    horizon = cfg["windowing"]["horizon_hours"]
    stride = cfg["windowing"]["stride_hours"]
    window_len = lookback + horizon

    log.info(
        "  L6 │ Sequence windowing (lookback=%dh, horizon=%dh, stride=%dh) …",
        lookback,
        horizon,
        stride,
    )

    target_cols = cfg.get("windowing", {}).get("target_features", ["pm25"])
    # If a target feature doesn't even exist in the dataframe, all windows are inherently invalid
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        log.warning("  L6 │   ✗ Missing target features %s — dropping all windows", missing_targets)
        return []

    windows = []
    n_total = 0
    n_dropped = 0

    for start_idx in range(0, len(df) - window_len + 1, stride):
        n_total += 1
        window = df.iloc[start_idx : start_idx + window_len]

        # Check if any target column has NaN within this window
        if window[target_cols].isna().any().any():
            n_dropped += 1
            continue

        windows.append(window)

    log.info(
        "  L6 │   Generated %d valid windows (dropped %d / %d total)",
        len(windows),
        n_dropped,
        n_total,
    )
    return windows


# ═════════════════════════════════════════════════════════════
# ORCHESTRATOR — Run all layers per-station
# ═════════════════════════════════════════════════════════════
def clean_station(
    raw_path: Path,
    station_name: str,
    pipeline_cfg: dict,
    date_range: dict,
) -> Tuple[Optional[pd.DataFrame], List[pd.DataFrame], List[dict]]:
    """
    Run Layers 1–3 on a single station CSV, split into chunks in L4,
    and run L5-L6 on each chunk.
    """
    log.info("━━━ Processing: %s ━━━", station_name)

    # Load raw CSV
    df = pd.read_csv(raw_path, index_col=0)

    # Layer 1
    df = layer1_standardize_index(df, date_range["start"], date_range["end"])

    station_report = {"City/Station": station_name}

    # Layer 2
    df = layer2_artifact_filter(df, pipeline_cfg, station_report)

    # Layer 3
    df = layer3_drift_and_inversion(df, pipeline_cfg)

    # Layer 4 (Chunk Splitting)
    chunks_info = layer4_chunk_splitting(df, pipeline_cfg, station_name, station_report)
    
    all_windows = []
    all_reports = []
    all_cleaned = []

    for chunk_df, c_report in chunks_info:
        if chunk_df is None:
            all_reports.append(c_report)
            continue

        # Layer 5 (Micro-imputation isolated to chunk)
        chunk_df = layer5_micro_imputation(chunk_df, pipeline_cfg)
        
        all_cleaned.append(chunk_df)

        # Layer 6 (Sequence Windowing isolated to chunk)
        windows = layer6_sequence_windowing(chunk_df, pipeline_cfg)
        
        c_report["Total_Valid_Continuous_Hours_Extracted"] = sum(len(w) for w in windows)
        
        all_windows.extend(windows)
        all_reports.append(c_report)

    final_cleaned_df = pd.concat(all_cleaned) if all_cleaned else None

    return final_cleaned_df, all_windows, all_reports


def main() -> None:
    """Run the full cleaning pipeline across all raw station files."""
    pipeline_cfg = _load_pipeline_cfg()
    station_cfg = _load_station_cfg()
    date_range = station_cfg["date_range"]

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*_raw.csv"))
    if not raw_files:
        log.error("No raw files found in %s. Run fetch_cpcb.py first.", RAW_DIR)
        return

    all_reports = []
    all_windows = []

    for raw_path in raw_files:
        # Derive station name from filename
        station_name = raw_path.stem.replace("_raw", "").replace("_", " ").title()

        cleaned, windows, reports = clean_station(
            raw_path, station_name, pipeline_cfg, date_range
        )
        all_reports.extend(reports)

        if cleaned is not None:
            # Save the cleaned (post Layer-5) station data
            out_path = PROC_DIR / raw_path.name.replace("_raw.csv", "_cleaned.csv")
            cleaned.to_csv(out_path)
            log.info("  → Saved cleaned data: %s", out_path.name)

        all_windows.extend(windows)

    # ── Quality Summary ──────────────────────────────────────
    log.info("═" * 60)
    log.info("QUALITY SUMMARY")
    log.info("═" * 60)
    report_df = pd.DataFrame(all_reports)
    summary_cols = [
        "City/Station", 
        "Chunk_ID", 
        "Chunk_Start_Date", 
        "Chunk_End_Date", 
        "Local_Completeness", 
        "Total_Valid_Continuous_Hours_Extracted", 
        "status"
    ]
    display_df = report_df[[c for c in summary_cols if c in report_df.columns]]
    log.info("\n%s", display_df.to_string(index=False))

    # Save quality report
    report_path = PROC_DIR / "quality_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info("Quality report saved to %s", report_path)

    # ── Save windowed sequences metadata ─────────────────────
    log.info(
        "Total valid training windows across all stations: %d", len(all_windows)
    )
    if all_windows:
        # Save a summary (first/last timestamp per window) for downstream use
        window_meta = []
        for i, w in enumerate(all_windows):
            window_meta.append(
                {
                    "window_id": i,
                    "start": w.index[0],
                    "end": w.index[-1],
                    "length_hours": len(w),
                }
            )
        meta_df = pd.DataFrame(window_meta)
        meta_path = PROC_DIR / "window_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        log.info("Window metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
