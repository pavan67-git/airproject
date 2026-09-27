import pandas as pd
import numpy as np
from pathlib import Path

def generate_qc_stats():
    raw_dir = Path("data/raw")
    
    print("| Station | Raw records | Negative removed | Duplicate removed | Freeze records removed | Interpolated input hours | Observed evaluation hours |")
    print("|---|---|---|---|---|---|---|")
    
    # Process only the 8 train/val/test stations we care about
    target_stations = ["delhi", "kolkata", "hyderabad", "ahmedabad", "maninagar_ahmedabad_gpcb", "lucknow", "patna"]
    
    for station in target_stations:
        file_path = raw_dir / f"{station}_raw.csv"
        if not file_path.exists():
            continue
            
        df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
        col = "pm25"
        if col not in df_raw.columns:
            continue
            
        # 1. Raw records (including all timestamps in the original file)
        raw_records = len(df_raw)
        
        # 2. Duplicate removed (when standardizing index)
        # Count how many duplicates there are
        dups = df_raw.index.duplicated().sum()
        
        # Drop duplicates for the rest of processing (mean)
        df = df_raw.groupby(df_raw.index).mean()
        
        # Grid it
        start_ts = df.index.min()
        end_ts = df.index.max()
        if pd.isna(start_ts):
            continue
            
        full_range = pd.date_range(start=start_ts, end=end_ts, freq="h")
        df = df.reindex(full_range)
        
        # 3. Negative removed
        mask_neg = (df[col] <= 0)
        neg_removed = mask_neg.sum()
        df.loc[mask_neg, col] = np.nan
        
        # Upper bound (1000)
        df.loc[df[col] > 1000, col] = np.nan
        
        # Adaptive baseline (approx)
        station_mean = df[col].mean()
        if not pd.isna(station_mean):
            dynamic_threshold = max(station_mean * 0.05, 3.0)
            rolling_std = df[col].rolling(window=12, min_periods=2).std()
            noise_mask = (df[col] < dynamic_threshold) & (rolling_std < 1.0)
            neg_removed += noise_mask.sum()
            df.loc[noise_mask, col] = np.nan
        
        # 4. Freeze removed (>= 6 hours flatline)
        series = df[col]
        is_same = series.eq(series.shift(1))
        block_id = (~is_same).cumsum()
        block_sizes = is_same.groupby(block_id).transform("sum") + 1
        flatline_mask = (block_sizes >= 6) & series.notna()
        freeze_removed = flatline_mask.sum()
        df.loc[flatline_mask, col] = np.nan
        
        # 5. Interpolated input hours (<= 3 hours gap)
        is_nan = df[col].isna()
        block_id = (~is_nan).cumsum()
        nan_blocks = is_nan[is_nan].groupby(block_id[is_nan])
        gap_sizes = nan_blocks.transform("size")
        
        eligible = pd.Series(False, index=df.index)
        eligible.loc[gap_sizes.index] = gap_sizes <= 3
        
        interpolated = df[col].interpolate(method="time")
        filled = df[col].copy()
        filled.loc[eligible] = interpolated.loc[eligible]
        
        interpolated_hours = (df[col].isna() & filled.notna()).sum()
        
        # 6. Observed evaluation hours (valid non-nan values BEFORE imputation)
        observed_eval = df[col].notna().sum()
        
        print(f"| {station.title()} | {raw_records} | {neg_removed} | {dups} | {freeze_removed} | {interpolated_hours} | {observed_eval} |")

if __name__ == "__main__":
    generate_qc_stats()
