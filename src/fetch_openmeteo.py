import pandas as pd
import numpy as np
import requests
import yaml
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
)
log = logging.getLogger("fetch_openmeteo")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"
CONFIG_PATH = ROOT_DIR / "configs" / "stations.yaml"
REPORT_PATH = PROC_DIR / "quality_report.csv"

def get_station_coords(station_name: str, config: dict):
    for st in config["stations"]:
        if st["name"].lower() == station_name.lower():
            return st.get("latitude"), st.get("longitude")
    return None, None

def fetch_openmeteo_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    # Add ±2 day padding to prevent timezone Off-By-One glitch
    start_ts = pd.to_datetime(start_date) - pd.Timedelta(days=2)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=2)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m,boundary_layer_height",
        "timezone": "Asia/Kolkata"
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    if "hourly" not in data:
        raise ValueError("API response missing 'hourly' data")
        
    df = pd.DataFrame(data["hourly"])
    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index)
    # Ensure index is exactly tz-aware Asia/Kolkata for perfect pandas join
    df.index = df.index.tz_localize(None).tz_localize("Asia/Kolkata")
    
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Meteorological Wind Vectors (Where wind is GOING, not coming from)
    df["U_wind"] = -df["wind_speed_10m"] * np.sin(df["wind_direction_10m"] * np.pi / 180)
    df["V_wind"] = -df["wind_speed_10m"] * np.cos(df["wind_direction_10m"] * np.pi / 180)
    
    # Vertical Dispersion Factor (safe clip to prevent zero division)
    df["inv_PBLH"] = 1.0 / df["boundary_layer_height"].clip(lower=10.0)
    
    # Cyclic Time Encodings
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12.0)
    
    # Drop raw columns to prevent multicollinearity
    cols_to_drop = ["wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    
    return df

def main():
    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
        
    # Load quality report
    if not REPORT_PATH.exists():
        log.error("Quality report not found: %s", REPORT_PATH)
        return
        
    report_df = pd.read_csv(REPORT_PATH)
    kept_chunks = report_df[report_df["status"] == "KEPT"]
    
    if kept_chunks.empty:
        log.warning("No kept chunks found in report.")
        return
        
    log.info("Found %d KEPT chunks to process.", len(kept_chunks))
    
    # Cache loaded CPCB DataFrames to avoid repeated IO
    cpcb_cache = {}
    
    for idx, row in kept_chunks.iterrows():
        station_name = row["City/Station"]
        chunk_id = row["Chunk_ID"]
        start_date = row["Chunk_Start_Date"]
        end_date = row["Chunk_End_Date"]
        
        log.info("━━━ Processing Chunk: %s ━━━", chunk_id)
        
        # Get coords
        lat, lon = get_station_coords(station_name, config)
        if lat is None or lon is None:
            log.error("Coordinates not found for station: %s", station_name)
            continue
            
        try:
            # 1. Fetch
            log.info("  → Fetching Open-Meteo data (lat: %s, lon: %s, %s to %s)", lat, lon, start_date, end_date)
            meteo_df = fetch_openmeteo_data(lat, lon, start_date, end_date)
            
            # 2. Engineer Features
            meteo_df = feature_engineer(meteo_df)
            
            # 3. Load & slice CPCB Data
            cpcb_file = PROC_DIR / f"{station_name.lower().replace(' ', '_')}_cleaned.csv"
            if station_name not in cpcb_cache:
                cpcb_cache[station_name] = pd.read_csv(cpcb_file, index_col=0)
                # Ensure index is parsed properly
                cpcb_cache[station_name].index = pd.to_datetime(cpcb_cache[station_name].index)
                
            cpcb_full = cpcb_cache[station_name]
            # Slice strictly to chunk boundaries
            cpcb_chunk = cpcb_full.loc[start_date:end_date]
            
            orig_rows = len(cpcb_chunk)
            
            # 4. Merge
            merged_df = cpcb_chunk.join(meteo_df, how="inner")
            
            # Drop NaNs strictly in the weather features (maintain CPCB rows)
            weather_cols = ["temperature_2m", "relative_humidity_2m", "surface_pressure", 
                            "precipitation", "U_wind", "V_wind", "inv_PBLH", 
                            "hour_sin", "hour_cos", "month_sin", "month_cos"]
            
            # Ensure we only check columns that exist in the merge
            check_cols = [c for c in weather_cols if c in merged_df.columns]
            merged_df.dropna(subset=check_cols, inplace=True)
            merged_rows = len(merged_df)
            
            log.info("  → Verification: Original_CPCB_Rows=%d | Merged_Rows=%d", orig_rows, merged_rows)
            
            # Save
            out_file = PROC_DIR / f"{chunk_id}_multimodal.csv"
            merged_df.to_csv(out_file)
            log.info("  → Saved %s", out_file.name)
            
        except Exception as e:
            log.error("Failed processing %s: %s", chunk_id, e)
            
        # 5. Rate limiting
        time.sleep(1.5)

if __name__ == "__main__":
    main()
