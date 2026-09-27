"""
inject_static_features.py
=========================
Extracts micro-spatial static land-use features from Google Earth Engine (GEE)
within a 1km buffer around each monitoring station and broadcasts these values 
to all Stage 1 Daily and Stage 2 Hourly datasets.

Features Extracted (within 1km radial buffer):
    - elevation (m): USGS/SRTMGL1_003
    - pop_density (persons/cell): WorldPop/GP/100m/pop_age_sex_cons_unadj (2020)
    - ndvi (Normalized Difference Vegetation Index): COPERNICUS/S2_HARMONIZED (2023 median, cloud < 30%)
    - ndbi (Normalized Difference Built-up Index): COPERNICUS/S2_HARMONIZED (2023 median, cloud < 30%)

Run with:
    python -m src.inject_static_features --project YOUR_GEE_PROJECT_ID
"""

import argparse
import logging
import time
import socket
from pathlib import Path

import ee
import pandas as pd
import yaml

# ── Force IPv4 resolution to prevent macOS IPv6 TCP timeouts ──
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(*args, **kwargs):
    res = orig_getaddrinfo(*args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET] or res
socket.getaddrinfo = getaddrinfo_ipv4

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("static_features")

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "stations.yaml"
HOURLY_DIR = ROOT_DIR / "data" / "final_multimodal"
DAILY_DIR = ROOT_DIR / "data" / "processed" / "stage1_daily"


# ═════════════════════════════════════════════════════════════
# STEP 1 — GEE STATIC FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════
def extract_station_static_features(lat: float, lon: float, station_name: str) -> dict:
    """
    Extract spatial mean of Elevation, Population Density, NDVI, and NDBI
    within a 1km radius of the station coordinates using GEE.
    """
    point = ee.Geometry.Point(lon, lat)
    buffer = point.buffer(1000)  # 1km radius

    # 1. Elevation (SRTM ~30m)
    elev_img = ee.Image("USGS/SRTMGL1_003").select("elevation")
    elev_val = elev_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30,
        bestEffort=True,
    ).get("elevation")

    # 2. Population Density (WorldPop ~100m, year 2020)
    pop_col = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex_cons_unadj").filter(
        ee.Filter.equals("year", 2020)
    )
    pop_img = pop_col.mean().select("population")
    pop_val = pop_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=100,
        bestEffort=True,
    ).get("population")

    # 3. Sentinel-2 NDVI & NDBI (2023 cloud-free median composite)
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate("2023-01-01", "2023-12-31")
        .filterBounds(buffer)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    s2_median = s2_col.median()

    # NDVI: (NIR - RED) / (NIR + RED) -> (B8 - B4) / (B8 + B4)
    ndvi_img = s2_median.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndvi_val = ndvi_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30,
        bestEffort=True,
    ).get("ndvi")

    # NDBI: (SWIR1 - NIR) / (SWIR1 + NIR) -> (B11 - B8) / (B11 + B8)
    ndbi_img = s2_median.normalizedDifference(["B11", "B8"]).rename("ndbi")
    ndbi_val = ndbi_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30,
        bestEffort=True,
    ).get("ndbi")

    # Fetch values simultaneously in one dictionary evaluation to minimize latency
    result_dict = ee.Dictionary({
        "elevation": elev_val,
        "pop_density": pop_val,
        "ndvi": ndvi_val,
        "ndbi": ndbi_val,
    }).getInfo()

    return {
        "station_name": station_name,
        "latitude": lat,
        "longitude": lon,
        "elevation": round(result_dict.get("elevation", 0.0), 3) if result_dict.get("elevation") is not None else 0.0,
        "pop_density": round(result_dict.get("pop_density", 0.0), 3) if result_dict.get("pop_density") is not None else 0.0,
        "ndvi": round(result_dict.get("ndvi", 0.0), 4) if result_dict.get("ndvi") is not None else 0.0,
        "ndbi": round(result_dict.get("ndbi", 0.0), 4) if result_dict.get("ndbi") is not None else 0.0,
    }


# ═════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Extract & Broadcast Static Land-Use Features")
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

    # ── Collect all processed files to identify active stations
    hourly_files = list(HOURLY_DIR.glob("*_multimodal.csv"))
    daily_files = list(DAILY_DIR.glob("*_stage1.csv"))

    log.info("Found %d hourly files and %d daily files to enhance.", len(hourly_files), len(daily_files))

    # Identify which stations actually appear in our chunks
    active_station_names = set()
    for f in hourly_files:
        # Example filename: Ahmedabad_C15_multimodal.csv or Maninagar Ahmedabad Gpcb_C18_multimodal.csv
        parts = f.stem.replace("_multimodal", "").split("_")
        station_name = parts[0]
        active_station_names.add(station_name)

    log.info("Active stations in dataset: %s", sorted(active_station_names))

    # ── Step 1: Extract Features for each station ────────────
    station_features = {}
    verification_records = []

    for st in config["stations"]:
        name = st["name"]
        if name not in active_station_names:
            log.info("Skipping station not in active chunks: %s", name)
            continue

        lat = st.get("latitude")
        lon = st.get("longitude")
        if not lat or not lon:
            log.error("Coordinates missing for %s", name)
            continue

        log.info("Extracting 1km GEE features for: %s (Lat: %s, Lon: %s) ...", name, lat, lon)
        try:
            feats = extract_station_static_features(lat, lon, name)
            station_features[name] = feats
            verification_records.append(feats)
            log.info("  ✓ %s: Elev=%.1f m | Pop=%.1f | NDVI=%.4f | NDBI=%.4f", 
                     name, feats["elevation"], feats["pop_density"], feats["ndvi"], feats["ndbi"])
            time.sleep(1.0)
        except Exception as e:
            log.error("Failed to extract GEE features for %s: %s", name, e)

    # ── Step 2: Verification Table ───────────────────────────
    log.info("═" * 80)
    log.info("GEOGRAPHIC PHYSICS VERIFICATION TABLE (1km Radial Buffer)")
    log.info("═" * 80)
    if verification_records:
        vdf = pd.DataFrame(verification_records)
        log.info("\n%s\n", vdf.to_string(index=False))
    else:
        log.error("No features extracted. Exiting.")
        return

    # ── Step 3: Broadcast to Hourly & Daily CSVs ─────────────
    log.info("Broadcasting static columns (elevation, pop_density, ndvi, ndbi) to CSV files ...")

    # Helper function to broadcast
    def update_files(file_list, suffix):
        count = 0
        for path in file_list:
            parts = path.stem.replace(suffix, "").split("_")
            station_name = parts[0]
            if station_name not in station_features:
                log.warning("No static features found for %s (%s)", station_name, path.name)
                continue

            feats = station_features[station_name]
            df = pd.read_csv(path, index_col=0)

            # Assign columns
            df["elevation"] = feats["elevation"]
            df["pop_density"] = feats["pop_density"]
            df["ndvi"] = feats["ndvi"]
            df["ndbi"] = feats["ndbi"]

            df.to_csv(path)
            count += 1
        log.info("  ✓ Successfully updated %d files with suffix '%s'", count, suffix)

    update_files(hourly_files, "_multimodal")
    update_files(daily_files, "_stage1")

    log.info("All datasets enhanced successfully with Zero-Shot micro-spatial context!")


if __name__ == "__main__":
    main()
