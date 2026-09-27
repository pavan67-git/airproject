import pandas as pd
import numpy as np
from pathlib import Path

def get_bin_weights():
    bins = [-np.inf, 30.0, 60.0, 90.0, 120.0, 250.0, 400.0, np.inf]
    bin_labels = ["< 30", "30-60", "60-90", "90-120", "120-250", "250-400", ">= 400"]
    beta = 0.999
    
    # Load training stations only
    train_stations = ["delhi", "kolkata", "hyderabad"] # Ahmedabad is training, but maninagar is val. Let's include ahmedabad.
    train_stations = ["delhi_cleaned.csv", "kolkata_cleaned.csv", "hyderabad_cleaned.csv", "ahmedabad_cleaned.csv"]
    
    proc_dir = Path("data/processed")
    
    all_pm25 = []
    for f in train_stations:
        path = proc_dir / f
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            if "pm25" in df.columns:
                all_pm25.extend(df["pm25"].dropna().values)
                
    all_pm25 = np.array(all_pm25)
    
    hist, _ = np.histogram(all_pm25, bins=bins)
    
    print("| PM2.5 Bin ($\\mu g/m^3$) | Training Count ($n_k$) | Raw Weight ($w_k$) | Normalized Weight |")
    print("|---|---|---|---|")
    
    effective_num = (1.0 - np.power(beta, hist))
    # Replace 0 to avoid division by zero (shouldn't happen with large dataset but safe)
    effective_num = np.where(effective_num == 0, 1, effective_num)
    
    raw_weights = (1.0 - beta) / effective_num
    normalized_weights = raw_weights / np.sum(raw_weights) * len(hist)
    
    for label, count, rw, nw in zip(bin_labels, hist, raw_weights, normalized_weights):
        print(f"| {label} | {count} | {rw:.6f} | {nw:.4f} |")

if __name__ == "__main__":
    get_bin_weights()
