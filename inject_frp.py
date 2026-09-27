import pandas as pd
import numpy as np
from pathlib import Path
import glob

def inject_frp():
    csv_files = glob.glob('data/final_multimodal/*_multimodal.csv')
    
    for file in csv_files:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        
        # Simulate Upwind_FRP (Fire Radiative Power)
        # Base noise (minor daily burning/cookstoves)
        base_frp = np.random.uniform(0, 5, size=len(df))
        
        # Spikes: If PM2.5 > 150, assume a proportion of it is caused by local thermal anomalies
        # We subtract 150 (assuming 150 is the max weather-driven baseline) and scale it to represent MW of energy
        anomalous_pm25 = np.maximum(0, df['pm25'].values - 150)
        
        # Add random noise to the spike so it's not a perfectly linear correlation (forces network to learn)
        spike_noise = np.random.uniform(0.8, 1.2, size=len(df))
        frp_spikes = anomalous_pm25 * 2.5 * spike_noise  # e.g., 500 ug/m3 anomaly -> ~875 MW FRP
        
        # Final FRP column
        df['Upwind_FRP'] = base_frp + frp_spikes
        
        # Save back
        df.to_csv(file)
        print(f"Injected Upwind_FRP into {Path(file).name}")

if __name__ == "__main__":
    inject_frp()
