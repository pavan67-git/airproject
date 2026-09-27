import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class AirQualityDataset(Dataset):
    def __init__(self, chunk_ids, hourly_dir='data/final_multimodal', daily_dir='data/processed/stage1_daily', seq_length=24, pred_length=24):
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        self.X_data = []
        self.Y_data = []
        
        # Feature ordering: Dynamic temporal (9) first, then Static contextual (6)
        # The 16th feature (daily baseline) is appended later in the sliding window
        self.features = [
            # Dynamic temporal (10) → CNN/BiLSTM path
            'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'precipitation',
            'U_wind', 'V_wind', 'inv_PBLH', 'Upwind_FRP', 'hour_sin', 'hour_cos',
            # Static contextual (6) → Late fusion path
            'elevation', 'pop_density', 'ndvi', 'ndbi', 'month_sin', 'month_cos',
        ]
        
        hourly_path = Path(hourly_dir)
        daily_path = Path(daily_dir)
        
        for chunk_id in chunk_ids:
            h_file = hourly_path / f"{chunk_id}_multimodal.csv"
            d_file = daily_path / f"{chunk_id}_stage1.csv"
            
            if not h_file.exists() or not d_file.exists():
                print(f"Skipping {chunk_id}: Files missing.")
                continue
                
            df_h = pd.read_csv(h_file, index_col=0, parse_dates=True)
            df_d = pd.read_csv(d_file, index_col=0, parse_dates=True)
            
            # Lookup dictionary for daily pm25 constraint
            df_d['date_str'] = df_d.index.strftime('%Y-%m-%d')
            daily_dict = dict(zip(df_d['date_str'], df_d['pm25']))
            
            X_arr = df_h[self.features].values
            Y_arr = df_h['pm25'].values
            times = df_h.index
            
            total_len = len(df_h)
            window_size = self.seq_length + self.pred_length
            
            # Sliding window over the chunk
            for i in range(total_len - window_size + 1):
                # T is the start of the forecast window
                T_idx = i + self.seq_length
                T_date_str = times[T_idx].strftime('%Y-%m-%d')
                
                # Fetch Stage 1 daily constraint for the target forecast day
                if T_date_str in daily_dict:
                    daily_pm25 = daily_dict[T_date_str]
                else:
                    continue
                    
                # Extract concurrent weather forecast X (T to T+24)
                x_hist = X_arr[T_idx:T_idx + self.pred_length].copy()
                
                # Append stage1_daily_pm25 constraint feature (16th feature, last column)
                daily_feature = np.full((self.seq_length, 1), daily_pm25)
                x_hist = np.hstack([x_hist, daily_feature])
                
                # Extract forecast target Y (T to T+24)
                y_future = Y_arr[T_idx:T_idx + self.pred_length].copy()
                
                # Check for NaNs
                if np.isnan(x_hist).any() or np.isnan(y_future).any():
                    continue
                    
                self.X_data.append(x_hist)
                self.Y_data.append(y_future)
                
        self.X_data = np.array(self.X_data, dtype=np.float32)
        self.Y_data = np.array(self.Y_data, dtype=np.float32)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return torch.tensor(self.X_data[idx]), torch.tensor(self.Y_data[idx])
