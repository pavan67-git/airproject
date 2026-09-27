import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models.dataset import AirQualityDataset
from torch.utils.data import DataLoader

def evaluate_stage1_daily(chunk_ids, name):
    print(f"\nEvaluating Stage 1 for {name} ({chunk_ids})")
    dataset = AirQualityDataset(chunk_ids)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_daily_pred = []
    all_daily_true = []
    
    for x_batch, y_batch in loader:
        # x_batch shape: [Batch, seq_length=24, features=16]
        # The 16th feature (index 15) is the Stage 1 daily baseline.
        # Since it's repeated for the sequence, we just take the first timestep.
        daily_baseline = x_batch[:, 0, -1].numpy()
        
        # Ground truth is hourly for the next 24 hours. [Batch, 24]
        # We need to average it to get the daily mean true value.
        daily_true = y_batch.numpy().mean(axis=1)
        
        all_daily_pred.append(daily_baseline)
        all_daily_true.append(daily_true)
        
    if not all_daily_pred:
        print(f"No data for {name}")
        return
        
    daily_pred = np.concatenate(all_daily_pred, axis=0)
    daily_true = np.concatenate(all_daily_true, axis=0)
    
    mae = mean_absolute_error(daily_true, daily_pred)
    rmse = np.sqrt(mean_squared_error(daily_true, daily_pred))
    r2 = r2_score(daily_true, daily_pred)
    
    print(f"  Daily MAE:  {mae:.4f}")
    print(f"  Daily RMSE: {rmse:.4f}")
    print(f"  Daily R²:   {r2:.4f}")

if __name__ == "__main__":
    evaluate_stage1_daily(['Maninagar Ahmedabad Gpcb_C18'], "Validation (Maninagar)")
    evaluate_stage1_daily(['Lucknow_C1'], "Lucknow (Zero-Shot)")
    evaluate_stage1_daily(['Patna_C20'], "Patna (Zero-Shot)")
