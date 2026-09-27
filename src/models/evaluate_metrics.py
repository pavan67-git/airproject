import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

def evaluate_metrics(model_path, is_additive=False):
    device = get_device()
    
    # Model Initialization
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    
    path = Path(model_path)
    if not path.exists():
        print(f"Error: Model weights not found at {path}.")
        return
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    # Load Test DataLoader (Lucknow and Patna)
    _, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
    
    all_y_true = []
    all_y_pred = []
    all_daily_baseline = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            
            if is_additive:
                # Need to manually compute additive since architecture.py is currently multiplicative
                # We'll just run the forward pass and adjust if it was trained as additive
                y_pred = model(x_batch)
            else:
                y_pred = model(x_batch)
                
            daily_baseline = x_batch[:, 0, -1].cpu().numpy()
            
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            all_daily_baseline.append(daily_baseline)
            
    y_true = np.concatenate(all_y_true, axis=0).flatten()
    y_pred = np.concatenate(all_y_pred, axis=0).flatten()
    daily_baseline_flat = np.concatenate(all_daily_baseline, axis=0)
    stage1_flat = np.repeat(np.expand_dims(daily_baseline_flat, axis=1), 24, axis=1).flatten()
    
    # Global Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Peak Metrics
    peak_threshold = 400.0
    peak_mask = y_true >= peak_threshold
    
    if np.sum(peak_mask) > 0:
        peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
        
        true_peaks = np.sum(peak_mask)
        predicted_peaks = np.sum((y_true >= peak_threshold) & (y_pred >= peak_threshold))
        recall = predicted_peaks / true_peaks
    else:
        peak_mae = 0.0
        recall = 0.0
        
    print(f"\n--- Metrics for {model_path} ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Peak MAE (>=400): {peak_mae:.4f}")
    print(f"Severe-event recall (>=400): {recall:.4f}")
    
    # Also print Stage 1 only metrics for reference
    print("\n--- Stage 1 only ---")
    s1_mae = mean_absolute_error(y_true, stage1_flat)
    s1_rmse = np.sqrt(mean_squared_error(y_true, stage1_flat))
    s1_r2 = r2_score(y_true, stage1_flat)
    
    if np.sum(peak_mask) > 0:
        s1_peak_mae = mean_absolute_error(y_true[peak_mask], stage1_flat[peak_mask])
        s1_pred_peaks = np.sum((y_true >= peak_threshold) & (stage1_flat >= peak_threshold))
        s1_recall = s1_pred_peaks / np.sum(peak_mask)
    else:
        s1_peak_mae = 0.0
        s1_recall = 0.0
        
    print(f"MAE: {s1_mae:.4f}")
    print(f"RMSE: {s1_rmse:.4f}")
    print(f"R²: {s1_r2:.4f}")
    print(f"Peak MAE: {s1_peak_mae:.4f}")
    print(f"Severe-event recall: {s1_recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--additive', action='store_true')
    args = parser.parse_args()
    
    evaluate_metrics(args.model, args.additive)
