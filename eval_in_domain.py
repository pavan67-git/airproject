import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataset import AirQualityDataset
from src.models.dataloaders import SCALE_INDICES
from src.models.train import get_device

def evaluate_in_domain():
    device = get_device()
    
    # 1. Model Initialization
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
    model.eval()
    
    scaler = joblib.load("models/saved/scaler.joblib")
    
    # Load a specific training station for in-domain evaluation
    station_id = 'Delhi_C14'
    print(f"Loading In-Domain Station: {station_id}")
    test_dataset = AirQualityDataset([station_id])
    
    # Apply input scaling
    if len(test_dataset.X_data) > 0:
        orig_shape = test_dataset.X_data[:, :, SCALE_INDICES].shape
        X_flat = test_dataset.X_data[:, :, SCALE_INDICES].reshape(-1, len(SCALE_INDICES))
        X_scaled = scaler.transform(X_flat).reshape(orig_shape)
        test_dataset.X_data[:, :, SCALE_INDICES] = X_scaled
        
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    
    print("Running inference over In-Domain data...")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            
    y_true_raw = np.concatenate(all_y_true, axis=0)
    y_pred_raw = np.concatenate(all_y_pred, axis=0)
    
    # Physical safety
    y_pred_raw = np.clip(y_pred_raw, a_min=1.0, a_max=None)
    
    # Metrics
    y_true_flat = y_true_raw.flatten()
    y_pred_flat = y_pred_raw.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    print("\n" + "="*50)
    print(f"IN-DOMAIN EVALUATION METRICS ({station_id})")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.4f} µg/m³")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} µg/m³")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    
    # Visualizing Peak
    max_values = np.max(y_true_raw, axis=1)
    peak_indices = np.where(max_values > 300)[0]
    
    if len(peak_indices) > 0:
        best_idx = peak_indices[np.argmax(max_values[peak_indices])]
        print(f"\nFound extreme peak PM2.5 > 300 at index {best_idx}.")
    else:
        best_idx = np.argmax(max_values)
        print(f"\nNo peak > 300 found. Falling back to max available at index {best_idx}.")
        
    y_true_seq = y_true_raw[best_idx]
    y_pred_seq = y_pred_raw[best_idx]
    
    # What was the daily baseline constraint?
    # Reconstruct original dataset to check 16th feature
    dummy_ds = AirQualityDataset([station_id])
    daily_baseline = dummy_ds.X_data[best_idx, 0, -1]
    
    print(f"Daily Baseline for this peak: {daily_baseline:.1f} µg/m³")
    print(f"Ground Truth Range: {y_true_seq.min():.1f} -> {y_true_seq.max():.1f}")
    print(f"Prediction Range:   {y_pred_seq.min():.1f} -> {y_pred_seq.max():.1f}")
    
    plot_path = f"results/in_domain_peak_{station_id}.png"
    plt.figure(figsize=(12, 6), facecolor='white')
    hours = np.arange(1, 25)
    
    plt.plot(hours, y_true_seq, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Ground Truth')
    plt.plot(hours, y_pred_seq, marker='s', linestyle='--', color='#d62728', linewidth=2.5, markersize=8, label='Predicted (Ratio x Baseline)')
    plt.axhline(y=daily_baseline, color='green', linestyle=':', label='Stage 1 Daily Baseline')
    
    plt.title(f"In-Domain Diurnal Forecast ({station_id})", fontsize=16, fontweight='bold')
    plt.xlabel("Hour (T+1 to T+24)", fontsize=14)
    plt.ylabel("$PM_{2.5}$ Concentration (µg/m³)", fontsize=14)
    plt.xticks(hours)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    evaluate_in_domain()
