import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

def evaluate():
    device = get_device()
    
    # 1. Model Initialization & Inference
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    model_path = Path("models/saved/best_hybrid_model.pth")
    
    if not model_path.exists():
        print(f"Error: Model weights not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model weights loaded successfully.")
    
    # Load Test DataLoader (using is_training=False to load saved scaler)
    _, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
    print("Test DataLoader loaded successfully.")
    
    all_y_true = []
    all_y_pred = []
    
    print("Running inference over the Test Set (Zero-Shot: Lucknow & Patna)...")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            
    y_true_array = np.concatenate(all_y_true, axis=0)
    y_pred_array = np.concatenate(all_y_pred, axis=0)
    
    # 2. No Inverse Transform needed (already in physical units)
    y_true_raw = y_true_array
    y_pred_raw = y_pred_array
    
    # Physical safety: enforce sensor detection limit
    y_pred_raw = np.clip(y_pred_raw, a_min=1.0, a_max=None)
    
    y_true_filtered = y_true_raw
    y_pred_filtered = y_pred_raw
    
    # Calculate Research Metrics
    y_true_flat = y_true_filtered.flatten()
    y_pred_flat = y_pred_filtered.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    print("\n" + "="*50)
    print("ZERO-SHOT EVALUATION METRICS (LUCKNOW & PATNA)")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.4f} µg/m³")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} µg/m³")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print("="*50)
    
    # 3. Visualizing the "Peak-Smoothing" Solution
    max_values = np.max(y_true_raw, axis=1)
    peak_indices = np.where(max_values > 200)[0]
    
    if len(peak_indices) > 0:
        best_idx = peak_indices[np.argmax(max_values[peak_indices])]
        print(f"\nFound sequence with peak PM2.5 > 200 at index {best_idx}.")
    else:
        best_idx = np.argmax(max_values)
        print(f"\nNo peak > 200 found. Falling back to max available peak at index {best_idx}.")
        
    y_true_seq = y_true_raw[best_idx]
    y_pred_seq = y_pred_raw[best_idx]
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "peak_prediction_lucknow.png"
    
    plt.figure(figsize=(12, 6), facecolor='white')
    hours = np.arange(1, 25)
    
    plt.plot(hours, y_true_seq, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Ground Truth ($PM_{2.5}$)')
    plt.plot(hours, y_pred_seq, marker='s', linestyle='--', color='#d62728', linewidth=2.5, markersize=8, label='Predicted ($\mathcal{L}_{fMAE}$ Optimized)')
    
    plt.title("Zero-Shot Diurnal Forecast (Unseen City: Lucknow/Patna)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Hour (T+1 to T+24)", fontsize=14, labelpad=10)
    plt.ylabel("$PM_{2.5}$ Concentration (µg/m³)", fontsize=14, labelpad=10)
    plt.xticks(hours, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    peak_hour = hours[np.argmax(y_true_seq)]
    peak_val = np.max(y_true_seq)
    plt.axvline(x=peak_hour, color='gray', linestyle=':', alpha=0.5)
    plt.annotate(f'Severe Smog Peak\n({peak_val:.1f} µg/m³)', 
                 xy=(peak_hour, peak_val), xytext=(peak_hour-3, peak_val+20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    evaluate()
