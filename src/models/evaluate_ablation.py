import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

def evaluate_ablation():
    device = get_device()
    
    # 1. Model Initialization
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    model_path = Path("models/saved/best_hybrid_model.pth")
    
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Test DataLoader
    _, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
    
    all_y_true = []
    all_y_pred = []
    all_daily_baseline = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            
            # Extract daily_baseline from x_batch (from the first timestep, last feature)
            daily_baseline = x_batch[:, 0, -1].cpu().numpy()
            
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            all_daily_baseline.append(daily_baseline)
            
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    daily_baseline = np.concatenate(all_daily_baseline, axis=0)
    
    # Reverse-calculate predicted ratio
    daily_baseline_expanded = np.expand_dims(daily_baseline, axis=1) # [Batch, 1]
    daily_baseline_safe = np.maximum(daily_baseline_expanded, 1e-5)
    
    if np.any(daily_baseline_expanded < 1e-5):
        print("[WARNING] Some daily_baseline values are extremely close to zero and hit the 1e-5 floor!")
        
    predicted_ratio = y_pred / daily_baseline_safe
    
    # Broadcast daily baseline for Stage 1 prediction (flat line)
    stage1_pred = np.repeat(daily_baseline_expanded, 24, axis=1) # [Batch, 24]
    
    # Ablation Metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    stage1_flat = stage1_pred.flatten()
    
    mae_stage1 = mean_absolute_error(y_true_flat, stage1_flat)
    rmse_stage1 = np.sqrt(mean_squared_error(y_true_flat, stage1_flat))
    r2_stage1 = r2_score(y_true_flat, stage1_flat)
    
    mae_stage2 = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse_stage2 = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2_stage2 = r2_score(y_true_flat, y_pred_flat)
    
    print("\n" + "="*60)
    print("ZERO-SHOT ABLATION METRICS (LUCKNOW & PATNA)")
    print("="*60)
    print("Stage 1 (Daily Baseline Only - Flatline):")
    print(f"  MAE:  {mae_stage1:.4f} µg/m³")
    print(f"  RMSE: {rmse_stage1:.4f} µg/m³")
    print(f"  R²:   {r2_stage1:.4f}")
    print("-" * 60)
    print("Stage 2 (Hybrid Network - Dynamic):")
    print(f"  MAE:  {mae_stage2:.4f} µg/m³")
    print(f"  RMSE: {rmse_stage2:.4f} µg/m³")
    print(f"  R²:   {r2_stage2:.4f}")
    print("="*60)
    
    # Peak Analysis
    max_values = np.max(y_true, axis=1)
    peak_indices = np.where(max_values > 400)[0]
    
    if len(peak_indices) > 0:
        best_idx = peak_indices[np.argmax(max_values[peak_indices])]
        print(f"\n[PEAK ANALYSIS] Found extreme event with PM2.5 > 400 at index {best_idx}.")
    else:
        best_idx = np.argmax(max_values)
        print(f"\n[PEAK ANALYSIS] No peak > 400 found. Falling back to max available peak at index {best_idx}.")
        
    y_true_seq = y_true[best_idx]
    y_pred_seq = y_pred[best_idx]
    stage1_seq = stage1_pred[best_idx]
    ratio_seq = predicted_ratio[best_idx]
    
    print("\n24-Hour Multiplicative Mapping (Predicted Ratio) for the Peak Event:")
    print(np.round(ratio_seq, 3))
    
    # Publication Plot
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "publication_peak_plot.png"
    
    plt.figure(figsize=(12, 6), facecolor='white')
    hours = np.arange(1, 25)
    
    plt.plot(hours, y_true_seq, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Ground Truth')
    plt.plot(hours, stage1_seq, marker='x', linestyle='--', color='gray', linewidth=2.5, markersize=8, label='Stage 1 Baseline (Daily Avg)')
    plt.plot(hours, y_pred_seq, marker='s', linestyle='-', color='#d62728', linewidth=2.5, markersize=8, label='Stage 2 Hybrid Forecast')
    
    plt.title("Ablation Analysis: Stage 1 Baseline vs Stage 2 Hybrid Prediction", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Hour (T+1 to T+24)", fontsize=14, labelpad=10)
    plt.ylabel("Raw $PM_{2.5}$ Concentration ($\mu g/m^3$)", fontsize=14, labelpad=10)
    plt.xticks(hours, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved publication plot to {plot_path}")

if __name__ == "__main__":
    evaluate_ablation()
