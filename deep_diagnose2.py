import torch
import numpy as np
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

device = get_device()
model = SpatioTemporalHybridNet(n_dynamic=9, n_static=6, seq_length=24, pred_length=24).to(device)
model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
x_batch, y_batch = next(iter(test_loader))

# Check if the 9 dynamic features actually have variation
print("="*60)
print("DYNAMIC FEATURE VARIANCE (per feature, across 24h)")
print("="*60)
feat_names = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'precipitation',
              'U_wind', 'V_wind', 'inv_PBLH', 'hour_sin', 'hour_cos']
for i in range(9):
    # Std across 24 hours, averaged over batch
    per_sample_std = x_batch[:, :, i].numpy().std(axis=1).mean()
    print(f"  {feat_names[i]:25s} — avg 24h std: {per_sample_std:.4f}")

# Check what the target Y looks like
print(f"\nTarget Y (Z-scored) — sample 0: {np.round(y_batch[0].numpy(), 3)}")
print(f"Target Y std across 24h (sample 0): {y_batch[0].numpy().std():.4f}")
print(f"Target Y std across 24h (batch avg): {y_batch.numpy().std(axis=1).mean():.4f}")

# THE KEY QUESTION: What is the OPTIMAL residual the network should predict?
# If daily_baseline_z = 1.56 and target_z varies from -0.37 to 17.6,
# then optimal residual = target_z - daily_baseline_z
print("\n" + "="*60)
print("OPTIMAL RESIDUAL ANALYSIS (Peak Sample 4877)")
print("="*60)

# Collect all data
all_x, all_y = [], []
for xb, yb in test_loader:
    all_x.append(xb.numpy())
    all_y.append(yb.numpy())
x_all = np.concatenate(all_x)
y_all = np.concatenate(all_y)

idx = 4877
daily_z = x_all[idx, 0, -1]
target_z = y_all[idx]
optimal_residual = target_z - daily_z

print(f"Daily baseline (Z): {daily_z:.4f}")
print(f"Target Y (Z):       {np.round(target_z, 3)}")
print(f"Optimal residual:   {np.round(optimal_residual, 3)}")
print(f"Optimal residual range: {optimal_residual.min():.3f} → {optimal_residual.max():.3f}")
print(f"Optimal residual std:   {optimal_residual.std():.3f}")

# What about a TYPICAL sample?
print("\n" + "="*60)
print("OPTIMAL RESIDUAL ANALYSIS (Typical Sample 0)")
print("="*60)
daily_z_0 = x_all[0, 0, -1]
target_z_0 = y_all[0]
optimal_res_0 = target_z_0 - daily_z_0
print(f"Daily baseline (Z): {daily_z_0:.4f}")
print(f"Target Y (Z):       {np.round(target_z_0, 3)}")
print(f"Optimal residual:   {np.round(optimal_res_0, 3)}")
print(f"Optimal residual range: {optimal_res_0.min():.3f} → {optimal_res_0.max():.3f}")
print(f"Optimal residual std:   {optimal_res_0.std():.3f}")

# What's the average residual std across ALL test samples?
all_daily_z = x_all[:, 0, -1]  # [N]
all_optimal_res = y_all - all_daily_z[:, None]  # [N, 24]
print("\n" + "="*60)
print("GLOBAL RESIDUAL STATISTICS")
print("="*60)
print(f"Mean optimal residual: {all_optimal_res.mean():.4f}")
print(f"Std optimal residual:  {all_optimal_res.std():.4f}")
print(f"Per-sample std (avg):  {all_optimal_res.std(axis=1).mean():.4f}")
print(f"Median per-sample std: {np.median(all_optimal_res.std(axis=1)):.4f}")
print(f"90th %ile per-sample std: {np.percentile(all_optimal_res.std(axis=1), 90):.4f}")

