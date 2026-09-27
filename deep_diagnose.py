"""
Deep diagnostic: trace the EXACT values at every stage of the pipeline
to find where the flatline originates.
"""
import torch
import numpy as np
import joblib
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

device = get_device()

# Load model
model = SpatioTemporalHybridNet(in_channels=15, seq_length=24, pred_length=24).to(device)
model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
model.eval()

# Load scalers
target_scaler = joblib.load("models/saved/target_scaler.joblib")
print(f"Target scaler mean: {target_scaler.mean_[0]:.4f}, std: {target_scaler.scale_[0]:.4f}")

# Load test data
_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)

# Grab first batch
x_batch, y_batch = next(iter(test_loader))
x_batch_dev = x_batch.to(device)

# ============ STAGE 1: RAW INPUT INSPECTION ============
print("\n" + "="*60)
print("STAGE 1: RAW INPUT TENSOR INSPECTION")
print("="*60)
print(f"X shape: {x_batch.shape}")  # [B, 24, 16]
print(f"Y shape: {y_batch.shape}")  # [B, 24]

# Check the 16th feature (daily baseline) — is it constant across 24 hours?
sample_0_f16 = x_batch[0, :, -1].numpy()
print(f"\nSample 0 — Feature 16 (daily baseline) across 24 hours:")
print(f"  Values: {sample_0_f16[:5]}... (first 5)")
print(f"  Std across 24 hours: {sample_0_f16.std():.6f}")
print(f"  Is constant? {sample_0_f16.std() < 1e-5}")

# Check the 15 scaled features — do they actually vary?
sample_0_f15 = x_batch[0, :, :15].numpy()
print(f"\nSample 0 — First 15 features std across 24 hours:")
for i in range(15):
    col_std = sample_0_f15[:, i].std()
    print(f"  Feature {i:2d}: std={col_std:.6f}")

# ============ STAGE 2: FORWARD PASS DECOMPOSITION ============
print("\n" + "="*60)
print("STAGE 2: FORWARD PASS DECOMPOSITION")
print("="*60)

with torch.no_grad():
    # Step A: Slice
    daily_baseline = x_batch_dev[:, 0, -1]
    cnn_input = x_batch_dev[:, :, :-1]
    
    # Step B: CNN
    x_in = cnn_input.permute(0, 2, 1)
    x_in = model.conv1d(x_in)
    x_in = model.batch_norm(x_in)
    x_in = model.relu(x_in)
    x_in = model.dropout(x_in)
    x_in = x_in.permute(0, 2, 1)
    
    # Step C: BiLSTM
    lstm_out, _ = model.bilstm(x_in)
    
    # Step D: Attention
    attended = model.attention(lstm_out)
    
    # Step E: Flatten + Regressor
    flat = attended.reshape(attended.size(0), -1)
    residual = model.regressor(flat)
    
    # Step F: Skip connection
    final = residual + daily_baseline.unsqueeze(-1)
    
    # Print stats at every stage
    print(f"CNN output — mean: {x_in.mean():.6f}, std: {x_in.std():.6f}")
    print(f"BiLSTM output — mean: {lstm_out.mean():.6f}, std: {lstm_out.std():.6f}")
    print(f"Attention output — mean: {attended.mean():.6f}, std: {attended.std():.6f}")
    
    print(f"\n*** RESIDUAL (the learned delta) ***")
    print(f"  Shape: {residual.shape}")
    print(f"  Mean:  {residual.mean():.6f}")
    print(f"  Std:   {residual.std():.6f}")
    print(f"  Min:   {residual.min():.6f}")
    print(f"  Max:   {residual.max():.6f}")
    
    # Check per-hour variation
    residual_np = residual.cpu().numpy()
    per_hour_std = residual_np.std(axis=0)  # std across batch for each hour
    per_sample_std = residual_np.std(axis=1)  # std across hours for each sample
    
    print(f"\n  Per-hour std (across batch):  {per_hour_std}")
    print(f"  Mean per-sample std (across 24 hours): {per_sample_std.mean():.6f}")
    
    print(f"\n*** DAILY BASELINE (skip connection) ***")
    print(f"  Mean:  {daily_baseline.mean():.6f}")
    print(f"  Std:   {daily_baseline.std():.6f}")
    
    print(f"\n*** FINAL OUTPUT (residual + baseline) ***")
    print(f"  Mean:  {final.mean():.6f}")
    print(f"  Std:   {final.std():.6f}")
    
    # ============ STAGE 3: THE CRITICAL TEST ============
    # Check the SPECIFIC sample used in the plot (index 4877)
    
# Now let's look at what the evaluate script actually plots
# It picks the sample with the highest peak
print("\n" + "="*60)
print("STAGE 3: CHECKING THE PLOTTED SAMPLE")
print("="*60)

all_y_true = []
all_y_pred = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yp = model(xb)
        all_y_true.append(yb.cpu().numpy())
        all_y_pred.append(yp.cpu().numpy())

y_true_arr = np.concatenate(all_y_true, axis=0)
y_pred_arr = np.concatenate(all_y_pred, axis=0)

# Inverse transform
orig_shape = y_true_arr.shape
y_true_raw = target_scaler.inverse_transform(y_true_arr.reshape(-1, 1)).reshape(orig_shape)
y_pred_raw = target_scaler.inverse_transform(y_pred_arr.reshape(-1, 1)).reshape(orig_shape)
y_pred_raw = np.clip(y_pred_raw, a_min=1.0, a_max=None)

# Find the peak sample (same logic as evaluate.py)
max_values = np.max(y_true_raw, axis=1)
peak_indices = np.where(max_values > 200)[0]
best_idx = peak_indices[np.argmax(max_values[peak_indices])]

y_true_seq = y_true_raw[best_idx]
y_pred_seq = y_pred_raw[best_idx]

print(f"Peak sample index: {best_idx}")
print(f"Ground truth (24 hours): {np.round(y_true_seq, 1)}")
print(f"Prediction   (24 hours): {np.round(y_pred_seq, 1)}")
print(f"Pred std across 24 hours: {y_pred_seq.std():.4f}")
print(f"Pred range: {y_pred_seq.min():.1f} to {y_pred_seq.max():.1f}")

# Check the Z-score predictions BEFORE inverse transform
y_pred_z = y_pred_arr[best_idx]
print(f"\nZ-score predictions (before inverse): {np.round(y_pred_z, 4)}")
print(f"Z-score pred std: {y_pred_z.std():.6f}")
print(f"Z-score pred range: {y_pred_z.min():.4f} to {y_pred_z.max():.4f}")

# What is the daily baseline for this sample?
# We need to trace it through the dataloader
xb_all = []
for xb, _ in test_loader:
    xb_all.append(xb.numpy())
x_all = np.concatenate(xb_all, axis=0)
daily_baseline_z = x_all[best_idx, 0, -1]
daily_baseline_raw = target_scaler.inverse_transform([[daily_baseline_z]])[0][0]
print(f"\nDaily baseline Z-score: {daily_baseline_z:.4f}")
print(f"Daily baseline raw: {daily_baseline_raw:.1f} µg/m³")

# The residual for this sample
residual_z = y_pred_z - daily_baseline_z
print(f"\nResidual Z-scores: {np.round(residual_z, 4)}")
print(f"Residual range: {residual_z.min():.4f} to {residual_z.max():.4f}")
print(f"Residual std: {residual_z.std():.6f}")

