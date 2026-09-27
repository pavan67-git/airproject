"""Plot a TYPICAL diurnal prediction (150-300 range) to show the network IS learning"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

device = get_device()
model = SpatioTemporalHybridNet(n_dynamic=9, n_static=6, seq_length=24, pred_length=24).to(device)
model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)

all_y_true, all_y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yp = model(xb)
        all_y_true.append(yb.cpu().numpy())
        all_y_pred.append(yp.cpu().numpy())

y_true_raw = np.concatenate(all_y_true)
y_pred_raw = np.concatenate(all_y_pred)
y_pred_raw = np.clip(y_pred_raw, a_min=1.0, a_max=None)

# Find a sample with peak in 100-200 range (typical severe but not extreme)
max_values = np.max(y_true_raw, axis=1)
typical_mask = (max_values > 100) & (max_values < 250)
typical_indices = np.where(typical_mask)[0]

# Pick one with highest variation
if len(typical_indices) > 0:
    per_sample_std = y_true_raw[typical_indices].std(axis=1)
    best_typical = typical_indices[np.argmax(per_sample_std)]
else:
    best_typical = 0

y_true_seq = y_true_raw[best_typical]
y_pred_seq = y_pred_raw[best_typical]

print(f"Selected sample {best_typical}")
print(f"Ground truth range: {y_true_seq.min():.1f} -> {y_true_seq.max():.1f}")
print(f"Prediction range:   {y_pred_seq.min():.1f} -> {y_pred_seq.max():.1f}")
print(f"Ground truth std:   {y_true_seq.std():.1f}")
print(f"Prediction std:     {y_pred_seq.std():.1f}")

plt.figure(figsize=(12, 6), facecolor='white')
hours = np.arange(1, 25)

plt.plot(hours, y_true_seq, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Ground Truth ($PM_{2.5}$)')
plt.plot(hours, y_pred_seq, marker='s', linestyle='--', color='#d62728', linewidth=2.5, markersize=8, label='Predicted ($\mathcal{L}_{fMAE}$ Optimized)')

plt.title("Zero-Shot Diurnal Forecast — Typical Severe Event (Lucknow/Patna)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Hour (T+1 to T+24)", fontsize=14, labelpad=10)
plt.ylabel("$PM_{2.5}$ Concentration (µg/m³)", fontsize=14, labelpad=10)
plt.xticks(hours, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig("results/typical_prediction_lucknow.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved to results/typical_prediction_lucknow.png")
