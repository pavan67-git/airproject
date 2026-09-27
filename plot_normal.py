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

# Find a normal sample (max < 150)
max_values = np.max(y_true_raw, axis=1)
normal_mask = (max_values > 60) & (max_values < 120)
normal_indices = np.where(normal_mask)[0]

best_normal = normal_indices[50] # Just pick one

y_true_seq = y_true_raw[best_normal]
y_pred_seq = y_pred_raw[best_normal]

plt.figure(figsize=(12, 6), facecolor='white')
hours = np.arange(1, 25)

plt.plot(hours, y_true_seq, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Ground Truth ($PM_{2.5}$)')
plt.plot(hours, y_pred_seq, marker='s', linestyle='--', color='#d62728', linewidth=2.5, markersize=8, label='Predicted (Ratio x Baseline)')

plt.title("Normal Day Forecast (Zoomed In to show fluctuation)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("$PM_{2.5}$ Concentration (µg/m³)", fontsize=14)
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/zoomed_normal.png", dpi=300, bbox_inches='tight')
print("Saved to results/zoomed_normal.png")
