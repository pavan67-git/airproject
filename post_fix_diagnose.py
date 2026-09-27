import torch
import torch.nn.functional as F
import numpy as np
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

device = get_device()
model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)

x_batch, y_batch = next(iter(test_loader))
x_batch_dev = x_batch.to(device)

with torch.no_grad():
    dynamic = x_batch_dev[:, :, :9]
    static = x_batch_dev[:, 0, 9:15]
    daily_baseline = x_batch_dev[:, 0, -1]
    
    x_in = dynamic.permute(0, 2, 1)
    x_in = model.conv1d(x_in)
    x_in = model.batch_norm(x_in)
    x_in = model.relu(x_in)
    x_in = model.dropout_cnn(x_in)
    x_in = x_in.permute(0, 2, 1)
    
    lstm_out, _ = model.bilstm(x_in)
    attended = model.attention(lstm_out)
    
    # ===== LATE FUSION (Time-Distributed) =====
    static_expanded = static.unsqueeze(1).expand(-1, 24, -1)
    combined = torch.cat([attended, static_expanded], dim=-1)
    ratio = F.softplus(model.regressor(combined)).squeeze(-1)
    final = ratio * daily_baseline.unsqueeze(-1)

    print("="*60)
    print("POST-FIX SIGNAL DIAGNOSTIC")
    print("="*60)
    print(f"CNN output       — std: {x_in.std():.6f}")
    print(f"BiLSTM output    — std: {lstm_out.std():.6f}")
    print(f"Attention output — std: {attended.std():.6f}")
    print(f"Signal preservation: {attended.std()/lstm_out.std()*100:.1f}%")
    print(f"\nRatio — mean: {ratio.mean():.4f}, std: {ratio.std():.4f}")
    print(f"Ratio — min:  {ratio.min():.4f}, max: {ratio.max():.4f}")
    print(f"Per-sample std (across 24 hours): {ratio.cpu().numpy().std(axis=1).mean():.4f}")

# Full evaluation on peak sample
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

max_values = np.max(y_true_raw, axis=1)
peak_indices = np.where(max_values > 200)[0]
best_idx = peak_indices[np.argmax(max_values[peak_indices])]

print(f"\n{'='*60}")
print(f"PEAK SAMPLE (index {best_idx})")
print(f"{'='*60}")
print(f"Ground truth: {np.round(y_true_raw[best_idx], 1)}")
print(f"Prediction:   {np.round(y_pred_raw[best_idx], 1)}")
print(f"Pred range:   {y_pred_raw[best_idx].min():.1f} -> {y_pred_raw[best_idx].max():.1f}")
print(f"Pred std:     {y_pred_raw[best_idx].std():.1f}")

# Physical residuals
daily_val = next(iter(test_loader))[0][best_idx % 256, 0, -1].item()
residual_val = y_pred_raw[best_idx] - daily_val
print(f"\nResidual values: {np.round(residual_val, 3)}")
print(f"Residual range:  {residual_val.min():.3f} -> {residual_val.max():.3f}")
