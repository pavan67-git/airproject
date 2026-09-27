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

all_ratios = []
with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        
        dynamic = x_batch[:, :, :9]
        static = x_batch[:, 0, 9:15]
        
        x_in = dynamic.permute(0, 2, 1)
        x_in = model.conv1d(x_in)
        x_in = model.batch_norm(x_in)
        x_in = model.relu(x_in)
        x_in = model.dropout_cnn(x_in)
        x_in = x_in.permute(0, 2, 1)
        
        lstm_out, _ = model.bilstm(x_in)
        attended = model.attention(lstm_out)
        
        flat_temporal = attended.reshape(attended.size(0), -1)
        combined = torch.cat([flat_temporal, static], dim=1)
        
        ratio = torch.nn.functional.softplus(model.regressor(combined))
        all_ratios.append(ratio.cpu().numpy())

ratio_arr = np.concatenate(all_ratios)

print("="*60)
print("RATIO DIAGNOSTIC")
print("="*60)
print(f"Mean predicted ratio: {ratio_arr.mean():.4f}")
print(f"Std predicted ratio:  {ratio_arr.std():.4f}")
print(f"Min predicted ratio:  {ratio_arr.min():.4f}")
print(f"Max predicted ratio:  {ratio_arr.max():.4f}")
print(f"Per-sample ratio std (avg): {ratio_arr.std(axis=1).mean():.4f}")

idx = 4877
print(f"\nPeak Sample ({idx}) Ratios: {np.round(ratio_arr[idx], 3)}")
