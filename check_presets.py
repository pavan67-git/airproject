import torch
import numpy as np
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

def check_dashboard_presets():
    device = get_device()
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
    model.eval()
    
    _, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
    
    all_x, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_x.append(xb.numpy())
            all_y.append(yb.numpy())
            
    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    max_values = np.max(y_all, axis=1)
    
    # Normal Day
    normal_indices = np.where((max_values > 60) & (max_values < 120))[0]
    preset_normal = normal_indices[50] if len(normal_indices) > 50 else (normal_indices[0] if len(normal_indices) > 0 else 0)
    
    # Typical Severe
    typical_indices = np.where((max_values > 150) & (max_values < 250))[0]
    preset_typical = typical_indices[np.argmax(y_all[typical_indices].std(axis=1))] if len(typical_indices) > 0 else 0
    
    for name, idx in [("Normal Day", preset_normal), ("Typical Severe", preset_typical)]:
        x_tensor = torch.tensor(x_all[idx:idx+1], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x_tensor).squeeze(0).cpu().numpy()
        true = y_all[idx]
        
        baseline = x_all[idx, 0, -1]  # 16th feature is daily baseline
        
        print(f"\n{'='*50}\n{name} (Index {idx})\n{'='*50}")
        print(f"Daily Baseline: {baseline:.1f}")
        print(f"Ground Truth: {true.min():.1f} -> {true.max():.1f}")
        print(f"Prediction:   {pred.min():.1f} -> {pred.max():.1f}")
        print(f"MAE: {np.mean(np.abs(true - pred)):.1f}")

if __name__ == "__main__":
    check_dashboard_presets()
