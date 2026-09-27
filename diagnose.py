import torch
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

device = get_device()
model = SpatioTemporalHybridNet(in_channels=15, seq_length=24, pred_length=24).to(device)
model.load_state_dict(torch.load("models/saved/best_hybrid_model.pth", map_location=device))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)

# Let's just look at one batch
for x_batch, y_batch in test_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    with torch.no_grad():
        # Let's break down the forward pass
        daily_baseline = x_batch[:, 0, -1]
        cnn_input = x_batch[:, :, :-1]
        
        x_in = cnn_input.permute(0, 2, 1)
        x_in = model.conv1d(x_in)
        x_in = model.batch_norm(x_in)
        x_in = model.relu(x_in)
        x_in = model.dropout(x_in)
        
        x_in = x_in.permute(0, 2, 1)
        lstm_out, _ = model.bilstm(x_in)
        attended_out = model.attention(lstm_out)
        
        residual = model.regressor(attended_out).squeeze(-1)
        
        final_output = residual + daily_baseline.unsqueeze(-1)
        
        print("--- Residual Diagnostics ---")
        print(f"Max residual (Z-score): {residual.max().item():.4f}")
        print(f"Min residual (Z-score): {residual.min().item():.4f}")
        print(f"Mean residual (Z-score): {residual.mean().item():.4f}")
        print(f"Std residual (Z-score): {residual.std().item():.4f}")
        
        print("\n--- Daily Baseline ---")
        print(f"Mean Baseline (Z-score): {daily_baseline.mean().item():.4f}")
        
        print("\n--- Final Output ---")
        print(f"Std Final Output (Z-score): {final_output.std(dim=1).mean().item():.4f}")
        break
