import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.models.architecture import SpatioTemporalHybridNet
from src.models.dataloaders import get_dataloaders
from src.models.train import get_device

def test():
    device = get_device()
    
    # Initialize model WITHOUT dropout (we'll hack the eval mode to disable it during training for a quick test)
    model = SpatioTemporalHybridNet(n_dynamic=9, n_static=6, seq_length=24, pred_length=24).to(device)
    # Standard MSE Loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Loading Data...")
    # Just train on ONE chunk (Delhi_C14) to see if it can fit it at all
    train_loader, _, _ = get_dataloaders(batch_size=256, is_training=True)
    
    print("Training with Standard MSE to check capacity...")
    for epoch in range(10): # Quick 10 epochs
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.2f}")
        
    model.eval()
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        y_pred = model(x_batch)
        
    y_true_np = y_batch.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Pick a sample
    idx = 10
    print(f"\nSample {idx} Ground Truth: {np.round(y_true_np[idx], 1)}")
    print(f"Sample {idx} Prediction:   {np.round(y_pred_np[idx], 1)}")
    print(f"Prediction std: {y_pred_np[idx].std():.2f}")
    
    # Check if ratio is fluctuating
    ratio = y_pred_np[idx] / x_batch[idx, 0, -1].item()
    print(f"Predicted Ratio: {np.round(ratio, 2)}")

if __name__ == "__main__":
    test()
