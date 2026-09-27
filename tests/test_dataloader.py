import torch
from src.models.dataloaders import get_dataloaders
from src.models.losses import FrequencyWeightedMAE
import time

def main():
    print("--- Starting Dataloader Initialization ---")
    t0 = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    print(f"Dataloaders initialized in {time.time()-t0:.2f} seconds.")
    
    # 1. Shape Check
    print("\n--- 1. Tensor Shape Check ---")
    X_batch, Y_batch = next(iter(train_loader))
    print(f"X_batch shape: {X_batch.shape} (Expected: [32, 24, 16])")
    print(f"Y_batch shape: {Y_batch.shape} (Expected: [32, 24])")
    
    # 2. Constraint Check
    print("\n--- 2. Daily Stage 1 Constraint Validation ---")
    # Feature index 15 is stage1_daily_pm25
    daily_pm25_feature = X_batch[0, :, 15]
    print(f"Sample 0 stage1_daily_pm25 over 24h:\n{daily_pm25_feature.numpy()}")
    is_constant = torch.all(daily_pm25_feature == daily_pm25_feature[0])
    print(f"Is constant over 24h? {is_constant.item()} (Expected: True)")
    
    # 3. Loss Function Check
    print("\n--- 3. Frequency-Weighted MAE Behavior ---")
    loss_fn = FrequencyWeightedMAE(beta=0.99)
    
    # Inject an extreme value and a normal value into Y_batch to observe weighting
    y_true = Y_batch.clone()
    y_true[0, 0] = 400.0 # Extreme (Rare bin)
    y_true[0, 1] = 30.0  # Normal (Common bin)
    
    weights = loss_fn.get_weights(y_true)
    
    print(f"Weight for Extreme PM2.5 (400.0): {weights[0, 0].item():.4f}")
    print(f"Weight for Normal PM2.5 (30.0) : {weights[0, 1].item():.4f}")
    
    if weights[0, 0].item() > weights[0, 1].item():
        print("-> SUCCESS: Peak-smoothing failure prevented. Extreme values are penalized heavier!")
    else:
        print("-> FAILED: Extreme value did not receive a higher weight.")

if __name__ == "__main__":
    main()
