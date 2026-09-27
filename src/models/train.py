import argparse
import time
from pathlib import Path
import torch
from torch.optim import AdamW

from src.models.architecture import SpatioTemporalHybridNet
from src.models.losses import FrequencyWeightedMAE
from src.models.dataloaders import get_dataloaders

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) backend strictly in FP32.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    else:
        print("Using CPU backend.")
        return torch.device("cpu")

def train(test_mode=False, batch_size=1024):
    device = get_device()
    
    # Initialize components with dual-path architecture
    model = SpatioTemporalHybridNet(n_dynamic=10, n_static=6, seq_length=24, pred_length=24).to(device)
    
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_hybrid_model.pth"
    
    if save_path.exists():
        print(f"Resuming training from existing weights at {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        
    criterion = FrequencyWeightedMAE(beta=0.99).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Lowered LR for fine-tuning
    
    print("Loading DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, is_training=True)
    
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_hybrid_model.pth"
    
    print("\nStarting Training Phase...")
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if test_mode and batch_idx >= 0:
                print(f"[Test Mode] Forward and Backward pass successful! Loss: {loss.item():.4f}")
                break
                
        avg_train_loss = train_loss / (1 if test_mode else len(train_loader))
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                
                if test_mode:
                    break
                    
        avg_val_loss = val_loss / (1 if test_mode else len(val_loader))
        
        # Monitor train/val gap for regularization health
        gap_pct = ((avg_val_loss - avg_train_loss) / avg_val_loss * 100) if avg_val_loss > 0 else 0
        print(f"Epoch [{epoch+1:02d}/{epochs}] ({time.time()-t0:.1f}s) - Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Gap: {gap_pct:.1f}%")
        
        if test_mode:
            print("Test Mode completed successfully.")
            return
            
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true', help='Run a single batch to test forward/backward loop.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    args = parser.parse_args()
    
    train(test_mode=args.test_mode, batch_size=args.batch_size)
