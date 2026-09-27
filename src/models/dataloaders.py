import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.models.dataset import AirQualityDataset

# Indices of raw physical features to scale (arbitrary ranges)
# After reordering: temperature, humidity, pressure, precip, U_wind, V_wind, inv_PBLH, Upwind_FRP, elevation, pop_density
SCALE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]

# Indices NOT scaled (bounded [-1,1] by construction): hour_sin, hour_cos, ndvi, ndbi, month_sin, month_cos
# These remain at indices [7, 8, 11, 12, 13, 14] — untouched by the scaler

def get_dataloaders(batch_size=32, is_training=True):
    train_chunks = [
        'Delhi_C14', 'Delhi_C15', 'Kolkata_C20', 'Kolkata_C21', 
        'Hyderabad_C19', 'Hyderabad_C20', 'Ahmedabad_C15', 'Ahmedabad_C16'
    ]
    val_chunks = ['Maninagar Ahmedabad Gpcb_C18'] 
    test_chunks = ['Lucknow_C1', 'Patna_C20'] 
    
    print(f"Initializing Train Loader (Chunks: {len(train_chunks)})")
    train_dataset = AirQualityDataset(train_chunks)
    print(f"Initializing Validation Loader (Chunks: {len(val_chunks)})")
    val_dataset = AirQualityDataset(val_chunks)
    print(f"Initializing Test Loader (Chunks: {len(test_chunks)})")
    test_dataset = AirQualityDataset(test_chunks)
    
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = save_dir / "scaler.joblib"
    
    if is_training:
        # Fit Input Scaler on ONLY the raw physical features (selective scaling)
        scaler = StandardScaler()
        X_train_subset = train_dataset.X_data[:, :, SCALE_INDICES].reshape(-1, len(SCALE_INDICES))
        scaler.fit(X_train_subset)
        joblib.dump(scaler, scaler_path)
        print(f"Fitted Feature Scaler and saved to {save_dir}")
    else:
        scaler = joblib.load(scaler_path)
        print(f"Loaded Feature Scaler from {save_dir}")
            
    # Apply Feature Scaler ONLY to raw physical features (in-place at exact column positions)
    def apply_scaling(dataset):
        if len(dataset.X_data) > 0:
            orig_shape = dataset.X_data[:, :, SCALE_INDICES].shape
            X_flat = dataset.X_data[:, :, SCALE_INDICES].reshape(-1, len(SCALE_INDICES))
            X_scaled = scaler.transform(X_flat).reshape(orig_shape)
            dataset.X_data[:, :, SCALE_INDICES] = X_scaled
            
    apply_scaling(train_dataset)
    apply_scaling(val_dataset)
    apply_scaling(test_dataset)
    # Target Y and the 16th feature (daily baseline) remain unscaled in raw physical units
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
