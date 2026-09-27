import numpy as np
from src.models.dataloaders import get_dataloaders
from src.models.dataset import AirQualityDataset
import joblib

_, _, test_loader = get_dataloaders(batch_size=256, is_training=False)
train_dataset = AirQualityDataset([
    'Delhi_C14', 'Delhi_C15', 'Kolkata_C20', 'Kolkata_C21', 
    'Hyderabad_C19', 'Hyderabad_C20', 'Ahmedabad_C15', 'Ahmedabad_C16'
])

target_scaler = joblib.load("models/saved/target_scaler.joblib")

y_train_raw = target_scaler.inverse_transform(train_dataset.Y_data.reshape(-1, 1))

print(f"Max PM2.5 in Train set: {y_train_raw.max():.2f}")
print(f"99th percentile in Train set: {np.percentile(y_train_raw, 99):.2f}")

test_dataset = AirQualityDataset(['Lucknow_C1', 'Patna_C20'])
y_test_raw = target_scaler.inverse_transform(test_dataset.Y_data.reshape(-1, 1))
print(f"Max PM2.5 in Test set: {y_test_raw.max():.2f}")
print(f"99th percentile in Test set: {np.percentile(y_test_raw, 99):.2f}")
