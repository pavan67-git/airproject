import joblib
target_scaler = joblib.load("models/saved/target_scaler.joblib")
print(f"Target Scaler Mean: {target_scaler.mean_[0]:.2f}")
print(f"Target Scaler Variance: {target_scaler.var_[0]:.2f}")
print(f"Target Scaler Std: {target_scaler.scale_[0]:.2f}")
