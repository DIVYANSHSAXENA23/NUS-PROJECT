"""
Train the ML model and save it for real-time use
Run this once to train and save the model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

print("=" * 50)
print("TRAINING ML MODEL FOR REAL-TIME USE")
print("=" * 50)

# Load and prepare data
df = pd.read_csv(
    "micro_seismic_data.csv",
    skiprows=1,
    header=0,
    names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
)
df = df.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})

print(f"\n✓ Loaded {len(df)} data points")

# Feature Engineering
df["mag_magnitude"] = np.sqrt(df["mag_x"]**2 + df["mag_y"]**2 + df["mag_z"]**2)
df["distance_change"] = df["distance"].diff().fillna(0)
df["vibration_freq"] = df["vibration"].rolling(window=5).mean().fillna(0)

# Prepare features
X = df[["distance", "distance_change", "mag_magnitude", "vibration_freq"]]
y = df["vibration"]

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✓ Random Forest model trained")

# Train Isolation Forest for anomaly detection
iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(X)
print("✓ Isolation Forest model trained")

# Calculate mean magnetometer magnitude for risk calculation
mag_mean = df["mag_magnitude"].mean()

# Save models and metadata
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(iso_model, "iso_model.pkl")
joblib.dump(mag_mean, "mag_mean.pkl")

print("\n✓ Models saved:")
print("  - rf_model.pkl (Random Forest)")
print("  - iso_model.pkl (Isolation Forest)")
print("  - mag_mean.pkl (Magnetometer mean)")

# Test accuracy
from sklearn.metrics import accuracy_score
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETE!")
print("You can now use real_time_monitor.py for live predictions")
print("=" * 50)

