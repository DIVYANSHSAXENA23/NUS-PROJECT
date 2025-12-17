"""
Visualization script for micro-seismic detection results
Generates plots for data analysis and model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load and process data (same as main script)
df = pd.read_csv(
    "micro_seismic_data.csv",
    skiprows=1,
    header=0,
    names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
)
df = df.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})

# Feature Engineering
df["mag_magnitude"] = np.sqrt(df["mag_x"]**2 + df["mag_y"]**2 + df["mag_z"]**2)
df["distance_change"] = df["distance"].diff().fillna(0)
df["vibration_freq"] = df["vibration"].rolling(window=5).mean().fillna(0)

X = df[["distance", "distance_change", "mag_magnitude", "vibration_freq"]]
y = df["vibration"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Anomaly detection
iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(X)
df["anomaly_flag"] = iso_model.predict(X)

# Risk index
def compute_risk(row):
    risk = 0
    if row["vibration"] == 1:
        risk += 40
    if abs(row["distance_change"]) > 5:
        risk += 25
    if row["mag_magnitude"] > df["mag_magnitude"].mean():
        risk += 20
    if row["anomaly_flag"] == -1:
        risk += 30
    return min(risk, 100)

df["risk_index"] = df.apply(compute_risk, axis=1)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Micro-Seismic Detection Analysis', fontsize=16, fontweight='bold')

# 1. Vibration over time
axes[0, 0].plot(df["timestamp"], df["vibration"], alpha=0.6)
axes[0, 0].set_title('Vibration Detection Over Time')
axes[0, 0].set_xlabel('Timestamp (ms)')
axes[0, 0].set_ylabel('Vibration (0/1)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distance measurements
axes[0, 1].plot(df["timestamp"], df["distance"], color='green', alpha=0.6)
axes[0, 1].set_title('Distance Measurements Over Time')
axes[0, 1].set_xlabel('Timestamp (ms)')
axes[0, 1].set_ylabel('Distance (cm)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Magnetometer magnitude
axes[0, 2].plot(df["timestamp"], df["mag_magnitude"], color='red', alpha=0.6)
axes[0, 2].set_title('Magnetometer Magnitude Over Time')
axes[0, 2].set_xlabel('Timestamp (ms)')
axes[0, 2].set_ylabel('Magnitude')
axes[0, 2].grid(True, alpha=0.3)

# 4. Risk index distribution
axes[1, 0].hist(df["risk_index"], bins=20, color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Risk Index Distribution')
axes[1, 0].set_xlabel('Risk Index')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

# 6. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

axes[1, 2].barh(feature_importance['feature'], feature_importance['importance'], color='purple', alpha=0.7)
axes[1, 2].set_title('Feature Importance')
axes[1, 2].set_xlabel('Importance')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'analysis_results.png'")
plt.show()

# Risk index over time
plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["risk_index"], color='red', linewidth=2, alpha=0.7)
plt.fill_between(df["timestamp"], df["risk_index"], alpha=0.3, color='red')
plt.axhline(y=50, color='orange', linestyle='--', label='Medium Risk Threshold')
plt.axhline(y=75, color='red', linestyle='--', label='High Risk Threshold')
plt.title('Disaster Risk Index Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Risk Index (0-100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('risk_index_timeline.png', dpi=300, bbox_inches='tight')
print("Risk index timeline saved as 'risk_index_timeline.png'")
plt.show()

print("\nVisualizations completed!")

