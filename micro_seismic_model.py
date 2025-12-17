# ==============================
# MICRO-SEISMIC + INSTABILITY ML
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------- 1. LOAD DATA ----------
# The raw CSV appears to have a metadata row first and then the real header row,
# so we skip the first line and set proper column names.
df = pd.read_csv(
    "micro_seismic_data.csv",
    skiprows=1,
    header=0,
    names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
)

# Standardise column names used later in the script
df = df.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})

print("\nDataset Loaded Successfully")
print(df.head())


# ---------- 2. FEATURE ENGINEERING ----------

# Magnetometer magnitude (orientation instability)
df["mag_magnitude"] = np.sqrt(
    df["mag_x"]**2 + df["mag_y"]**2 + df["mag_z"]**2
)

# Distance change (structural movement)
df["distance_change"] = df["distance"].diff().fillna(0)

# Rolling vibration frequency (micro seismic activity)
df["vibration_freq"] = df["vibration"].rolling(window=5).mean().fillna(0)


# ---------- 3. FEATURE SET & LABEL ----------
X = df[
    ["distance", "distance_change", "mag_magnitude", "vibration_freq"]
]

y = df["vibration"]   # Binary label (0 = no vibration, 1 = vibration)


# ---------- 4. TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------- 5. RANDOM FOREST MODEL ----------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# ---------- 6. MODEL EVALUATION ----------
y_pred = rf_model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------- 7. ANOMALY DETECTION ----------
iso_model = IsolationForest(
    contamination=0.1,
    random_state=42
)

iso_model.fit(X)

df["anomaly_score"] = iso_model.decision_function(X)
df["anomaly_flag"] = iso_model.predict(X)   # -1 = anomaly


# ---------- 8. DISASTER RISK INDEX ----------
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


# ---------- 9. SHOW FINAL OUTPUT ----------
print("\nFINAL OUTPUT WITH RISK INDEX")
print(df[[
    "timestamp",
    "vibration",
    "distance",
    "mag_magnitude",
    "anomaly_flag",
    "risk_index"
]].head())


# ---------- 10. LIVE PREDICTION FUNCTION ----------
def predict_vibration(distance, mag_x, mag_y, mag_z, vibration_freq):
    mag_mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)

    sample = pd.DataFrame(
        [[distance, 0, mag_mag, vibration_freq]],
        columns=[
            "distance",
            "distance_change",
            "mag_magnitude",
            "vibration_freq"
        ]
    )

    return rf_model.predict(sample)[0]


print("\nScript Finished Successfully")
