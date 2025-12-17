"""
Demo mode for testing real-time monitor without hardware
Simulates Arduino sensor data for testing purposes
"""

import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

print("=" * 70)
print("DEMO MODE - Testing without hardware")
print("=" * 70)

# Load models
try:
    rf_model = joblib.load("rf_model.pkl")
    iso_model = joblib.load("iso_model.pkl")
    mag_mean = joblib.load("mag_mean.pkl")
    print("[OK] Models loaded successfully\n")
except FileNotFoundError:
    print("[ERROR] Models not found! Run 'python train_and_save_model.py' first.")
    exit(1)

# Load historical data to simulate
df = pd.read_csv(
    "micro_seismic_data.csv",
    skiprows=1,
    header=0,
    names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
)
df = df.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})

# Initialize tracking variables
previous_distance = None
vibration_history = []

def compute_features(data):
    """Compute features from sensor data"""
    global previous_distance, vibration_history
    
    # Magnetometer magnitude
    mag_magnitude = np.sqrt(data['mag_x']**2 + data['mag_y']**2 + data['mag_z']**2)
    
    # Distance change
    if previous_distance is not None:
        distance_change = data['distance'] - previous_distance
    else:
        distance_change = 0
    previous_distance = data['distance']
    
    # Vibration frequency
    vibration_history.append(data['vibration'])
    if len(vibration_history) > 5:
        vibration_history.pop(0)
    vibration_freq = np.mean(vibration_history) if vibration_history else 0
    
    return {
        'distance': data['distance'],
        'distance_change': distance_change,
        'mag_magnitude': mag_magnitude,
        'vibration_freq': vibration_freq
    }

def predict(features):
    """Make predictions"""
    X = pd.DataFrame([[
        features['distance'],
        features['distance_change'],
        features['mag_magnitude'],
        features['vibration_freq']
    ]], columns=['distance', 'distance_change', 'mag_magnitude', 'vibration_freq'])
    
    vibration_pred = rf_model.predict(X)[0]
    anomaly_flag = iso_model.predict(X)[0]
    anomaly_score = iso_model.decision_function(X)[0]
    
    return vibration_pred, anomaly_flag, anomaly_score

def compute_risk_index(vibration, distance_change, mag_magnitude, anomaly_flag):
    """Compute risk index"""
    risk = 0
    if vibration == 1:
        risk += 40
    if abs(distance_change) > 5:
        risk += 25
    if mag_magnitude > mag_mean:
        risk += 20
    if anomaly_flag == -1:
        risk += 30
    return min(risk, 100)

def get_risk_level(risk_index):
    """Get risk level"""
    if risk_index < 30:
        return "LOW", "[GREEN]"
    elif risk_index < 60:
        return "MEDIUM", "[YELLOW]"
    elif risk_index < 80:
        return "HIGH", "[ORANGE]"
    else:
        return "CRITICAL", "[RED]"

def display_result(data, features, vibration_pred, anomaly_flag, risk_index, risk_level, risk_emoji):
    """Display formatted result"""
    print("\n" + "=" * 70)
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MODE] DEMO (Simulated Data)")
    print("-" * 70)
    print(f"[SENSORS] Sensor Readings:")
    print(f"   Distance: {data['distance']:.2f} cm")
    print(f"   Magnetometer: X={data['mag_x']}, Y={data['mag_y']}, Z={data['mag_z']}")
    print(f"   Magnitude: {features['mag_magnitude']:.2f}")
    print(f"   Vibration: {data['vibration']} (Raw)")
    print("-" * 70)
    print(f"[ML] ML Predictions:")
    print(f"   Vibration Detected: {'YES' if vibration_pred == 1 else 'NO'}")
    print(f"   Anomaly: {'YES' if anomaly_flag == -1 else 'NO'}")
    print("-" * 70)
    print(f"{risk_emoji} Risk Index: {risk_index}/100 - {risk_level}")
    print("=" * 70)

# Main demo loop
print("\n[START] Starting demo with simulated sensor data...")
print("Press Ctrl+C to stop\n")

try:
    # Simulate reading from first 20 rows of data
    for idx in range(min(20, len(df))):
        row = df.iloc[idx]
        
        # Simulate sensor data
        data = {
            'timestamp': row['timestamp'],
            'vibration': row['vibration'],
            'mag_x': row['mag_x'],
            'mag_y': row['mag_y'],
            'mag_z': row['mag_z'],
            'distance': row['distance']
        }
        
        # Compute features
        features = compute_features(data)
        
        # Make predictions
        vibration_pred, anomaly_flag, anomaly_score = predict(features)
        
        # Compute risk
        risk_index = compute_risk_index(
            vibration_pred,
            features['distance_change'],
            features['mag_magnitude'],
            anomaly_flag
        )
        
        # Get risk level
        risk_level, risk_emoji = get_risk_level(risk_index)
        
        # Display result
        display_result(data, features, vibration_pred, anomaly_flag, 
                      risk_index, risk_level, risk_emoji)
        
        # Simulate delay (like Arduino sends data every 500ms)
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("[STOP] DEMO STOPPED")
    print("=" * 70)
print("\n[SUCCESS] Demo completed successfully!")
print("If this worked, the real-time monitor should work with hardware too.")

print("\n" + "=" * 70)
print("Demo Summary:")
print("=" * 70)
print("[OK] Models loaded and working")
print("[OK] Feature engineering working")
print("[OK] Predictions working")
print("[OK] Risk index calculation working")
print("[OK] Display formatting working")
print("\n[SUCCESS] Integration is ready for hardware!")

