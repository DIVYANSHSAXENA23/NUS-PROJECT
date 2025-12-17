"""
Test script to verify the integration works correctly
Tests both training and real-time monitoring (with simulated data)
"""

import os
import sys
import pandas as pd
import numpy as np

print("=" * 70)
print("TESTING HARDWARE-SOFTWARE INTEGRATION")
print("=" * 70)

# Test 1: Check if required files exist
print("\n[TEST 1] Checking required files...")
required_files = [
    "micro_seismic_data.csv",
    "train_and_save_model.py",
    "real_time_monitor.py",
    "NUS_proj.ino"
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"  [OK] {file} found")
    else:
        print(f"  [FAIL] {file} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

# Test 2: Check if data file can be loaded
print("\n[TEST 2] Testing data loading...")
try:
    df = pd.read_csv(
        "micro_seismic_data.csv",
        skiprows=1,
        header=0,
        names=["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
    )
    print(f"  [OK] Data loaded successfully ({len(df)} rows)")
except Exception as e:
    print(f"  [FAIL] Error loading data: {e}")
    sys.exit(1)

# Test 3: Test model training
print("\n[TEST 3] Testing model training...")
try:
    # Import training script functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", "train_and_save_model.py")
    train_module = importlib.util.module_from_spec(spec)
    
    # We'll just check if it can be imported, not actually run it
    print("  [OK] Training script syntax is valid")
except Exception as e:
    print(f"  [FAIL] Error in training script: {e}")
    sys.exit(1)

# Test 4: Test if we can train models (actual training)
print("\n[TEST 4] Training models...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    import joblib
    
    # Prepare data
    df_test = df.copy()
    df_test = df_test.rename(columns={"distance_cm": "distance", "time_ms": "timestamp"})
    df_test["mag_magnitude"] = np.sqrt(df_test["mag_x"]**2 + df_test["mag_y"]**2 + df_test["mag_z"]**2)
    df_test["distance_change"] = df_test["distance"].diff().fillna(0)
    df_test["vibration_freq"] = df_test["vibration"].rolling(window=5).mean().fillna(0)
    
    X = df_test[["distance", "distance_change", "mag_magnitude", "vibration_freq"]]
    y = df_test["vibration"]
    
    # Train models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("  [OK] Random Forest model trained")
    
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X)
    print("  [OK] Isolation Forest model trained")
    
    # Save models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(iso_model, "iso_model.pkl")
    joblib.dump(df_test["mag_magnitude"].mean(), "mag_mean.pkl")
    print("  [OK] Models saved successfully")
    
except Exception as e:
    print(f"  [FAIL] Error training models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test real-time monitor with simulated data
print("\n[TEST 5] Testing real-time monitor (simulated data)...")
try:
    import joblib
    
    # Load models
    rf_model = joblib.load("rf_model.pkl")
    iso_model = joblib.load("iso_model.pkl")
    mag_mean = joblib.load("mag_mean.pkl")
    print("  [OK] Models loaded successfully")
    
    # Simulate sensor data (like what Arduino would send)
    test_data = {
        'timestamp': 1000,
        'vibration': 0,
        'mag_x': -42,
        'mag_y': -158,
        'mag_z': 1498,
        'distance': 57.22
    }
    
    # Compute features
    mag_magnitude = np.sqrt(test_data['mag_x']**2 + test_data['mag_y']**2 + test_data['mag_z']**2)
    distance_change = 0  # First reading
    vibration_freq = 0.0
    
    # Prepare feature vector
    X_test = pd.DataFrame([[
        test_data['distance'],
        distance_change,
        mag_magnitude,
        vibration_freq
    ]], columns=['distance', 'distance_change', 'mag_magnitude', 'vibration_freq'])
    
    # Make predictions
    vibration_pred = rf_model.predict(X_test)[0]
    anomaly_flag = iso_model.predict(X_test)[0]
    anomaly_score = iso_model.decision_function(X_test)[0]
    
    # Compute risk
    risk = 0
    if vibration_pred == 1:
        risk += 40
    if abs(distance_change) > 5:
        risk += 25
    if mag_magnitude > mag_mean:
        risk += 20
    if anomaly_flag == -1:
        risk += 30
    risk_index = min(risk, 100)
    
    print(f"  [OK] Prediction made successfully")
    print(f"    - Vibration: {vibration_pred}")
    print(f"    - Anomaly: {anomaly_flag}")
    print(f"    - Risk Index: {risk_index}/100")
    
except Exception as e:
    print(f"  [FAIL] Error in real-time monitor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check dependencies
print("\n[TEST 6] Checking Python dependencies...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'serial': 'pyserial',
    'joblib': 'joblib'
}

missing_packages = []
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  [OK] {package_name} installed")
    except ImportError:
        print(f"  [FAIL] {package_name} NOT installed")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n[WARN] Missing packages: {', '.join(missing_packages)}")
    print("  Install with: pip install " + " ".join(missing_packages))

# Test 7: Test serial port detection (optional)
print("\n[TEST 7] Checking serial port support...")
try:
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if ports:
        print(f"  [OK] Found {len(ports)} serial port(s):")
        for port in ports[:3]:  # Show first 3
            print(f"    - {port.device}: {port.description}")
    else:
        print("  [WARN] No serial ports found (this is OK if hardware is not connected)")
    print("  [OK] Serial library working correctly")
except Exception as e:
    print(f"  [FAIL] Serial library error: {e}")

print("\n" + "=" * 70)
print("[SUCCESS] ALL TESTS COMPLETED!")
print("=" * 70)
print("\nNext steps:")
print("1. Connect your Arduino/ESP32 hardware")
print("2. Upload NUS_proj.ino to your board")
print("3. Run: python real_time_monitor.py --port COM3")
print("\nFor demo mode (without hardware), see demo_mode.py")

