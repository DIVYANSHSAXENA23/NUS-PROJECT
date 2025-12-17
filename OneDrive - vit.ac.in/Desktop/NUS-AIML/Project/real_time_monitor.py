import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import serial
import pickle


MODEL_PATH = "rf_model.pkl"

# Heuristic baseline for magnetometer magnitude (can be tuned from training stats)
MAG_BASELINE = 1500.0

# Risk weighting constants (0–100 overall)
SEISMIC_MAX_RISK = 40.0   # ML-based seismic risk (Random Forest probability)
MOVEMENT_MAX_RISK = 20.0  # sudden distance change / collapse
PROXIMITY_MAX_RISK = 30.0 # persistent close obstacle / blocked path
MAG_MAX_RISK = 10.0       # magnetic disturbance


def load_model(model_path: str = MODEL_PATH):
    """Load the trained Random Forest vibration model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def parse_csv_line(line: str):
    """
    Parse one CSV line from the microcontroller.

    Expected format:
        time_ms,vibration,mag_x,mag_y,mag_z,distance_cm
    """
    parts = line.strip().split(",")
    if len(parts) != 6:
        raise ValueError("Unexpected data format")

    time_ms = int(parts[0])
    vibration = int(parts[1])
    mag_x = float(parts[2])
    mag_y = float(parts[3])
    mag_z = float(parts[4])
    distance = float(parts[5])

    return time_ms, vibration, mag_x, mag_y, mag_z, distance


def compute_features(prev_distance, distance, mag_x, mag_y, mag_z):
    """
    Compute engineered features to match the training pipeline:
    - mag_magnitude: sqrt(x^2 + y^2 + z^2)
    - distance_change: difference from previous distance
    """
    mag_magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)

    if prev_distance is None:
        distance_change = 0.0
    else:
        distance_change = distance - prev_distance

    return mag_magnitude, distance_change


def main():
    parser = argparse.ArgumentParser(description="Real-time micro-seismic monitoring")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    args = parser.parse_args()

    model = load_model()

    ser = serial.Serial(args.port, args.baudrate, timeout=2)
    print(f"Connected to {args.port} at {args.baudrate} baud.")

    prev_distance = None

    try:
        while True:
            raw = ser.readline().decode(errors="ignore")
            if not raw.strip():
                continue

            try:
                time_ms, vibration_raw, mag_x, mag_y, mag_z, distance = parse_csv_line(raw)
            except Exception:
                # Skip malformed lines
                continue

            # ---- Feature engineering (same as training) ----
            mag_magnitude, distance_change = compute_features(
                prev_distance, distance, mag_x, mag_y, mag_z
            )
            prev_distance = distance

            # Build feature DataFrame to match training
            # Features: distance, distance_change, mag_magnitude
            X_df = pd.DataFrame(
                [{
                    "distance": distance,
                    "distance_change": distance_change,
                    "mag_magnitude": mag_magnitude,
                }]
            )

            # -------------------------------------------------------
            # 1) SEISMIC COMPONENT (ML-based)
            # -------------------------------------------------------
            # Use Random Forest probability of vibration (0–1)
            prob_1 = float(model.predict_proba(X_df)[0][1])

            # Map probability linearly to a 0–SEISMIC_MAX_RISK band
            seismic_risk = SEISMIC_MAX_RISK * prob_1

            # -------------------------------------------------------
            # 2) SUDDEN MOVEMENT COMPONENT (distance_change)
            # -------------------------------------------------------
            # Large |distance_change| suggests a collapse / sudden shift.
            # Ignore small noise within +/- 1 cm.
            movement_epsilon = 1.0  # cm
            movement_scale = 1.5    # risk points per cm beyond epsilon

            movement_delta = max(0.0, abs(distance_change) - movement_epsilon)
            movement_risk = min(MOVEMENT_MAX_RISK, movement_scale * movement_delta)

            # -------------------------------------------------------
            # 3) PROXIMITY COMPONENT (absolute distance)
            # -------------------------------------------------------
            # Even if movement stops (distance_change -> 0),
            # a persistently small distance indicates a blocked path
            # or fallen object near the sensor.
            # This is INDEPENDENT of movement - it's about current position.
            if distance <= 0 or distance > 200:  # Invalid readings
                proximity_risk = 0.0
            elif distance <= 5.0:
                # Critical: almost touching sensor/structure (fallen object)
                proximity_risk = PROXIMITY_MAX_RISK  # 30 points
            elif distance <= 15.0:
                # High: very close obstacle
                proximity_risk = 20.0  # Fixed 20 points
            elif distance <= 30.0:
                # Moderate: nearby but not immediate
                proximity_risk = 10.0  # Fixed 10 points
            else:
                # Safe: good clearance (>30 cm)
                proximity_risk = 0.0

            # -------------------------------------------------------
            # 4) MAGNETIC DISTURBANCE COMPONENT (mag_magnitude)
            # -------------------------------------------------------
            # Compare current magnitude to a baseline from training.
            # Higher than baseline by a margin contributes up to MAG_MAX_RISK.
            mag_excess = max(0.0, mag_magnitude - MAG_BASELINE)
            if mag_excess > 100.0:
                magnetic_risk = MAG_MAX_RISK
            elif mag_excess > 50.0:
                magnetic_risk = MAG_MAX_RISK / 2.0
            else:
                magnetic_risk = 0.0

            # -------------------------------------------------------
            # COMBINE ALL COMPONENTS INTO 0–100 RISK INDEX
            # -------------------------------------------------------
            total_risk = seismic_risk + movement_risk + proximity_risk + magnetic_risk
            total_risk = int(min(100.0, total_risk))

            # ---- Output for demo / debugging ----
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print("=" * 70)
            print(f"Time: {timestamp_str}")
            print("-" * 70)
            print("Sensor Readings:")
            print(f"  Distance: {distance:.2f} cm")
            print(f"  Magnetometer: X={mag_x:.2f}, Y={mag_y:.2f}, Z={mag_z:.2f}")
            print(f"  Magnitude: {mag_magnitude:.2f}")
            print(f"  Vibration (raw): {vibration_raw}")
            print("-" * 70)
            print("ML Predictions:")
            print(f"  Vibration probability: {prob_1:.2f}")
            print("-" * 70)
            print("Risk Components:")
            print(f"  Seismic (ML)     : {seismic_risk:.1f}/{SEISMIC_MAX_RISK}")
            print(f"  Movement         : {movement_risk:.1f}/{MOVEMENT_MAX_RISK}")
            print(f"  Proximity        : {proximity_risk:.1f}/{PROXIMITY_MAX_RISK}")
            print(f"  Magnetic         : {magnetic_risk:.1f}/{MAG_MAX_RISK}")
            print("-" * 70)
            print(f"TOTAL RISK INDEX   : {total_risk}/100")
            print("=" * 70)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping real-time monitor.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()


