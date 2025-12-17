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
VIBRATION_SENSOR_RISK = 40.0  # Hardware vibration sensor override (highest priority)
ML_MAX_RISK = 30.0            # ML-based seismic risk (Random Forest probability)
DISTANCE_PROXIMITY_RISK_CRITICAL = 30.0  # Distance < 10 cm (object fall/collapse)
DISTANCE_PROXIMITY_RISK_HIGH = 15.0      # Distance < 25 cm (close obstacle)
MOVEMENT_MAX_RISK = 20.0      # Sudden distance change (>5 cm)
MAG_MAX_RISK = 10.0           # Magnetic disturbance


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

            # ===============================================================
            # SENSOR-FUSION BASED HAZARD SCORING
            # ===============================================================
            # Priority order: Vibration sensor > ML > Distance > Movement > Magnetic
            
            # -------------------------------------------------------
            # 1) VIBRATION SENSOR OVERRIDE (HIGHEST PRIORITY)
            # -------------------------------------------------------
            # Hardware vibration sensor is ground truth - if it triggers,
            # immediately add risk regardless of ML prediction.
            vibration_sensor_risk = 0.0
            if vibration_raw == 1:
                vibration_sensor_risk = VIBRATION_SENSOR_RISK  # 40 points

            # -------------------------------------------------------
            # 2) ML CONTRIBUTION (Probabilistic, not binary)
            # -------------------------------------------------------
            # Use Random Forest probability of vibration (0–1)
            # This is INDEPENDENT of vibration_raw - ML learns from features only
            prob_1 = float(model.predict_proba(X_df)[0][1])
            
            # Map probability linearly to 0–30 risk points
            ml_risk = int(30 * prob_1)

            # -------------------------------------------------------
            # 3) DISTANCE-BASED HAZARD (Object fall / collapse proximity)
            # -------------------------------------------------------
            # Persistent close distance indicates fallen object or collapse,
            # independent of movement. This ensures risk stays high even
            # when object becomes stationary.
            distance_proximity_risk = 0.0
            if distance > 0 and distance <= 200:  # Valid readings only
                if distance < 10.0:
                    # Critical: object very close (likely fallen/collapsed)
                    distance_proximity_risk = DISTANCE_PROXIMITY_RISK_CRITICAL  # 30 points
                elif distance < 25.0:
                    # High: close obstacle
                    distance_proximity_risk = DISTANCE_PROXIMITY_RISK_HIGH  # 15 points

            # -------------------------------------------------------
            # 4) DISTANCE CHANGE (Structural movement)
            # -------------------------------------------------------
            # Large sudden distance changes indicate structural movement/collapse
            movement_risk = 0.0
            if abs(distance_change) > 5.0:  # Threshold: 5 cm
                movement_risk = MOVEMENT_MAX_RISK  # 20 points

            # -------------------------------------------------------
            # 5) MAGNETOMETER INSTABILITY
            # -------------------------------------------------------
            # High magnetometer magnitude indicates magnetic disturbance
            # or structural instability affecting sensor orientation
            magnetic_risk = 0.0
            if mag_magnitude > MAG_BASELINE:
                magnetic_risk = MAG_MAX_RISK  # 10 points

            # -------------------------------------------------------
            # COMBINE ALL COMPONENTS INTO 0–100 RISK INDEX
            # -------------------------------------------------------
            total_risk = (
                vibration_sensor_risk +
                ml_risk +
                distance_proximity_risk +
                movement_risk +
                magnetic_risk
            )
            total_risk = int(min(100.0, total_risk))  # Cap at 100

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
            print(f"  Vibration Sensor : {vibration_sensor_risk:.1f}/{VIBRATION_SENSOR_RISK} (hardware override)")
            print(f"  ML Probability   : {ml_risk}/{ML_MAX_RISK} (prob={prob_1:.2f})")
            print(f"  Distance Proximity: {distance_proximity_risk:.1f} (dist={distance:.2f} cm)")
            print(f"  Movement         : {movement_risk:.1f}/{MOVEMENT_MAX_RISK} (change={distance_change:.2f} cm)")
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


