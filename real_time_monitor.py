"""
Real-time Hardware-Software Integration
Reads sensor data from Arduino/ESP32 via Serial and makes real-time predictions
"""

import serial
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import sys

# Try to load models (train first if they don't exist)
try:
    rf_model = joblib.load("rf_model.pkl")
    iso_model = joblib.load("iso_model.pkl")
    mag_mean = joblib.load("mag_mean.pkl")
    print("✓ Models loaded successfully")
except FileNotFoundError:
    print("❌ Models not found! Please run 'train_and_save_model.py' first.")
    sys.exit(1)


class RealTimeMonitor:
    def __init__(self, port='COM3', baud_rate=115200):
        """
        Initialize serial connection to Arduino/ESP32
        
        Args:
            port: Serial port (Windows: COM3, COM4, etc. | Linux/Mac: /dev/ttyUSB0, /dev/ttyACM0)
            baud_rate: Serial baud rate (must match Arduino code)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.previous_distance = None
        self.vibration_history = []
        
    def connect(self):
        """Establish serial connection"""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            print(f"✓ Connected to {self.port} at {self.baud_rate} baud")
            time.sleep(2)  # Wait for Arduino to initialize
            # Skip header line
            self.ser.readline()
            return True
        except serial.SerialException as e:
            print(f"❌ Error connecting to {self.port}: {e}")
            print("\nAvailable ports:")
            self.list_ports()
            return False
    
    def list_ports(self):
        """List available serial ports"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if ports:
            print("\nAvailable COM ports:")
            for port in ports:
                print(f"  - {port.device}: {port.description}")
        else:
            print("  No COM ports found")
    
    def read_sensor_data(self):
        """Read one line of sensor data from Arduino"""
        if self.ser and self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) == 6:
                        return {
                            'timestamp': int(parts[0]),
                            'vibration': int(parts[1]),
                            'mag_x': int(parts[2]),
                            'mag_y': int(parts[3]),
                            'mag_z': int(parts[4]),
                            'distance': float(parts[5])
                        }
            except (ValueError, UnicodeDecodeError) as e:
                print(f"⚠ Error parsing data: {e}")
        return None
    
    def compute_features(self, data):
        """Compute features from raw sensor data"""
        # Magnetometer magnitude
        mag_magnitude = np.sqrt(data['mag_x']**2 + data['mag_y']**2 + data['mag_z']**2)
        
        # Distance change
        if self.previous_distance is not None:
            distance_change = data['distance'] - self.previous_distance
        else:
            distance_change = 0
        self.previous_distance = data['distance']
        
        # Vibration frequency (rolling average of last 5 readings)
        self.vibration_history.append(data['vibration'])
        if len(self.vibration_history) > 5:
            self.vibration_history.pop(0)
        vibration_freq = np.mean(self.vibration_history)
        
        return {
            'distance': data['distance'],
            'distance_change': distance_change,
            'mag_magnitude': mag_magnitude,
            'vibration_freq': vibration_freq
        }
    
    def predict(self, features):
        """Make predictions using trained models"""
        # Prepare feature vector
        X = pd.DataFrame([[
            features['distance'],
            features['distance_change'],
            features['mag_magnitude'],
            features['vibration_freq']
        ]], columns=['distance', 'distance_change', 'mag_magnitude', 'vibration_freq'])
        
        # Vibration prediction
        vibration_pred = rf_model.predict(X)[0]
        
        # Anomaly detection
        anomaly_flag = iso_model.predict(X)[0]
        anomaly_score = iso_model.decision_function(X)[0]
        
        return vibration_pred, anomaly_flag, anomaly_score
    
    def compute_risk_index(self, vibration, distance_change, mag_magnitude, anomaly_flag):
        """Compute disaster risk index (0-100)"""
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
    
    def get_risk_level(self, risk_index):
        """Get risk level description"""
        if risk_index < 30:
            return "LOW", "🟢"
        elif risk_index < 60:
            return "MEDIUM", "🟡"
        elif risk_index < 80:
            return "HIGH", "🟠"
        else:
            return "CRITICAL", "🔴"
    
    def display_result(self, data, features, vibration_pred, anomaly_flag, risk_index, risk_level, risk_emoji):
        """Display formatted result"""
        print("\n" + "=" * 70)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        print(f"📊 Sensor Readings:")
        print(f"   Distance: {data['distance']:.2f} cm")
        print(f"   Magnetometer: X={data['mag_x']}, Y={data['mag_y']}, Z={data['mag_z']}")
        print(f"   Magnitude: {features['mag_magnitude']:.2f}")
        print(f"   Vibration: {data['vibration']} (Raw)")
        print("-" * 70)
        print(f"🤖 ML Predictions:")
        print(f"   Vibration Detected: {'YES' if vibration_pred == 1 else 'NO'}")
        print(f"   Anomaly: {'YES' if anomaly_flag == -1 else 'NO'}")
        print("-" * 70)
        print(f"{risk_emoji} Risk Index: {risk_index}/100 - {risk_level}")
        print("=" * 70)
    
    def run(self, save_data=False):
        """Main monitoring loop"""
        if not self.connect():
            return
        
        print("\n" + "=" * 70)
        print("🚀 REAL-TIME MONITORING STARTED")
        print("=" * 70)
        print("Press Ctrl+C to stop\n")
        
        data_log = []
        
        try:
            while True:
                # Read sensor data
                data = self.read_sensor_data()
                
                if data:
                    # Compute features
                    features = self.compute_features(data)
                    
                    # Make predictions
                    vibration_pred, anomaly_flag, anomaly_score = self.predict(features)
                    
                    # Compute risk index
                    risk_index = self.compute_risk_index(
                        vibration_pred,
                        features['distance_change'],
                        features['mag_magnitude'],
                        anomaly_flag
                    )
                    
                    # Get risk level
                    risk_level, risk_emoji = self.get_risk_level(risk_index)
                    
                    # Display result
                    self.display_result(data, features, vibration_pred, anomaly_flag, 
                                      risk_index, risk_level, risk_emoji)
                    
                    # Save data if requested
                    if save_data:
                        data_log.append({
                            'timestamp': data['timestamp'],
                            'distance': data['distance'],
                            'mag_x': data['mag_x'],
                            'mag_y': data['mag_y'],
                            'mag_z': data['mag_z'],
                            'vibration': data['vibration'],
                            'vibration_pred': vibration_pred,
                            'anomaly_flag': anomaly_flag,
                            'risk_index': risk_index
                        })
                
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("⏹️  MONITORING STOPPED")
            print("=" * 70)
            
            if save_data and data_log:
                df_log = pd.DataFrame(data_log)
                filename = f"live_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_log.to_csv(filename, index=False)
                print(f"\n✓ Data saved to {filename}")
        
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("✓ Serial connection closed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time micro-seismic monitoring')
    parser.add_argument('--port', type=str, default='COM3',
                       help='Serial port (default: COM3 for Windows, /dev/ttyUSB0 for Linux)')
    parser.add_argument('--baud', type=int, default=115200,
                       help='Baud rate (default: 115200)')
    parser.add_argument('--save', action='store_true',
                       help='Save live data to CSV file')
    
    args = parser.parse_args()
    
    # Auto-detect port on Linux/Mac
    import platform
    if platform.system() != 'Windows' and args.port == 'COM3':
        args.port = '/dev/ttyUSB0'  # Common Linux port
    
    monitor = RealTimeMonitor(port=args.port, baud_rate=args.baud)
    monitor.run(save_data=args.save)


if __name__ == "__main__":
    main()

