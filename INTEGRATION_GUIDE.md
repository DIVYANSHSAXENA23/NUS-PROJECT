# Hardware-Software Integration Guide

This guide explains how to integrate the Arduino/ESP32 hardware with the Python ML software for real-time monitoring.

## 🔌 System Architecture

```
Arduino/ESP32 (Hardware)  →  Serial/USB  →  Python (Software)  →  ML Predictions
     Sensors                   115200 baud      Real-time            Risk Index
```

## 📋 Prerequisites

1. **Hardware Setup:**
   - ESP32 or Arduino board
   - QMC5883L magnetometer sensor
   - HC-SR04 ultrasonic distance sensor
   - Vibration sensor
   - All sensors connected as per `NUS_proj.ino`

2. **Software Setup:**
   - Python 3.x installed
   - Arduino IDE (for uploading firmware)
   - All Python dependencies installed

## 🚀 Quick Start

### Step 1: Upload Arduino Code

1. Open `NUS_proj.ino` in Arduino IDE
2. Select your board (ESP32 or Arduino)
3. Select the correct COM port
4. Upload the code
5. Open Serial Monitor (115200 baud) to verify data is being sent

### Step 2: Train and Save ML Model

Before running real-time monitoring, train the model once:

```bash
python train_and_save_model.py
```

This will create:
- `rf_model.pkl` - Random Forest classifier
- `iso_model.pkl` - Isolation Forest for anomalies
- `mag_mean.pkl` - Mean magnetometer value for risk calculation

### Step 3: Run Real-Time Monitoring

**Windows:**
```bash
python real_time_monitor.py --port COM3
```

**Linux/Mac:**
```bash
python real_time_monitor.py --port /dev/ttyUSB0
```

**With data logging:**
```bash
python real_time_monitor.py --port COM3 --save
```

## 🔧 Finding Your Serial Port

### Windows
1. Open Device Manager
2. Look under "Ports (COM & LPT)"
3. Find your Arduino/ESP32 (usually COM3, COM4, etc.)

### Linux
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

### Mac
```bash
ls /dev/cu.usb*
```

Or check Arduino IDE: Tools → Port

## 📊 Usage Examples

### Basic Monitoring
```bash
python real_time_monitor.py --port COM3
```

### Custom Baud Rate
```bash
python real_time_monitor.py --port COM3 --baud 9600
```

### Save Live Data
```bash
python real_time_monitor.py --port COM3 --save
```
Data will be saved as `live_data_YYYYMMDD_HHMMSS.csv`

## 🎯 Output Explanation

The real-time monitor displays:

- **Sensor Readings**: Raw data from hardware
  - Distance (cm)
  - Magnetometer X, Y, Z values
  - Magnetometer magnitude
  - Vibration status

- **ML Predictions**:
  - Vibration detection (YES/NO)
  - Anomaly detection (YES/NO)

- **Risk Index** (0-100):
  - 🟢 LOW (0-29): Normal conditions
  - 🟡 MEDIUM (30-59): Some concern
  - 🟠 HIGH (60-79): Significant risk
  - 🔴 CRITICAL (80-100): Immediate action needed

## 🔍 Troubleshooting

### "Models not found" Error
**Solution:** Run `train_and_save_model.py` first

### "Serial port not found" Error
**Solution:** 
1. Check if Arduino is connected
2. Verify COM port in Device Manager
3. Make sure no other program is using the port (close Arduino IDE Serial Monitor)

### "Permission denied" (Linux/Mac)
**Solution:**
```bash
sudo chmod 666 /dev/ttyUSB0
```
Or add your user to the dialout group:
```bash
sudo usermod -a -G dialout $USER
```

### No Data Received
**Solution:**
1. Check baud rate matches (115200)
2. Verify Arduino code is uploaded correctly
3. Check sensor connections
4. Try resetting the Arduino board

### Data Parsing Errors
**Solution:**
- Ensure Arduino Serial Monitor is closed
- Check that data format matches: `time_ms,vibration,mag_x,mag_y,mag_z,distance_cm`

## 📝 Code Structure

### `train_and_save_model.py`
- Trains ML models on historical data
- Saves models for real-time use
- Run once before monitoring

### `real_time_monitor.py`
- Connects to Arduino via serial
- Reads sensor data in real-time
- Makes predictions using saved models
- Displays risk index and alerts

### `NUS_proj.ino`
- Arduino firmware
- Reads sensors every 500ms
- Sends CSV data via Serial

## 🔄 Workflow

1. **Initial Setup:**
   ```
   Upload Arduino Code → Train Model → Start Monitoring
   ```

2. **Daily Use:**
   ```
   Connect Hardware → Run real_time_monitor.py → Monitor Output
   ```

3. **Data Collection:**
   ```
   Run with --save flag → Analyze CSV files later
   ```

## 💡 Tips

- **Calibration**: Let sensors stabilize for 10-20 seconds after power-on
- **Sampling Rate**: Arduino sends data every 500ms (2 Hz)
- **Risk Thresholds**: Adjust in `compute_risk_index()` function if needed
- **Model Updates**: Retrain model periodically with new data for better accuracy

## 🎓 Next Steps

- Add email/SMS alerts for critical risk levels
- Create web dashboard for remote monitoring
- Store data in database for historical analysis
- Add more sensors (temperature, humidity, etc.)

## 📞 Support

For issues or questions, please open an issue on GitHub.

