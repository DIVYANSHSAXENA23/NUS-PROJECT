# Micro-Seismic & Structural Instability Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning-based system for detecting micro-seismic activity and structural instability using sensor data. This project combines Random Forest classification and Isolation Forest anomaly detection to predict vibrations and compute a disaster risk index.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes sensor data (magnetometer readings, distance measurements, and vibration signals) to:
- **Predict micro-seismic vibrations** using a Random Forest classifier
- **Detect anomalies** in sensor readings using Isolation Forest
- **Compute a disaster risk index** (0-100) based on multiple factors

The system is designed for real-time monitoring of structural integrity and early warning of potential structural failures.

## ✨ Features

- **Feature Engineering**: 
  - Magnetometer magnitude calculation (3-axis)
  - Distance change detection (structural movement)
  - Rolling vibration frequency analysis

- **Machine Learning Models**:
  - Random Forest Classifier for vibration prediction
  - Isolation Forest for anomaly detection

- **Risk Assessment**:
  - Multi-factor disaster risk index (0-100)
  - Real-time prediction function for live monitoring

## 🛠️ Technologies Used

### Software
- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning models and evaluation metrics
  - RandomForestClassifier
  - IsolationForest
  - train_test_split
  - accuracy_score, confusion_matrix, classification_report

### Hardware & Firmware
- **Arduino/ESP32** - Microcontroller for sensor data collection
- **QMC5883L** - 3-axis magnetometer sensor
- **HC-SR04** - Ultrasonic distance sensor
- **Vibration Sensor** - Digital vibration detector

## 📦 Installation

### Prerequisites

- Python 3.x installed on your system
- (Optional) Arduino IDE if you want to use the sensor data collection code

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DIVYANSHSAXENA23/NUS-PROJECT.git
   cd NUS-PROJECT
   ```

2. **Install required Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using `py` on Windows:
   ```bash
   py -m pip install -r requirements.txt
   ```
   
   This will install: pandas, numpy, scikit-learn, matplotlib, seaborn

3. **Hardware Setup (Optional)**:
   - Upload `NUS_proj.ino` to your ESP32/Arduino board
   - Connect sensors as per pin definitions in the code:
     - Vibration sensor → Pin 34
     - Ultrasonic sensor → Trig: Pin 26, Echo: Pin 27
     - QMC5883L magnetometer → I2C (SDA: Pin 21, SCL: Pin 22)
   - Open Serial Monitor at 115200 baud to collect data

## 🚀 Usage

### Data Collection (Hardware)

If you want to collect your own sensor data:

1. Upload `NUS_proj.ino` to your ESP32/Arduino board using Arduino IDE
2. Connect the sensors as specified in the code
3. Open Serial Monitor (115200 baud) to view CSV output
4. Copy the serial output and save it as `micro_seismic_data.csv`

### Running the Model

Simply execute the main script:

```bash
python micro_seismic_model.py
```

Or on Windows:
```bash
py micro_seismic_model.py
```

### Visualization

Generate visualizations and analysis plots:

```bash
python visualize_results.py
```

This will create:
- `analysis_results.png` - Comprehensive analysis dashboard
- `risk_index_timeline.png` - Risk index over time visualization

### Data Format

The script expects a CSV file named `micro_seismic_data.csv` with the following columns:
- `time_ms` - Timestamp in milliseconds
- `vibration` - Binary vibration indicator (0 or 1)
- `mag_x`, `mag_y`, `mag_z` - Magnetometer readings (3-axis)
- `distance_cm` - Distance measurement in centimeters

### Live Prediction

The script includes a function for real-time predictions. **Note**: The model must be trained first by running the full script. After training, you can use the `predict_vibration()` function:

```python
# The function is defined in the script and uses the trained rf_model
# Example usage (after running the script):
prediction = predict_vibration(
    distance=50.0,
    mag_x=-42,
    mag_y=-158,
    mag_z=1498,
    vibration_freq=0.2
)

print(f"Vibration detected: {prediction}")  # Returns 0 or 1
```

## 📁 Project Structure

```
NUS-PROJECT/
│
├── micro_seismic_model.py           # Main ML pipeline script
├── visualize_results.py              # Data visualization and analysis
├── micro_seismic_data.csv           # Input sensor data
├── micro_seismic_data_augmented.csv # Augmented dataset (if available)
├── NUS_proj.ino                     # Arduino code for sensor data collection
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
├── CONTRIBUTING.md                  # Contribution guidelines
└── README.md                        # Project documentation
```

## 🧠 Model Details

### Feature Engineering

1. **Magnetometer Magnitude**: 
   ```python
   mag_magnitude = sqrt(mag_x² + mag_y² + mag_z²)
   ```
   Measures orientation instability.

2. **Distance Change**: 
   ```python
   distance_change = diff(distance)
   ```
   Tracks structural movement over time.

3. **Vibration Frequency**: 
   ```python
   vibration_freq = rolling_mean(vibration, window=5)
   ```
   Calculates rolling average of vibration signals.

### Model Architecture

- **Random Forest Classifier**:
  - 100 estimators
  - Trained on 80% of data (20% test set)
  - Stratified split to maintain class distribution

- **Isolation Forest**:
  - Contamination rate: 10%
  - Detects anomalies in feature space

### Risk Index Calculation

The disaster risk index (0-100) is computed based on:
- **Vibration detected** (+40 points)
- **Large distance change** (>5 units) (+25 points)
- **High magnetometer magnitude** (above mean) (+20 points)
- **Anomaly detected** (+30 points)

Maximum risk score is capped at 100.

## 📊 Output

The script outputs:

1. **Dataset Preview**: First 5 rows of loaded data
2. **Model Performance Metrics**:
   - Accuracy score
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
3. **Final Results**: DataFrame with:
   - Timestamp
   - Vibration status
   - Distance measurements
   - Magnetometer magnitude
   - Anomaly flags
   - Risk index

### Example Output

```
MODEL PERFORMANCE
Accuracy: 0.9705882352941176

Confusion Matrix:
[[33  0]
 [ 1  0]]

FINAL OUTPUT WITH RISK INDEX
   timestamp  vibration  distance  mag_magnitude  anomaly_flag  risk_index
0         52          0     -1.00    1506.894820             1           0
1        584          0     -1.00    1504.697976             1           0
...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Quick steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

**Divyansh Saxena**
- GitHub: [@DIVYANSHSAXENA23](https://github.com/DIVYANSHSAXENA23)

**Keshav Kaushish**
- GitHub: [@keshavkaushish25-debug](https://github.com/keshavkaushish25-debug)

**Ssanvee Vijay**
- GitHub: [@SsanveeVijay](https://github.com/SsanveeVijay)

## 🙏 Acknowledgments

- National University of Singapore (NUS) for project support
- scikit-learn community for excellent ML tools

---

⭐ If you find this project helpful, please consider giving it a star!

