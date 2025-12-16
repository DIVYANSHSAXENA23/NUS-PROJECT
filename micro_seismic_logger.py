import serial
import csv
import time

PORT = "COM3"        # CHANGE THIS
BAUD = 115200
CSV_FILE = "micro_seismic_data.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

with open(CSV_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    print("Recording started... Press CTRL+C to stop\n")

    try:
        while True:
            line = ser.readline().decode("utf-8").strip()
            if line:
                print(line)
                data = line.split(",")
                if len(data) == 6:
                    writer.writerow(data)

    except KeyboardInterrupt:
        print("\nRecording stopped.")
        ser.close()
