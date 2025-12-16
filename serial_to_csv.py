# serial_to_csv.py
import argparse
import csv
import sys
import time
import serial  # pip install pyserial


def parse_args():
    p = argparse.ArgumentParser(description="Read ESP32 serial data and save to CSV.")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbserial-0001 or COM3")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--out", default="micro_seismic_data.csv")
    p.add_argument("--label", choices=["seismic", "non-seismic"], required=True)
    p.add_argument("--append", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mode = "a" if args.append else "w"

    header = ["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm", "label"]

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Failed to open serial port {args.port}: {e}")
        sys.exit(1)

    print(f"Reading from {args.port} @ {args.baud} → {args.out} (label={args.label})")
    print("Ctrl+C to stop.")

    with open(args.out, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Expect: time_ms,vibration,mag_x,mag_y,mag_z,distance_cm
                parts = line.split(",")
                if len(parts) != 6:
                    print(f"Skip malformed: {line}")
                    continue

                try:
                    t = int(parts[0])
                    vib = int(parts[1])
                    mx = float(parts[2])
                    my = float(parts[3])
                    mz = float(parts[4])
                    dist = float(parts[5])
                except ValueError:
                    print(f"Skip non-numeric: {line}")
                    continue

                writer.writerow([t, vib, mx, my, mz, dist, args.label])
                f.flush()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()


if __name__ == "__main__":
    main()