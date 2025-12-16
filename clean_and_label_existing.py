"""
Clean and label a micro seismic CSV.

Default input format (based on your file):
    Line 1: clk_drv:0x00,q_drv:0x00,...
    Line 2: time_ms,vibration,mag_x,mag_y,mag_z,distance_cm
    Lines 3+: numeric data

Label rule:
    vibration == 1  -> "seismic"
    vibration == 0  -> "non-seismic"
"""

import argparse
import pandas as pd  # pip install pandas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and label a micro seismic CSV.")
    parser.add_argument(
        "--in_csv",
        default="micro_seismic_data.csv",
        help="Input CSV path (default: micro_seismic_data.csv in this folder).",
    )
    parser.add_argument(
        "--out_csv",
        default="micro_seismic_data_labeled.csv",
        help="Output labeled CSV path (default: micro_seismic_data_labeled.csv in this folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Skip the first "clk_drv:..." line
    df = pd.read_csv(args.in_csv, skiprows=1)

    required_cols = ["time_ms", "vibration", "mag_x", "mag_y", "mag_z", "distance_cm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    # Simple labeling rule; update here if your team wants a different rule
    df["label"] = df["vibration"].apply(lambda v: "seismic" if v == 1 else "non-seismic")

    df.to_csv(args.out_csv, index=False)

    print(f"Input CSV : {args.in_csv}")
    print(f"Output CSV: {args.out_csv}")
    print("Label counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()


