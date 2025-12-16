"""
Split a labeled micro seismic dataset into train / validation / test CSV files.

Assumes input CSV (e.g. micro_seismic_data_labeled.csv) has at least:
    time_ms, vibration, mag_x, mag_y, mag_z, distance_cm, label
"""

import argparse

import pandas as pd  # pip install pandas
from sklearn.model_selection import train_test_split  # pip install scikit-learn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split labeled dataset into train/val/test CSVs.")
    parser.add_argument(
        "--in_csv",
        required=True,
        help="Path to labeled input CSV (with 'label' column).",
    )
    parser.add_argument("--train_csv", default="train.csv", help="Output train CSV path.")
    parser.add_argument("--val_csv", default="val.csv", help="Output validation CSV path.")
    parser.add_argument("--test_csv", default="test.csv", help="Output test CSV path.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction (default: 0.15).")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test fraction (default: 0.15).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    # First: split off the test set
    temp_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.random_state,
    )

    # Then: split remaining data into train and validation
    val_fraction_of_temp = args.val_size / (1.0 - args.test_size)

    train_df, val_df = train_test_split(
        temp_df,
        test_size=val_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=args.random_state,
    )

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

    print(f"Input: {args.in_csv}")
    print(f"Train: {len(train_df)} rows -> {args.train_csv}")
    print(f"Val  : {len(val_df)} rows -> {args.val_csv}")
    print(f"Test : {len(test_df)} rows -> {args.test_csv}")


if __name__ == "__main__":
    main()


