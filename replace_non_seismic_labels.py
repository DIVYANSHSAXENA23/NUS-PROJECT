"""
Replace 'non-seismic' labels with 'seismic' in a CSV, without changing anything else.

Default target file is micro_seismic_data_augmented.csv in this folder.
"""

import argparse

import pandas as pd  # pip install pandas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace 'non-seismic' with 'seismic' in the label column of a CSV."
    )
    parser.add_argument(
        "--csv",
        default="micro_seismic_data_augmented.csv",
        help="CSV file to modify in-place (default: micro_seismic_data_augmented.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)

    if "label" not in df.columns:
        raise ValueError(f"File {args.csv} does not have a 'label' column.")

    before_counts = df["label"].value_counts(dropna=False)

    # Replace only the literal string 'non-seismic'
    df["label"] = df["label"].replace("non-seismic", "seismic")

    after_counts = df["label"].value_counts(dropna=False)

    df.to_csv(args.csv, index=False)

    print(f"Updated {args.csv} in place.")
    print("Label counts before:")
    print(before_counts)
    print("\nLabel counts after:")
    print(after_counts)


if __name__ == "__main__":
    main()


