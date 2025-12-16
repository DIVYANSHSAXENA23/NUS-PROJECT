"""
Generate additional synthetic micro seismic data from an existing labeled CSV.

Approach (simple AIML-style augmentation):
- Read your labeled dataset (e.g. micro_seismic_data_labeled.csv).
- Sample existing rows with replacement and add small Gaussian noise to numeric
  sensor values.
- Force all synthetic rows to label = "seismic" (no class splitting).

This is not a full physics simulator, but gives you more rows that are
statistically similar to your real data for prototyping ML models.
"""

import argparse
from typing import List

import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas


NUMERIC_COLUMNS: List[str] = [
    "time_ms",
    "mag_x",
    "mag_y",
    "mag_z",
    "distance_cm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a labeled micro seismic dataset.")
    parser.add_argument(
        "--in_csv",
        default="micro_seismic_data_labeled.csv",
        help="Input labeled CSV path (default: micro_seismic_data_labeled.csv).",
    )
    parser.add_argument(
        "--out_csv",
        default="micro_seismic_data_augmented.csv",
        help="Output augmented CSV path (default: micro_seismic_data_augmented.csv).",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="How many extra samples to create relative to original size "
        "(e.g. 1.0 = same number of new rows as original).",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.05,
        help="Scale of Gaussian noise as a fraction of each column's standard deviation.",
    )
    return parser.parse_args()


def augment_all(
    df_all: pd.DataFrame,
    n_new: int,
    noise_scale: float,
) -> pd.DataFrame:
    """Create n_new synthetic rows (all labeled seismic) by sampling + noise."""
    if n_new <= 0 or df_all.empty:
        return df_all.iloc[0:0].copy()

    # Sample base rows with replacement
    sampled = df_all.sample(n=n_new, replace=True, random_state=None).reset_index(drop=True)

    # Compute per-column statistics for numeric columns
    means = df_all[NUMERIC_COLUMNS].mean()
    stds = df_all[NUMERIC_COLUMNS].std().replace(0, 1e-9)

    # Draw Gaussian noise
    noise = np.random.randn(n_new, len(NUMERIC_COLUMNS)) * (stds.values * noise_scale)

    # Apply noise
    sampled_numeric = sampled[NUMERIC_COLUMNS].to_numpy(dtype=float) + noise

    # For time_ms and distance, make sure they stay non-negative
    for idx, col in enumerate(NUMERIC_COLUMNS):
        if col in ("time_ms", "distance_cm"):
            sampled_numeric[:, idx] = np.clip(sampled_numeric[:, idx], a_min=0.0, a_max=None)

    # Put numeric data back
    for i, col in enumerate(NUMERIC_COLUMNS):
        sampled[col] = sampled_numeric[:, i]

    # Force label to seismic for all synthetic rows
    sampled["label"] = "seismic"
    # Optionally set vibration to 1 to be consistent with seismic label
    sampled["vibration"] = 1

    return sampled


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    required_cols = NUMERIC_COLUMNS + ["vibration", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    n_original = len(df)
    n_new_total = int(n_original * args.factor)

    if n_new_total <= 0:
        raise ValueError("factor must be > 0 to create additional samples.")

    print(f"Loaded {n_original} rows from {args.in_csv}")
    print(f"Generating ~{n_new_total} synthetic rows (factor={args.factor})...")

    # Create synthetic data (all labeled seismic)
    df_new = augment_all(df, n_new_total, args.noise_scale)

    # Combine original + synthetic
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(args.out_csv, index=False)

    print(f"Saved augmented dataset with {len(df_combined)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


