
"""
CE 49X: Introduction to Computational Thinking and Data Science
Lab 02: Soil Test Data Analysis

Script name: lab02_soil_analysis.py

What this script does (per assignment):
  - Loads soil_test.csv with pandas.
  - Cleans data by handling missing values (fill with column mean by default or drop rows).
  - (Optional) Removes outliers for a chosen numeric column using a z-score threshold.
  - Computes descriptive statistics (min, max, mean, median, std) for one or more numeric columns.
  - Organizes code into functions and includes error handling.

How to run (examples):
  python lab02_soil_analysis.py
  python lab02_soil_analysis.py --csv /mnt/data/soil_test.csv
  python lab02_soil_analysis.py --missing drop
  python lab02_soil_analysis.py --column soil_ph nitrogen --outlier_col soil_ph --zscore 3.0

Requirements:
  pip install pandas numpy

Reflection (answered in comments as requested by the lab):
  Q1) Why might filling NaNs with the mean be preferable to dropping rows?
      A1) It keeps sample size intact and can reduce bias introduced by removing non-randomly missing rows.
          However, it shrinks variance slightly and may understate uncertainty.
  Q2) What is the effect of outliers on basic statistics?
      A2) Outliers can skew the mean and inflate the standard deviation; median is more robust.
  Q3) When is it safer to drop rows with missing values?
      A3) When the proportion of missing is small, the missingness mechanism is close to MCAR,
          or when imputation assumptions would be too strong for the analysis goal.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Core functionality
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the soil test dataset from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"CSV file not found at '{csv_path}'. Please check the path and try again."
        ) from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"The CSV at '{csv_path}' is malformed: {e}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Handle missing values using the provided strategy.

    strategy: 'mean' (default) or 'drop'
    columns: columns to consider; if None, use numeric cols for 'mean' and all cols for 'drop'.
    """
    df_clean = df.copy()
    if strategy not in {"mean", "drop"}:
        raise ValueError("strategy must be either 'mean' or 'drop'")

    if strategy == "mean":
        target_cols = list(columns) if columns is not None else df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in target_cols:
            if col not in df_clean.columns:
                raise KeyError(f"Column '{col}' not found for mean imputation.")
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean(skipna=True))
    else:  # drop
        df_clean = df_clean.dropna(subset=list(columns) if columns is not None else None)
    return df_clean


def remove_outliers_zscore(
    df: pd.DataFrame,
    column: Optional[str] = None,
    z: float = 3.0,
) -> pd.DataFrame:
    """Remove outliers from a numeric column using a simple z-score threshold."""
    if column is None:
        return df.copy()
    if column not in df.columns:
        raise KeyError(f"Outlier removal column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Outlier removal requires a numeric column; '{column}' is not numeric.")

    series = df[column].astype(float)
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return df.copy()
    zscores = (series - mean) / std
    mask = zscores.abs() <= z
    return df.loc[mask].copy()


def compute_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Compute min, max, mean, median, std for a numeric column."""
    if column not in df.columns:
        raise KeyError(f"Statistics column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Statistics require a numeric column; '{column}' is not numeric.")
    s = df[column].astype(float)
    return {
        "min": float(s.min(skipna=True)),
        "max": float(s.max(skipna=True)),
        "mean": float(s.mean(skipna=True)),
        "median": float(s.median(skipna=True)),
        "std": float(s.std(skipna=True, ddof=0)),
    }


def print_statistics(stats: Dict[str, float], column: str) -> None:
    """Pretty-print descriptive statistics for a column."""
    print(f"\nDescriptive statistics for '{column}':")
    print("-" * (len(column) + 30))
    print(f"Minimum:           {stats['min']:.3f}")
    print(f"Maximum:           {stats['max']:.3f}")
    print(f"Mean:              {stats['mean']:.3f}")
    print(f"Median:            {stats['median']:.3f}")
    print(f"Standard Deviation:{stats['std']:.3f}")


def infer_default_columns(df: pd.DataFrame) -> List[str]:
    """Infer sensible default columns for stats: prefer 'soil_ph' if present; otherwise use all numeric columns except ID-like cols."""
    if 'soil_ph' in df.columns and pd.api.types.is_numeric_dtype(df['soil_ph']):
        return ['soil_ph']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in {"sample_id", "id"}]
    return numeric_cols or []


# -----------------------------
# CLI and script entry
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 02: Soil Test Data Analysis")
    parser.add_argument(
        "--csv", type=str, default="soil_test.csv",
        help="Path to the soil_test.csv file (default: soil_test.csv)",
    )
    parser.add_argument(
        "--missing", choices=["mean", "drop"], default="mean",
        help="Missing value handling strategy (default: mean)",
    )
    parser.add_argument(
        "--column", nargs="*", default=None,
        help="Column(s) to compute statistics for. If omitted, uses 'soil_ph' if available; otherwise all numeric columns.",
    )
    parser.add_argument(
        "--outlier_col", type=str, default=None,
        help="Optional column name for outlier removal using z-scores (e.g., 'soil_ph').",
    )
    parser.add_argument(
        "--zscore", type=float, default=3.0,
        help="Absolute z-score threshold for outlier removal (default: 3.0).",
    )
    parser.add_argument(
        "--save_clean", action="store_true",
        help="If provided, save the cleaned dataset to 'soil_test_cleaned.csv'.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1) Load
    try:
        df = load_data(args.csv)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
        return 1

    # 2) Clean missing
    try:
        df_clean = handle_missing_values(df, strategy=args.missing, columns=None)
    except Exception as e:
        print(f"[ERROR] Failed during missing value handling: {e}", file=sys.stderr)
        return 1

    # 3) Optional: outlier removal
    try:
        df_clean = remove_outliers_zscore(df_clean, column=args.outlier_col, z=args.zscore)
    except Exception as e:
        print(f"[ERROR] Failed during outlier removal: {e}", file=sys.stderr)
        return 1

    # Optionally save cleaned data next to the input
    if args.save_clean:
        try:
            out_path = "soil_test_cleaned.csv"
            df_clean.to_csv(out_path, index=False)
            print(f"[INFO] Cleaned dataset saved to '{out_path}'.")
        except Exception as e:
            print(f"[WARN] Could not save cleaned dataset: {e}", file=sys.stderr)

    # 4) Determine which columns to summarize
    if args.column is not None and len(args.column) > 0:
        columns_to_summarize = args.column
    else:
        columns_to_summarize = infer_default_columns(df_clean)
        if not columns_to_summarize:
            print("[WARN] No numeric columns found to summarize.")
            return 0

    # 5) Compute & print stats
    for col in columns_to_summarize:
        try:
            stats = compute_statistics(df_clean, col)
            print_statistics(stats, col)
        except Exception as e:
            print(f"[WARN] Skipping column '{col}': {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
