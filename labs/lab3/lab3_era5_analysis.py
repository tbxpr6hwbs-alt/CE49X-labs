#!/usr/bin/env python3
"""
Lab 3: ERA5 Weather Data Analysis
- Loads ERA5 10m wind components (u10m, v10m) for Berlin and Munich
- Computes wind speed and basic wind direction
- Produces monthly, seasonal, and diurnal (UTC) aggregates
- Saves summary CSVs and three figures
- Designed to be robust and easy to reuse

Usage:
  python lab3_era5_analysis.py \
      --berlin_csv /path/to/berlin_era5_wind_20241231_20241231.csv \
      --munich_csv /path/to/munich_era5_wind_20241231_20241231.csv \
      --out_dir ./labs/lab3

Notes:
- Times are treated as UTC (per lab instruction).
- Seasons (DJF, MAM, JJA, SON) are assigned by month. DJF is split across years.
- Only wind-speed aggregates are computed (per clarification).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_city_csv(path: Path, city_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (lowercase)
    df.columns = [c.lower() for c in df.columns]
    for col in ["timestamp", "u10m", "v10m"]:
        if col not in df.columns:
            raise ValueError(f"{city_name}: Missing required column: {col}")
    # Parse timestamps (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        bad = df[df["timestamp"].isna()]
        raise ValueError(f"{city_name}: Found unparsable timestamps in rows: {bad.index.tolist()[:5]} ...")
    # Compute wind speed and direction
    df["wind_speed"] = np.sqrt(df["u10m"]**2 + df["v10m"]**2)
    theta_rad = np.arctan2(df["v10m"], df["u10m"])
    theta_deg = np.degrees(theta_rad)
    df["wind_dir_deg"] = (270 - theta_deg) % 360
    df["city"] = city_name
    return df

def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF (Winter)"
    if m in (3, 4, 5):
        return "MAM (Spring)"
    if m in (6, 7, 8):
        return "JJA (Summer)"
    return "SON (Autumn)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--berlin_csv", type=Path, required=True, help="Path to Berlin CSV file")
    parser.add_argument("--munich_csv", type=Path, required=True, help="Path to Munich CSV file")
    parser.add_argument("--out_dir", type=Path, default=Path("./labs/lab3"), help="Output directory")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    berlin = load_city_csv(args.berlin_csv, "Berlin")
    munich = load_city_csv(args.munich_csv, "Munich")
    df = pd.concat([berlin, munich], ignore_index=True)

    # Time features (UTC)
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["hour_utc"] = df["timestamp"].dt.hour
    df["season"] = df["month"].map(month_to_season)

    # City summary stats
    city_stats = (
        df.groupby("city")["wind_speed"]
          .agg(mean="mean", std="std", min="min", p50="median", p95=lambda s: s.quantile(0.95), max="max", n="count")
          .reset_index()
          .round(3)
    )
    city_stats.to_csv(out / "city_wind_speed_stats.csv", index=False)

    # Monthly mean wind speed
    monthly = (
        df.groupby(["city", "year", "month", "month_name"])["wind_speed"]
          .mean()
          .reset_index()
          .sort_values(["city", "year", "month"])
    )
    monthly["wind_speed"] = monthly["wind_speed"].round(3)
    monthly.to_csv(out / "monthly_wind_speed.csv", index=False)

    # Seasonal mean wind speed
    seasonal = (
        df.groupby(["city", "year", "season"])["wind_speed"]
          .mean()
          .reset_index()
          .sort_values(["city", "year", "season"])
    )
    seasonal["wind_speed"] = seasonal["wind_speed"].round(3)
    seasonal.to_csv(out / "seasonal_wind_speed.csv", index=False)

    # Diurnal cycle (UTC)
    diurnal = (
        df.groupby(["city", "hour_utc"])["wind_speed"]
          .mean()
          .reset_index()
          .sort_values(["city", "hour_utc"])
    )
    diurnal["wind_speed"] = diurnal["wind_speed"].round(3)
    diurnal.to_csv(out / "diurnal_wind_speed_utc.csv", index=False)

    # --- Plots ---
    # 1) Monthly averages line chart
    plt.figure()
    for city in ["Berlin", "Munich"]:
        sub = monthly[monthly["city"] == city]
        x = sub.apply(lambda r: f'{int(r["year"])}-{int(r["month"]):02d}', axis=1)
        y = sub["wind_speed"].values
        plt.plot(x, y, marker="o", label=city)
    plt.title("Monthly Mean 10m Wind Speed (UTC)")
    plt.xlabel("Month")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_monthly_wind_speed.png", dpi=150)
    plt.close()

    # 2) Seasonal averages bar chart
    season_order = ["DJF (Winter)", "MAM (Spring)", "JJA (Summer)", "SON (Autumn)"]
    seasonal_avg = seasonal.groupby(["city", "season"])["wind_speed"].mean().reset_index()
    x = np.arange(len(season_order))
    width = 0.35
    ber_vals = [seasonal_avg.query("city=='Berlin' and season==@s")["wind_speed"].mean() if not seasonal_avg.query("city=='Berlin' and season==@s").empty else np.nan for s in season_order]
    mun_vals = [seasonal_avg.query("city=='Munich' and season==@s")["wind_speed"].mean() if not seasonal_avg.query("city=='Munich' and season==@s").empty else np.nan for s in season_order]
    plt.figure()
    plt.bar(x - width/2, ber_vals, width, label="Berlin")
    plt.bar(x + width/2, mun_vals, width, label="Munich")
    plt.xticks(x, season_order)
    plt.title("Seasonal Mean 10m Wind Speed (UTC)")
    plt.xlabel("Season")
    plt.ylabel("Wind speed (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_seasonal_wind_speed.png", dpi=150)
    plt.close()

    # 3) Diurnal cycle
    plt.figure()
    for city in ["Berlin", "Munich"]:
        sub = diurnal[diurnal["city"] == city]
        plt.plot(sub["hour_utc"], sub["wind_speed"], marker="o", label=city)
    plt.title("Diurnal Cycle of 10m Wind Speed (UTC)")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(range(0,24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_diurnal_wind_speed_utc.png", dpi=150)
    plt.close()

    print("Done. Outputs saved to:", out.resolve())

if __name__ == "__main__":
    main()
