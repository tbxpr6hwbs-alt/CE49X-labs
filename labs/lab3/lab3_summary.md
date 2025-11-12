# Lab 3 Summary: ERA5 Wind Analysis (Berlin vs Munich, UTC)

## Findings (3–5 sentences)
Using ERA5 10 m wind components for Berlin and Munich, I computed wind speed as √(u10m² + v10m²) and analyzed it in UTC.
Monthly means show clear variability across months, with Munich and Berlin exhibiting comparable magnitudes but different month-to-month patterns.
Seasonal aggregation (DJF/MAM/JJA/SON) indicates noticeable differences between warm and cold seasons; note that DJF may be partially represented depending on the data window.
The diurnal cycle (UTC) reveals modest hour-of-day structure, with peaks and lulls varying by city; this should be interpreted cautiously since local time effects are not included by choice.
Overall distributional statistics are summarized below.

## Key Stats (wind speed, m/s)
- Berlin: mean=3.312 m/s, p95=6.241 m/s, max=8.547 m/s (n=1100)
- Munich: mean=2.502 m/s, p95=5.237 m/s, max=9.055 m/s (n=1100)

## How I might use Skyrim in a civil/environmental project
Skyrim is an open-source toolkit that streamlines access to numerical weather prediction and forecast data (e.g., downloading, organizing, and programmatically querying model outputs). In a civil/environmental engineering context, I would use Skyrim to automate retrieval of wind (and other meteorological) forecasts for construction scheduling, crane operation safety envelopes, and wind loading checks on temporary structures. By integrating Skyrim into a pipeline with pandas and matplotlib, I could pull fresh forecasts, compute site-specific wind exceedance probabilities, and trigger alerts or dashboards that inform field teams. This approach reduces manual data handling, increases reproducibility, and supports better risk-aware decision making on job sites.

## Artifacts produced
- Script: `labs/lab3/lab3_era5_analysis.py`
- CSVs: `city_wind_speed_stats.csv`, `monthly_wind_speed.csv`, `seasonal_wind_speed.csv`, `diurnal_wind_speed_utc.csv`
- Figures: `fig_monthly_wind_speed.png`, `fig_seasonal_wind_speed.png`, `fig_diurnal_wind_speed_utc.png`

## Notes
- All computations use UTC, per instruction.
- Only wind-speed aggregates were computed (no temperature), per instruction.
- Seasonal means may represent partial seasons if the dataset does not cover a complete year.
