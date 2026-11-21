# Data Dictionary — Smart Home Energy Consumption

**Source:** https://www.kaggle.com/datasets/mexwell/smart-home-energy-consumption
**File:** `data/smart_home_energy_consumption_large.csv`
**Size:** 100,000 rows × 8 columns · 0 missing values · 0 duplicate rows
**Period covered:** 1 Jan 2023 – 31 Dec 2023 (366 distinct dates, 500 homes)

## Target

| Column | Type | Unit | Range | Mean | Notes |
|---|---|---|---|---|---|
| `Energy Consumption (kWh)` | float64 (continuous) | kilowatt-hours | 0.10 – 5.00 | 1.50 (sd 1.18) | Regression target. Continuous numeric value, **not** a class label. |

## Input features

| Column | Raw type | Unit / values | Use in model |
|---|---|---|---|
| `Home ID` | int64 | 1 – 500 | **Dropped** — an identifier, not a household characteristic (corr. with target ≈ 0.00) |
| `Appliance Type` | categorical (10) | Fridge, Oven, Dishwasher, Heater, Microwave, Air Conditioning, Computer, TV, Washing Machine, Lights | One-hot encode |
| `Time` | string `HH:MM` | 00:00 – 23:59 | Extract `hour` (0–23) |
| `Date` | string `YYYY-MM-DD` | 2023 | Extract `month` and `day_of_week` |
| `Outdoor Temperature (°C)` | float64 | °C, −10.0 to 40.0 | Numeric, scale for linear/polynomial models |
| `Season` | categorical (4) | Winter (Jan–Mar), Spring (Apr–Jun), Summer (Jul–Sep), Fall (Oct–Dec) | One-hot encode (note: derived from month, so redundant with `month`) |
| `Household Size` | int64 | 1 – 5 people | Numeric |

## Key EDA findings (important for results and viva)

1. **Appliance Type is the only real driver of energy use.**
   - Air Conditioning and Heater: 2.0 – 5.0 kWh (mean ≈ 3.5)
   - Fridge: 0.1 – 0.5 kWh (mean 0.30)
   - All other 7 appliances: 0.2 – 2.0 kWh (mean ≈ 1.10)
2. **Temperature, season and household size have almost no effect.** Correlations with the target are ≈ 0.00, and per-season and per-household-size averages are all ≈ 1.50 kWh. Example: heater usage averages the same in summer as in winter.
3. **Within each appliance, consumption looks uniformly random** (std ≈ range/√12, the signature of a uniform distribution). This suggests the dataset is **synthetic**.
4. **Consequence: there is a performance ceiling of R² ≈ 0.75.** Even a perfect model can only explain the between-appliance differences. A quick Random Forest baseline already reaches R² ≈ 0.75. Expect all four models to score similarly once Appliance Type is encoded properly.

> Recommended framing for the report: "Model comparison shows that appliance type explains ~75% of variance; the remaining variance is not explained by any available feature, indicating that richer data (appliance power rating, usage duration, real meter readings) would be needed for further gains."
