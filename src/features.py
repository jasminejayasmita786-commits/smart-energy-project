"""Shared feature engineering for training (src/train.py) and the Streamlit app.

The preprocessing pipeline built here is pickled with joblib, so this module must
stay importable as ``src.features`` from the project root.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TARGET = "Energy Consumption (kWh)"
CATEGORICAL = ["Appliance Type", "Season"]
NUMERIC = ["Outdoor Temperature (°C)", "Household Size", "hour", "month", "day_of_week"]
RAW_INPUTS = ["Appliance Type", "Time", "Date", "Outdoor Temperature (°C)", "Season", "Household Size"]

APPLIANCES = ["Air Conditioning", "Computer", "Dishwasher", "Fridge", "Heater",
              "Lights", "Microwave", "Oven", "TV", "Washing Machine"]
SEASONS = ["Fall", "Spring", "Summer", "Winter"]


def season_from_month(month: int) -> str:
    """Season convention used in the dataset: Winter Jan–Mar, Spring Apr–Jun, Summer Jul–Sep, Fall Oct–Dec."""
    return ["Winter", "Spring", "Summer", "Fall"][(month - 1) // 3]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Home ID, turn Time into hour and Date into month / day_of_week."""
    out = df.drop(columns=["Home ID", TARGET], errors="ignore").copy()
    out["hour"] = pd.to_datetime(out["Time"], format="%H:%M").dt.hour
    dates = pd.to_datetime(out["Date"], format="%Y-%m-%d")
    out["month"] = dates.dt.month
    out["day_of_week"] = dates.dt.dayofweek
    return out.drop(columns=["Time", "Date"])


def build_preprocessor() -> Pipeline:
    """Raw rows -> model-ready matrix, identical to the notebook's features.

    Numeric columns pass through; Appliance Type and Season are one-hot encoded with
    the first (alphabetical) category dropped, matching pd.get_dummies(drop_first=True).
    """
    encode = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(categories=[APPLIANCES, SEASONS], drop="first",
                                  sparse_output=False, dtype=int), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    pre = Pipeline([
        ("time_features", FunctionTransformer(add_time_features)),
        ("encode", encode),
    ])
    return pre.set_output(transform="pandas")
