"""Train the four final regression models with the settings tuned in
notebooks/energy_model.ipynb and save them to models/.

Run from the project root:
    .venv/bin/python -m src.train
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so the pickled preprocessor references src.features

from src.features import TARGET, build_preprocessor  # noqa: E402

RANDOM_STATE = 42
DATA_PATH = ROOT / "data" / "smart_home_energy_consumption_large.csv"
MODEL_DIR = ROOT / "models"

# Best hyperparameters found by GridSearchCV in the notebook (sections 8 and 9)
DT_PARAMS = {"max_depth": 12, "min_samples_leaf": 200}
RF_PARAMS = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20}

MODELS = {
    "Linear Regression": ("linear_regression.joblib", Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression())])),
    "Polynomial Regression (deg 2)": ("polynomial_regression.joblib", Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", LinearRegression())])),
    "Decision Tree": ("decision_tree.joblib",
                      DecisionTreeRegressor(random_state=RANDOM_STATE, **DT_PARAMS)),
    "Random Forest": ("random_forest.joblib",
                      RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **RF_PARAMS)),
}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    raw_train, raw_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=RANDOM_STATE)

    pre = build_preprocessor()
    X_train = pre.fit_transform(raw_train)
    X_test = pre.transform(raw_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
    print(f"Saved models/preprocessor.joblib  ({X_train.shape[1]} features)")

    metadata = {"feature_names": list(X_train.columns), "random_state": RANDOM_STATE,
                "train_rows": len(X_train), "test_rows": len(X_test), "models": {}}
    for name, (filename, model) in MODELS.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        info = {
            "file": filename,
            "MAE": mean_absolute_error(y_test, pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_test, pred),
            "fit_seconds": round(time.time() - t0, 2),
        }
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = dict(zip(X_train.columns, model.feature_importances_.tolist()))
        metadata["models"][name] = info
        joblib.dump(model, MODEL_DIR / filename, compress=3)
        print(f"Saved models/{filename:<28} test R² = {info['R2']:.4f}  RMSE = {info['RMSE']:.4f}")

    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Saved models/metadata.json")


if __name__ == "__main__":
    main()
