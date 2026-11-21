"""Smart Home Energy Predictor — Streamlit app.

Run from the project root:
    .venv/bin/streamlit run app.py
Models are created by:  .venv/bin/python -m src.train
"""
import datetime as dt
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import APPLIANCES, SEASONS, TARGET, season_from_month

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "smart_home_energy_consumption_large.csv"
PRICE_PER_KWH = 8.0  # ₹ per kWh

st.set_page_config(page_title="Smart Home Energy Predictor", page_icon="⚡", layout="wide")


# ---------- Loading (cached) ----------
@st.cache_resource
def load_artifacts():
    if not (MODEL_DIR / "metadata.json").exists():
        return None, None, None
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    models = {name: joblib.load(MODEL_DIR / info["file"]) for name, info in meta["models"].items()}
    return pre, models, meta


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


pre, models, meta = load_artifacts()
if models is None:
    st.error("No trained models found in `models/`. Run `.venv/bin/python -m src.train` first.")
    st.stop()


def show_plot(filename, caption=None):
    path = OUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"`outputs/{filename}` not found — run `notebooks/energy_model.ipynb` to generate it.")


def pretty(col):
    return col.replace("Appliance Type_", "Appliance: ").replace("Season_", "Season: ")


st.title("⚡ Smart Home Energy Predictor")
st.caption("Predict appliance energy use from the Kaggle Smart Home Energy Consumption dataset "
           "(100,000 readings, 2023). Four regression models trained on 80% of the data.")

tab_predict, tab_compare, tab_insights = st.tabs(["🔮 Predict", "📊 Model comparison", "💡 Insights"])

# ---------- Tab 1: Predict ----------
with tab_predict:
    DEFAULT_DATE = dt.date(2023, 7, 15)
    if "season" not in st.session_state:
        st.session_state.season = season_from_month(DEFAULT_DATE.month)

    def sync_season():
        st.session_state.season = season_from_month(st.session_state.date.month)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Inputs")
        c1, c2 = st.columns(2)
        with c1:
            appliance = st.selectbox("Appliance", APPLIANCES, index=APPLIANCES.index("Heater"))
            date = st.date_input("Date", value=DEFAULT_DATE, key="date", on_change=sync_season)
            season = st.selectbox("Season", SEASONS, key="season",
                                  help="Auto-filled from the date (Winter Jan–Mar, Spring Apr–Jun, "
                                       "Summer Jul–Sep, Fall Oct–Dec). You can override it.")
        with c2:
            time_ = st.slider("Time of day", value=dt.time(18, 0), step=dt.timedelta(minutes=15),
                              format="HH:mm")
            temperature = st.slider("Outdoor temperature (°C)", -10.0, 40.0, 25.0, step=0.5)
            household = st.slider("Household size (people)", 1, 5, 3)
        model_name = st.selectbox("Model", list(models), index=list(models).index("Random Forest"))

    row = pd.DataFrame([{
        "Appliance Type": appliance,
        "Time": time_.strftime("%H:%M"),
        "Date": date.strftime("%Y-%m-%d"),
        "Outdoor Temperature (°C)": temperature,
        "Season": season,
        "Household Size": household,
    }])
    X = pre.transform(row)
    preds = {name: max(float(m.predict(X)[0]), 0.0) for name, m in models.items()}
    kwh = preds[model_name]

    with right:
        st.subheader("Prediction")
        st.metric("Predicted energy (kWh)", f"{kwh:.2f}", border=True)
        st.metric("Estimated cost (₹8/kWh)", f"₹{kwh * PRICE_PER_KWH:.2f}", border=True,
                  help=f"Predicted kWh × ₹{PRICE_PER_KWH:.0f}/kWh")
        r2 = meta["models"][model_name]["R2"]
        rmse = meta["models"][model_name]["RMSE"]
        st.caption(f"{model_name}: test R² = {r2:.3f}, typical error (RMSE) ≈ ±{rmse:.2f} kWh "
                   f"(≈ ±₹{rmse * PRICE_PER_KWH:.2f}).")

        st.markdown("**All models for these inputs**")
        st.dataframe(
            pd.DataFrame({"Model": list(preds),
                          "Predicted kWh": [round(v, 3) for v in preds.values()],
                          "Cost (₹)": [round(v * PRICE_PER_KWH, 2) for v in preds.values()]}),
            hide_index=True,
        )

        data = load_data()
        appl = data.loc[data["Appliance Type"] == appliance, TARGET]
        st.caption(f"In the dataset, **{appliance}** readings range "
                   f"{appl.min():.2f}–{appl.max():.2f} kWh (mean {appl.mean():.2f}).")

# ---------- Tab 2: Model comparison ----------
with tab_compare:
    st.subheader("Test-set metrics (20% hold-out, 20,000 readings)")
    metrics_path = OUT_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).set_index("Model")
        st.dataframe(
            metrics.style.format({c: "{:.4f}" for c in ["MAE", "MSE", "RMSE", "R²", "Train R²"]}
                                 | {"Fit time (s)": "{:.1f}"})
                   .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#1baf7a55")
                   .highlight_max(subset=["R²"], color="#1baf7a55"),
        )
        best = metrics["R²"].idxmax()
        spread = metrics["R²"].max() - metrics["R²"].min()
        st.markdown(f"**{best}** scores best on R², but all four models are within "
                    f"**{spread:.4f}** R² of each other: in practice they perform the same. "
                    "Lower is better for MAE, MSE and RMSE; higher is better for R².")
    else:
        st.info("`outputs/model_metrics.csv` not found — run the notebook to generate it.")

    show_plot("05_model_comparison.png")
    c1, c2 = st.columns(2)
    with c1:
        show_plot("06_actual_vs_predicted.png", "Actual vs predicted: predictions form one band per appliance group")
        show_plot("08_residual_distribution.png", "Residual distribution")
    with c2:
        show_plot("07_residuals_vs_predicted.png", "Residuals vs predicted")
        show_plot("04_decision_tree_tuning.png", "Decision Tree GridSearchCV: depth vs CV error")

# ---------- Tab 3: Insights ----------
with tab_insights:
    data = load_data()
    appliance_means = data.groupby("Appliance Type")[TARGET].mean()
    explained = data["Appliance Type"].map(appliance_means)
    share = 1 - ((data[TARGET] - explained) ** 2).sum() / ((data[TARGET] - data[TARGET].mean()) ** 2).sum()

    st.subheader("Key finding")
    st.success(f"**Appliance type alone explains ~75% ({share:.1%}) of the variation in energy use.** "
               "Predicting each appliance's average consumption already gives R² ≈ "
               f"{share:.3f}, and none of the four models does meaningfully better.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Explained by appliance", f"{share:.1%}", border=True)
    k2.metric("Best model R²", f"{max(m['R2'] for m in meta['models'].values()):.3f}", border=True)
    k3.metric("Unexplained noise", f"{1 - share:.1%}", border=True)

    st.markdown("""
- **Heaters and air conditioners** use 2–5 kWh per reading; **fridges** only 0.1–0.5 kWh; everything else 0.2–2 kWh.
- **Temperature, season, household size and time of day have almost no effect**: their correlations with energy use are ≈ 0, and the models give them near-zero importance.
- Within each appliance, consumption is spread almost uniformly at random, which suggests the dataset is **synthetic**. The remaining ~25% of variation can't be predicted from these features. Better predictions would need richer data such as appliance power rating, usage duration or real smart-meter readings.
""")

    st.markdown("**Feature importance (Random Forest, top 10)**")
    imp = (pd.Series(meta["models"]["Random Forest"]["feature_importances"])
             .rename(index=pretty).sort_values(ascending=False).head(10))
    st.bar_chart(imp.rename("Importance"), horizontal=True, sort="-Importance", color="#6250d6")
    st.caption("Air Conditioning and Fall don't appear because they are the baseline "
               "categories dropped during one-hot encoding.")
    show_plot("02_energy_by_appliance.png", "Energy consumption by appliance type")

    with st.expander("More plots: both tree models' importance, linear coefficients, correlations"):
        show_plot("09_feature_importance.png")
        show_plot("10_linear_coefficients.png")
        show_plot("03_correlation_heatmap.png")
