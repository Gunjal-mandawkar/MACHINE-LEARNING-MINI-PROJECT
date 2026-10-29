import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from preprocess import *

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

MODEL_NAMES = ["Gradient Boosting", "Random Forest", "KNN", "Logistic Regression"]


@st.cache_resource
def load_everything():
    raw = pd.read_csv("city_day.csv")
    df = prepare(raw)
    X, y = df[FEATURES], df["AQI_Bucket"]
    # SAME split as the notebook
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    info = joblib.load("models/preprocessing.pkl")
    models = {n: joblib.load("models/" + n.replace(" ", "_") + ".pkl") for n in MODEL_NAMES}
    meta = df.loc[X_test.index, ["City", "Date"]]
    return raw, X_test, y_test, meta, models, info


def predict(model_name, X_df):
    """Fill missing with training median, scale if the model needs it, then predict."""
    X_df = X_df.fillna(info["median"])
    data = info["scaler"].transform(X_df) if model_name in info["scaled_models"] else X_df
    return models[model_name].predict(data)


raw, X_test, y_test, meta, models, info = load_everything()
cities = sorted(raw["City"].unique())

st.title("🌫️ Air Quality (AQI) Category Predictor")
st.write("Predicts the AQI category of a city-day from pollutant readings.")

tab1, tab2, tab3 = st.tabs(["Test cases", "Enter your own values", "Model comparison"])

# ---------------- Tab 1: test cases ----------------
with tab1:
    st.subheader("Predictions on unseen test data")
    model_name = st.selectbox("Choose a model", MODEL_NAMES, key="m1")
    n = st.slider("Number of test cases", 5, 100, 20)
    seed = st.number_input("Seed (change to get different cases)", 0, 9999, 1)

    sample = X_test.sample(n, random_state=int(seed))
    table = meta.loc[sample.index].copy()
    table["Date"] = table["Date"].dt.date
    for c in ["PM2.5", "PM10", "NO2", "CO"]:
        table[c] = sample[c].round(1)
    table["Actual"] = y_test.loc[sample.index].values
    table["Predicted"] = predict(model_name, sample)
    table["Correct?"] = np.where(table["Actual"] == table["Predicted"], "Yes", "No")

    st.metric("Accuracy on these cases", f"{(table['Correct?'] == 'Yes').mean() * 100:.1f}%")
    st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)

# ---------------- Tab 2: manual input ----------------
with tab2:
    st.subheader("Enter pollutant values")
    model_name2 = st.selectbox("Choose a model", MODEL_NAMES, key="m2")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", cities)
    month = c2.selectbox("Month", list(range(1, 13)), index=0)

    st.write("**Today's readings**")
    defaults = {"PM2.5": 65.0, "PM10": 115.0, "NO": 15.0, "NO2": 35.0, "NOx": 30.0, "NH3": 20.0,
                "CO": 1.0, "SO2": 12.0, "O3": 40.0, "Benzene": 2.0, "Toluene": 5.0, "Xylene": 1.0}
    today, cols = {}, st.columns(4)
    for i, p in enumerate(POLLUTANTS):
        today[p] = cols[i % 4].number_input(p, 0.0, 2000.0, defaults[p], key="t" + p)

    st.write("**Yesterday's readings**")
    yest, cols = {}, st.columns(4)
    for i, p in enumerate(LAG_COLS):
        yest[p] = cols[i].number_input("Yesterday " + p, 0.0, 2000.0, defaults[p], key="y" + p)

    if st.button("Predict", type="primary"):
        row = dict(today)
        for p in LAG_COLS:
            row[p + "_prev"] = yest[p]
            row[p + "_avg3"] = (today[p] + yest[p]) / 2      # simple estimate of the 3-day average
        row["City_code"] = city_codes(raw)[city]
        row["Month"] = month
        result = predict(model_name2, pd.DataFrame([row])[FEATURES])[0]
        st.success(f"Predicted AQI category: **{result}**")

# ---------------- Tab 3: comparison ----------------
with tab3:
    st.subheader("Comparison of all models (test set)")
    st.dataframe(pd.read_csv("results.csv"), use_container_width=True, hide_index=True)
    st.image("figures/model_accuracy.png")
    st.image("figures/confusion_matrix_best.png", width=500)
