"""Data cleaning + feature creation. Used by BOTH the notebook and the Streamlit app."""
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
LAG_COLS = ["PM2.5", "PM10", "NO2", "CO"]          # pollutants for which we add "yesterday" + "3-day average"
CLASS_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
FEATURES = (POLLUTANTS
            + [c + s for c in LAG_COLS for s in ("_prev", "_avg3")]
            + ["City_code", "Month"])


def clean_data(raw):
    """Remove duplicates, rows with no target, and sort by city and date."""
    df = raw.drop_duplicates()
    df = df.dropna(subset=["AQI_Bucket"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def fill_missing(df):
    """Fill gaps using neighbouring days of the SAME city (linear interpolation, up to 7 days)."""
    for col in POLLUTANTS:
        df[col] = df.groupby("City")[col].transform(
            lambda s: s.interpolate(limit=7, limit_direction="both"))
    return df


def add_features(df):
    """Yesterday's value and 3-day average (past days only), plus city code and month."""
    for col in LAG_COLS:
        g = df.groupby("City")[col]
        df[col + "_prev"] = g.shift(1)
        df[col + "_avg3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["City_code"] = df["City"].map(city_codes(df))
    df["Month"] = df["Date"].dt.month
    return df


def city_codes(df):
    return {c: i for i, c in enumerate(sorted(df["City"].unique()))}


def prepare(raw):
    return add_features(fill_missing(clean_data(raw)))
