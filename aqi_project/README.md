# AQI Category Prediction (ML Mini Project)

Predicts the AQI category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) of an Indian city-day
from air-pollutant readings. Dataset: `city_day.csv` (Kaggle - Air Quality Data in India).

## Models compared
Logistic Regression, KNN, Random Forest, Gradient Boosting.
Best: **Gradient Boosting (~85.5% test accuracy)**. Full table in `results.csv`.

## Files
| File | Purpose |
|---|---|
| `AQI_Project.ipynb` | Main notebook: EDA, cleaning, features, 4 models, comparison, saving |
| `preprocess.py` | Cleaning + feature code (shared by notebook and app) |
| `app.py` | Streamlit web app |
| `models/` | Saved trained models |
| `figures/` | Charts used in the report |
| `results.csv` | Metrics table of all models |

## How to run
```
pip install -r requirements.txt
jupyter notebook AQI_Project.ipynb      # run all cells (re-creates models/, figures/, results.csv)
streamlit run app.py
```
