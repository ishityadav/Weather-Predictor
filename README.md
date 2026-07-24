# 🌤️ Weather-Predictor

A machine learning web app that predicts **temperature, humidity, and pressure** for Indian cities based on the date, time, state, and city — built with XGBoost and deployed with Streamlit.

**🔗 Live app:** https://weather-predictor-ishiyadav.streamlit.app/

---

## Overview

Weather-Predictor is a multi-output regression project trained on historical Indian weather data. Given a **state**, **city**, **date**, and **hour of day**, the model predicts three weather variables simultaneously:

- 🌡️ Temperature (°C)
- 💧 Humidity (%)
- 🌬️ Pressure (hPa)

The trained model is wrapped in a simple Streamlit UI where a user picks a state, then a city (filtered by state), a date, and an hour — and gets an instant prediction.

## How it works

1. **Data preparation** (`weather_main.ipynb`)
   - Raw weather records (`city_name`, `dt_txt`, `temp`, `pressure`, `humidity`, `state`) are loaded.
   - The timestamp column `dt_txt` is parsed and decomposed into `hour`, `day`, and `month` features.
   - `city_name` and `state` are label-encoded into `city_enc` and `state_enc` using `LabelEncoder`.

2. **Modeling**
   - Features used: `hour`, `day`, `month`, `city_enc`, `state_enc`
   - Targets: `temp`, `humidity`, `pressure` (predicted jointly)
   - Model: `XGBRegressor` wrapped in a `MultiOutputRegressor` from scikit-learn, trained on an 80/20 train-test split.

3. **Evaluation** (on the held-out test set)

   | Target      | MAE  | RMSE | R²   |
   |-------------|------|------|------|
   | Temperature | 1.36 | 1.82 | 0.94 |
   | Humidity    | 7.33 | 9.86 | 0.81 |
   | Pressure    | 0.89 | 1.19 | 0.90 |

4. **Serialization**
   - The trained model and encoders are saved as pickle files: `weather_multi_model.pkl`, `le_city.pkl`, `le_state.pkl`.
   - A `state_city_map.pkl` dictionary (state → list of cities) is also saved so the app's dropdowns can be populated without needing the raw dataset at inference time.

5. **App** (`deployapp.py`)
   - A Streamlit interface lets the user pick a state, city, day, month, and hour.
   - The categorical inputs are encoded with the saved `LabelEncoder`s, combined with the date/time fields, and passed to the model.
   - The app displays the predicted temperature, humidity, and pressure.

## Project structure

```
Weather-Predictor/
├── deployapp.py              # Streamlit app (inference/UI)
├── weather_main.ipynb         # Data cleaning, training, and evaluation notebook
├── weather_multi_model.pkl    # Trained MultiOutputRegressor (XGBoost) model
├── le_city.pkl                # Fitted LabelEncoder for city names
├── le_state.pkl                # Fitted LabelEncoder for states
├── state_city_map.pkl          # Dict mapping each state to its cities
├── requirements.txt           # Python dependencies
└── .devcontainer/              # Dev container config (for GitHub Codespaces / VS Code)
```

## Tech stack

- **Python**
- **XGBoost** — multi-output gradient-boosted regression model
- **scikit-learn** — `MultiOutputRegressor`, `LabelEncoder`, train/test split, metrics
- **pandas / numpy** — data processing
- **Streamlit** — web app / UI

## Getting started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/ishityadav/Weather-Predictor.git
cd Weather-Predictor
pip install -r requirements.txt
```

### Run the app locally

```bash
streamlit run deployapp.py
```

The app will open in your browser (default: `http://localhost:8501`). Select a state, city, date, and hour to get a live weather prediction.

### Retrain the model (optional)

Open `weather_main.ipynb` in Jupyter to explore the data cleaning steps, retrain the model on your own dataset, and regenerate the `.pkl` files.

## Notes

- The city dropdown is dynamically filtered based on the selected state using `state_city_map.pkl`, so users can only pick valid state/city combinations.
- Predictions are only meaningful for cities/states seen during training, since encoding is done with the fitted `LabelEncoder`s.

## License

No license specified. Feel free to open an issue if you'd like this clarified.
