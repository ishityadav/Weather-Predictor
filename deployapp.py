import streamlit as st
import pickle
import numpy as np

# Load model & encoders
model = pickle.load(open("weather_multi_model.pkl", "rb"))
le_city = pickle.load(open("le_city.pkl", "rb"))
le_state = pickle.load(open("le_state.pkl", "rb"))
state_city_map = pickle.load(open("state_city_map.pkl", "rb"))

st.title("🌤 Weather Prediction (Temp, Humidity, Pressure)")

# First select state
state = st.selectbox("Select State", options=sorted(state_city_map.keys()))

# Then filter cities based on state
city = st.selectbox("Select City", options=sorted(state_city_map[state]))

# Date selectors
day = st.selectbox("Day", list(range(1, 32)))
month = st.selectbox("Month", 
                     ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"])
month_num = list(range(1,13))[ ["January","February","March","April","May","June",
                                "July","August","September","October","November","December"].index(month) ]

hour = st.selectbox("Hour of Day", list(range(0,24)))

# Encode categorical inputs
city_enc = le_city.transform([city])[0]
state_enc = le_state.transform([state])[0]

# Prepare input
X_input = np.array([[hour, day, month_num, city_enc, state_enc]])

# Predict
pred = model.predict(X_input)[0]  # [temp, humidity, pressure]

st.success(f"🌡 Temperature: {pred[0]:.2f} °C")
st.info(f"💧 Humidity: {pred[1]:.2f} %")
st.warning(f"🌬 Pressure: {pred[2]:.2f} hPa")
