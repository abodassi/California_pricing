import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("xgb3_model.pkl")

st.title("🏠 California Housing Price Predictor")

st.markdown("Adjust the sliders and select location to predict housing prices:")

# Input sliders based on your stats
longitude = st.slider("longitude", -124.35, -114.31, -124.35)
latitude = st.slider("latitude", 32.54, 41.95, 32.54)
housing_median_age = st.slider("Housing Median Age", 1, 52, 1)
total_rooms = st.slider("Total Rooms", 2, 39320, 2)
total_bedrooms = st.slider("Total Bedrooms", 1, 6445, 1)
population = st.slider("Population", 3, 35682, 3)
households = st.slider("Households", 1, 6082, 1)
median_income = st.slider("Median Income", 0.5, 15.0, .5)

# One-hot encoded location selection
location = st.radio("Proximity to Ocean", ("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"))

# One-hot encode the selected location
location_cols = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
location_vals = [1 if loc == location else 0 for loc in location_cols]

# Assemble input
input_df = pd.DataFrame([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income,
    *location_vals
]], columns=[
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
])

# Predict and display
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.session_state.prediction_result = int(prediction)

# Display the prediction if it exists
if "prediction_result" in st.session_state:
    st.success(f"🏡 Predicted Median House Value: **${st.session_state.prediction_result:,}**")