import streamlit as st
import pickle
import joblib
import numpy as np

# -------------------------------
# Load Model, Features & Scaler
# -------------------------------

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("top10features.pkl", "rb") as f:
    top_features = pickle.load(f)

scaler = joblib.load("num_scaler.pkl")

# -------------------------------
# App Config & Title
# -------------------------------
st.set_page_config(page_title="NYC Fare Estimator", layout="centered")
st.title("🚕 NYC Green Taxi Fare Estimator")

# Optional: Summary Section
with st.expander("About this project"):
    st.markdown("""
    This tool estimates the total fare for NYC green taxi trips using a February 2024 data to build the machine learning model. I trained on cleaned and feature-engineered trip data. 
    It focuses on the top 10 features most relevant to predicting fare amount, with preprocessing handled by a trained scaler.

    Just fill in the trip details below to get a real-time prediction.
    """)

# -------------------------------
# Input Fields
# -------------------------------
st.header("Enter Trip Details")

zone_range = list(range(1, 264))  # TLC zones
ratecodes = [1, 2, 3, 4, 5, 6]
hours = list(range(0, 24))
weekdays = list(range(0, 7))

user_inputs = {}

col1, col2 = st.columns(2)

with col1:
    user_inputs["trip_distance"] = st.number_input("Trip Distance (miles)", min_value=0.0, value=3.2)
    user_inputs["tip_amount"] = st.number_input("Tip Amount ($)", min_value=0.0, value=1.5)
    user_inputs["mta_tax"] = st.number_input("MTA Tax ($)", min_value=0.0, value=0.5, step=0.1)
    user_inputs["tolls_amount"] = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0)
    user_inputs["RatecodeID"] = st.selectbox("Ratecode ID", ratecodes)

with col2:
    user_inputs["trip_duration"] = st.number_input("Trip Duration (minutes)", min_value=0.0, value=12.0)
    user_inputs["passenger_count"] = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    user_inputs["hour"] = st.selectbox("Pickup Hour", hours)
    user_inputs["weekday"] = st.selectbox("Day of Week (0 = Monday)", weekdays)

col3, col4 = st.columns(2)
with col3:
    user_inputs["PUZone"] = st.selectbox("Pickup Zone ID", zone_range)
with col4:
    user_inputs["DOZone"] = st.selectbox("Drop-off Zone ID", zone_range)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Estimate Fare"):
    try:
        input_ordered = np.array([user_inputs[feat] for feat in top_features]).reshape(1, -1)
        scaled_input = scaler.transform(input_ordered)
        prediction = model.predict(scaled_input)[0]
        st.success(f"Estimated Total Fare: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr style='margin-top: 40px;'>
<p style='text-align: center; font-size: 0.85em; color: gray;'>
    Built from scratch with custom preprocessing and model training.
</p>
""", unsafe_allow_html=True)
