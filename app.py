import joblib
import pandas as pd
import streamlit as st

model = joblib.load("ad_click_model.joblib")

st.title("Ad Click Predictor")

site_time = st.number_input("Daily Time Spent on Site", min_value=0.0, value=65.0)
age = st.number_input("Age", min_value=18, max_value=100, value=36)
income = st.number_input("Area Income", min_value=0.0, value=55000.0)
internet_usage = st.number_input("Daily Internet Usage", min_value=0.0, value=180.0)
gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Daily Time Spent on Site": site_time,
        "Age": age,
        "Area Income": income,
        "Daily Internet Usage": internet_usage,
        "Male": 1 if gender == "Male" else 0,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]

    if prediction == 1:
        st.success("The model predicts this user will click the ad.")
    else:
        st.info("The model predicts this user will not click the ad.")

    st.write(f"Estimated click probability: {probability:.1%}")
