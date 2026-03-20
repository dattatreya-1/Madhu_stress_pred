import streamlit as st
import pickle
import numpy as np

# ===== LOAD MODEL =====
model = pickle.load(open("best_xgb_model.pkl", "rb"))

st.title("🧠 Human Stress Detection System")

st.write("Enter psychological, sleep and mobile usage details")

st.subheader("Personality Features")

openness = st.number_input("Openness", 0.0, 10.0)
conscientiousness = st.number_input("Conscientiousness", 0.0, 10.0)
extraversion = st.number_input("Extraversion", 0.0, 10.0)
agreeableness = st.number_input("Agreeableness", 0.0, 10.0)
neuroticism = st.number_input("Neuroticism", 0.0, 10.0)

st.subheader("Sleep Features")

sleep_time = st.number_input("Sleep Time")
wake_time = st.number_input("Wake Time")
sleep_duration = st.number_input("Sleep Duration")
psqi = st.number_input("PSQI Score")

st.subheader("Mobile Usage")

call_duration = st.number_input("Call Duration")
num_calls = st.number_input("Number of Calls")
num_sms = st.number_input("Number of SMS")
screen_on = st.number_input("Screen On Duration")

st.subheader("Sensor Features")

skin = st.number_input("Skin Conductance")
accelerometer = st.number_input("Accelerometer")
mobility_radius = st.number_input("Mobility Radius")
mobility_distance = st.number_input("Mobility Distance")

# ===== PREDICTION BUTTON =====

if st.button("Predict Stress"):

    features = np.array([[

        openness,
        conscientiousness,
        extraversion,
        agreeableness,
        neuroticism,
        sleep_time,
        wake_time,
        sleep_duration,
        psqi,
        call_duration,
        num_calls,
        num_sms,
        screen_on,
        skin,
        accelerometer,
        mobility_radius,
        mobility_distance

    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Person is UNDER STRESS\n\nProbability = {probability:.2f}")
    else:
        st.success(f"✅ Person is NOT UNDER STRESS\n\nProbability = {probability:.2f}")
