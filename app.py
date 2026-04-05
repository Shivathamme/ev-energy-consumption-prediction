import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(BASE_DIR, "energy_pipeline.pkl"))

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Electric Bus Energy Consumption Predictor",
    layout="centered"
)

st.title("🚍 Electric Bus Energy Consumption Predictor")
st.write(
    "Predict **energy consumption (kWh)** using vehicle, driving, and environmental conditions."
)

st.divider()

# --------------------------------------------------
# Vehicle & Driving Inputs
# --------------------------------------------------
st.subheader("🔧 Vehicle & Driving Inputs")

vehicle_id = st.number_input("Vehicle ID", min_value=0, value=1001)

speed = st.number_input("Speed (km/h)", min_value=0.0, value=40.0)
acceleration = st.number_input("Acceleration (m/s²)", value=0.5)

battery_state = st.number_input("Battery State (%)", 0.0, 100.0, value=80.0)
battery_voltage = st.number_input("Battery Voltage (V)", value=380.0)
battery_temp = st.number_input("Battery Temperature (°C)", value=30.0)

slope = st.number_input("Road Slope (%)", value=1.0)

# --------------------------------------------------
# Environment & Load Inputs
# --------------------------------------------------
st.subheader("🌦️ Environment & Load")

temperature = st.number_input("Ambient Temperature (°C)", value=28.0)
humidity = st.number_input("Humidity (%)", value=60.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)

tire_pressure = st.number_input("Tire Pressure (psi)", value=34.0)
vehicle_weight = st.number_input("Vehicle Weight (kg)", value=13000.0)

distance = st.number_input("Distance Travelled (km)", min_value=0.0, value=12.0)

# --------------------------------------------------
# Categorical Inputs
# --------------------------------------------------
st.subheader("🛣️ Driving Conditions")

driving_mode = st.selectbox(
    "Driving Mode",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Eco", 2: "Normal", 3: "Sport"}[x]
)

road_type = st.selectbox(
    "Road Type",
    options=[1, 2, 3],
    format_func=lambda x: {1: "City", 2: "Highway", 3: "Hilly"}[x]
)

traffic = st.selectbox(
    "Traffic Condition",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x]
)

weather = st.selectbox(
    "Weather Condition",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear",
        2: "Rainy",
        3: "Foggy",
        4: "Windy"
    }[x]
)

# --------------------------------------------------
# Prepare input DataFrame (MUST MATCH TRAINING FEATURES)
# --------------------------------------------------
input_df = pd.DataFrame([{
    "Vehicle_ID": vehicle_id,
    "Speed_kmh": speed,
    "Acceleration_ms2": acceleration,
    "Battery_State_%": battery_state,
    "Battery_Voltage_V": battery_voltage,
    "Battery_Temperature_C": battery_temp,
    "Slope_%": slope,
    "Temperature_C": temperature,
    "Humidity_%": humidity,
    "Wind_Speed_ms": wind_speed,
    "Tire_Pressure_psi": tire_pressure,
    "Vehicle_Weight_kg": vehicle_weight,
    "Distance_Travelled_km": distance,
    "Driving_Mode": driving_mode,
    "Road_Type": road_type,
    "Traffic_Condition": traffic,
    "Weather_Condition": weather
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("🔮 Predict Energy Consumption"):
    prediction = pipeline.predict(input_df)
    st.success(f"⚡ Estimated Energy Consumption: **{prediction[0]:.2f} kWh**")

    st.caption(
        "Prediction generated using a machine learning pipeline "
        "trained on electric bus operational data."
    )
