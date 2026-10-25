
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    saved = joblib.load("model.pkl")
    return saved

saved = load_model()
model = saved["model"]
features = saved["features"]

st.title("🏠 California House Price Prediction")
st.write(
    "Enter the house and area details below "
    "to estimate the median house value."
)

st.info(
    "This model was trained on the California Housing dataset. "
    "Predictions are estimates, not guaranteed market prices."
)

with st.form("house_details"):
    st.subheader("Enter House Details")

    med_inc = st.number_input(
        "Median Income (in $10,000 units)",
        min_value=0.0, max_value=20.0,
        value=5.0, step=0.1
    )

    house_age = st.number_input(
        "Median House Age (years)",
        min_value=1.0, max_value=100.0,
        value=20.0, step=1.0
    )

    ave_rooms = st.number_input(
        "Average Rooms",
        min_value=1.0, max_value=50.0,
        value=6.0, step=0.5
    )

    ave_bedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.1, max_value=10.0,
        value=1.0, step=0.1
    )

    population = st.number_input(
        "Area Population",
        min_value=1.0, max_value=50000.0,
        value=1500.0, step=100.0
    )

    ave_occup = st.number_input(
        "Average Occupancy",
        min_value=0.1, max_value=100.0,
        value=3.0, step=0.1
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.0, max_value=42.0,
        value=34.0, step=0.1
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-125.0, max_value=-114.0,
        value=-118.3, step=0.1
    )

    submitted = st.form_submit_button(
        "Predict House Price"
    )

if submitted:
    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    input_data = input_data[features]
    prediction = float(model.predict(input_data)[0])
    price = prediction * 100000

    st.subheader("Prediction Result")

    if price < 0:
        st.warning(
            "The model produced a negative estimate. "
            "Please try different input values."
        )
    else:
        st.success(
            f"Estimated Median House Value: ${price:,.0f}"
        )

    st.caption(
        "The model predicts median value in units of "
        "$100,000. Actual prices may differ."
    )

