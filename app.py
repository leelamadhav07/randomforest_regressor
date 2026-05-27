import streamlit as st
import pickle
import numpy as np

# LOAD MODEL

with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# PAGE CONFIG

st.set_page_config(page_title="California Housing Prediction", layout="centered")

st.title("California Housing Price Prediction")

st.write("Random Forest Regressor")

# SHOW HYPERPARAMETERS

st.subheader("Model Hyperparameters")

st.write("Number of Trees: 100")

st.write("Max Depth: 10")

st.write("Min Samples Split: 5")

st.write("Min Samples Leaf: 2")

# USER INPUTS

MedInc = st.slider("Median Income", 0.0, 20.0, 5.0)

HouseAge = st.slider("House Age", 1.0, 60.0, 20.0)

AveRooms = st.slider("Average Rooms", 1.0, 20.0, 5.0)

AveBedrms = st.slider("Average Bedrooms", 1.0, 10.0, 2.0)

Population = st.slider("Population", 1.0, 50000.0, 1000.0)

AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)

Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)

Longitude = st.slider("Longitude", -125.0, -113.0, -120.0)

# PREDICTION BUTTON

if st.button("Predict House Price"):
    input_data = np.array(
        [
            [
                MedInc,
                HouseAge,
                AveRooms,
                AveBedrms,
                Population,
                AveOccup,
                Latitude,
                Longitude,
            ]
        ]
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ${prediction[0] * 100000:,.2f}")
