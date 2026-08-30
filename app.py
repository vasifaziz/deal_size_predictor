import json
import streamlit as st
import joblib
import pandas as pd
from datetime import date
from pathlib import Path

# Import required functions so pickle can unpickle custom transformers
from dealsize_pipeline import (
    predict_new_data,
    add_date_features,
    build_date_transformer,
    build_preprocessor,
    INVERSE_LABEL_MAP,
    LABEL_MAP
)

# Load label mappings
Small = LABEL_MAP["Small"]
Medium = LABEL_MAP["Medium"]
Large = LABEL_MAP["Large"]

@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    import os

    # Try multiple path strategies
    paths_to_try = [
        Path(__file__).parent / "artifacts" / "dealsize_pipeline.pkl",
        Path("artifacts/dealsize_pipeline.pkl"),
        Path("./artifacts/dealsize_pipeline.pkl"),
    ]

    # Debug: print current working directory and file location
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Script location: {Path(__file__).parent}")

    for model_path in paths_to_try:
        st.write(f"Trying path: {model_path} (exists: {model_path.exists()})")
        if model_path.exists():
            try:
                return joblib.load(model_path)
            except Exception as e:
                st.error(f"Failed to load from {model_path}: {str(e)}")
                continue

    # If all fails, show what files are available
    st.error("Could not find model file. Available files:")
    for root, dirs, files in os.walk("."):
        for file in files:
            st.write(os.path.join(root, file))

    raise FileNotFoundError("Model file not found in any expected location")

model = load_model()


st.title('Deal Size Predictor')

st.write("'Predict Deal Size using Machine Learning! Classify Dealsize as multiclass from low to high value.'")


QUANTITYORDERED = st.slider("Quantity Ordered", min_value = 1, max_value = 97, value = 35)

PRICEEACH = st.number_input("Unit Price", min_value = 20, max_value = 300, value = 95)

DAYS_SINCE_LASTORDER = st.slider("Days Since Last Order", min_value = 42, max_value = 3562, value = 3184)

MSRP = st.slider("Manufacturing Selling Retail Price", min_value = 33, max_value = 214, value = 99)

ORDERLINENUMBER = st.slider("Order Line Number", min_value = 1, max_value = 18, value = 6)

PRODUCTLINE = st.selectbox("AutoMotive Type", ['Motorcycles', 'Classic Cars', 'Trucks and Buses', 'Vintage Cars', 'Planes', 'Ships', 'Trains'])

COUNTRY = st.selectbox("Country", ['USA', 'France', 'Norway', 'Australia', 'Finland', 'Austria', 'UK', 'Spain', 'Sweden', 'Singapore', 'Canada', 'Japan', 'Italy', 'Denmark', 'Belgium', 'Philippines', 'Germany', 'Switzerland', 'Ireland'])

ORDERDATE = st.date_input(
    "Order Date",
    value=date.today()
)

ORDERDATE = ORDERDATE.strftime("%Y-%m-%d")

input_data = pd.DataFrame({
	"QUANTITYORDERED": [QUANTITYORDERED],
	"PRICEEACH": [PRICEEACH],
	"DAYS_SINCE_LASTORDER": [DAYS_SINCE_LASTORDER],
	"MSRP": [MSRP],
	"ORDERLINENUMBER": [ORDERLINENUMBER], 
	"PRODUCTLINE": [PRODUCTLINE],
	"COUNTRY": [COUNTRY],
	"ORDERDATE": [ORDERDATE]
	})

st.subheader("Input Sent to the Pipeline")

st.dataframe(input_data)


if st.button("Predict"):
	prediction = model.predict(input_data)[0]

	probability = model.predict_proba(input_data)[0]

	st.subheader("Prediction Result")

	if prediction == Small:
		st.success("The Deal Size is Small")

		st.write(
			f"**Probability of Deal Size is Small: **"
			f"{probability[Small]:.2%}"
			)

	elif prediction == Medium:
		st.warning("The Deal Size is Medium")

		st.write(
			f"**Probability of Deal Size is Medium: **"
			f"{probability[Medium]:.2%}"
			)

	elif prediction == Large:
		st.error("The Deal Size is Large")

		st.write(
			f"**Probability of Deal Size is Large: **"
			f"{probability[Large]:.2%}"
			)
