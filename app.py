
import streamlit as st
import pandas as pd
import joblib
import os
import base64
from datetime import date

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Get project folder location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Background image function
def set_background(image_file):

    image_path = os.path.join(BASE_DIR, image_file)

    if os.path.exists(image_path):
        with open(image_path, "rb") as image:
            encoded_image = base64.b64encode(
                image.read()
            ).decode()

        background_css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        h1 {{
            color: #123B7A;
            font-weight: 800;
        }}

        .stButton > button {{
            background: linear-gradient(90deg, #1769D2, #39A0ED);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.65rem 1.2rem;
        }}
        </style>
        """

        st.markdown(background_css, unsafe_allow_html=True)

    else:
        st.warning("Background image not found.")

# Apply background
set_background("background.png")

# Application title
st.title("📊 Retail Sales Forecasting")

st.write("Predict retail sales using Machine Learning.")

# Correct model path (root folder)
model_path = os.path.join(BASE_DIR, "sales_model.pkl")

# Load trained model
if not os.path.exists(model_path):
    st.error("Model file not found. Please check sales_model.pkl in GitHub.")
    st.stop()

try:
    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    features = saved_data["features"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# Prediction inputs
st.subheader("Enter Prediction Details")

store = st.number_input(
    "Store ID",
    min_value=1,
    value=1,
    step=1
)

selected_date = st.date_input(
    "Prediction Date",
    value=date.today()
)

promo = st.selectbox(
    "Promotion",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

state_holiday = st.selectbox(
    "State Holiday",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "No Holiday",
        1: "Public Holiday",
        2: "Easter Holiday",
        3: "Christmas"
    }[x]
)

# Prepare input data
input_values = {
    "Store": store,
    "Year": selected_date.year,
    "Month": selected_date.month,
    "Day": selected_date.day,
    "DayOfWeek": selected_date.weekday(),
    "Promo": promo,
    "StateHoliday": state_holiday
}

input_df = pd.DataFrame([input_values])

input_df = input_df.reindex(
    columns=features,
    fill_value=0
)

# Predict sales
if st.button("Predict Sales"):

    try:
        prediction = model.predict(input_df)[0]

        st.success("Prediction completed!")

        st.metric(
            "Predicted Sales",
            f"{prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
