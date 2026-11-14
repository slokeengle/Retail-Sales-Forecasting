import streamlit as st
import pandas as pd
import joblib
import os
import base64
from datetime import date

# Must be the first Streamlit command
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Background image function
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

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


# Apply background BEFORE displaying app content
set_background("background.png")

# App content
st.title("📊 Retail Sales Forecasting")
st.write("Predict retail sales using Machine Learning.")

model_path = "models/sales_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

saved_data = joblib.load(model_path)
model = saved_data["model"]
features = saved_data["features"]

st.subheader("Enter Prediction Details")

store = st.number_input("Store ID", min_value=1, value=1, step=1)
selected_date = st.date_input("Prediction Date", value=date.today())

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
input_df = input_df.reindex(columns=features, fill_value=0)

if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success("Prediction completed!")
    st.metric("Predicted Sales", f"{prediction:,.2f}")