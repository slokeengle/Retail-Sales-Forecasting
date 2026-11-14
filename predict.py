
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/sales_model.pkl")

# Input date
date = "2025-01-21"

# Convert date
date = pd.to_datetime(date)

# Prepare input data
X = data[
    ["Store", "Year", "Month", "Day",
     "DayOfWeek", "Promo", "StateHoliday"]
]

y = data["Sales"]
# Predict sales
prediction = model.predict(input_data)

print("Predicted Sales:", prediction[0])
