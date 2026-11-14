import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
data = pd.read_csv("dataset/train.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("Dataset columns:")
print(data.columns.tolist())

# 2. Check required columns
required_columns = ["Date", "Sales"]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"Required column '{column}' is missing from your dataset."
        )

# 3. Clean data
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")

# Remove rows with missing Date or Sales
data = data.dropna(subset=["Date", "Sales"])

# If Open column exists, use only open stores
if "Open" in data.columns:
    data = data[data["Open"] == 1]

# 4. Create date features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek

# 5. Convert StateHoliday only if the column exists
if "StateHoliday" in data.columns:
    data["StateHoliday"] = data["StateHoliday"].astype(str)

    data["StateHoliday"] = data["StateHoliday"].map({
        "0": 0,
        "a": 1,
        "b": 2,
        "c": 3
    }).fillna(0)

# 6. Select features that exist in your dataset
possible_features = [
    "Store",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Promo",
    "StateHoliday"
]

features = [
    column for column in possible_features
    if column in data.columns
]

print("Features used for training:")
print(features)

if not features:
    raise ValueError("No usable features were found in the dataset.")

# 7. Sort data by date
data = data.sort_values("Date")

# 8. Split data chronologically
split_date = data["Date"].quantile(0.8)

train_data = data[data["Date"] <= split_date]
test_data = data[data["Date"] > split_date]

X_train = train_data[features]
y_train = train_data["Sales"]

X_test = test_data[features]
y_test = test_data["Sales"]

# 9. Create machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# 10. Train model
print("Training the model...")
model.fit(X_train, y_train)

# 11. Make predictions
predictions = model.predict(X_test)

# 12. Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("Model training completed!")
print("Mean Absolute Error:", round(mae, 2))

# 13. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "models/sales_model.pkl"
)

print("Model saved successfully!")