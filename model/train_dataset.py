import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # For saving the model

print("Loading dataset...")
df = pd.read_csv("flight_dataset.csv")

# --- 1. FEATURE ENGINEERING ---
# We need to convert text (like 'Airline') into numbers for the model.
# We also extract 'Hour of Day' from 'Departure Time'.

# Drop rows with missing values for simplicity
df = df.dropna()

# Convert Departure Time to just the Hour
df['Departure Time'] = pd.to_datetime(df['Departure Time']).dt.hour

# --- 2. DEFINE FEATURES (X) and TARGET (y) ---
# This is what we want to predict
target = "Total Price"

# These are the features we'll use to predict the price
features = [
    "Days Until Departure",
    "Airline",
    "Departure Airport",
    "Arrival Airport",
    "Departure Time", # This is now the hour
    "Cabin"
]

X = df[features]
y = df[target]

# --- 3. CREATE A PREPROCESSING PIPELINE ---
# This pipeline will automatically handle our text features
# 'categorical_features' will be One-Hot Encoded
categorical_features = ["Airline", "Departure Airport", "Arrival Airport", "Cabin"]
numeric_features = ["Days Until Departure", "Departure Time"]

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 4. CREATE AND TRAIN THE MODEL ---
# Now we build the full pipeline: 1. Preprocess data, 2. Train Random Forest
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting model training (this may take a minute)...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. EVALUATE THE MODEL ---
print("\n--- Model Evaluation ---")
preds = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"R-squared (R²): {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} INR")
print(f"(This means your model's price predictions are off by ~{mae:.2f} INR on average)")

# --- 6. SAVE THE MODEL ---
model_filename = "flight_price_model.joblib"
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved successfully as {model_filename}")

# --- 7. HOW TO USE IT (Example) ---
print("\n--- Example Prediction ---")
# Let's create a new, unseen flight
example_flight = pd.DataFrame({
    "Days Until Departure": [30],
    "Airline": ["AI"],
    "Departure Airport": ["DEL"],
    "Arrival Airport": ["MAA"],
    "Departure Time": [17], # 5 PM
    "Cabin": ["ECONOMY"]
})

predicted_price = model_pipeline.predict(example_flight)[0]
print(f"Model's predicted 'fair price' for this flight: {predicted_price:.2f} INR")