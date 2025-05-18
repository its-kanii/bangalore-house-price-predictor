import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load your data (replace with your CSV path)
df = pd.read_csv(r"C:\Users\kanim\Desktop\Bengaluru_House_price\Bengaluru_House_Data.csv")

# Strip whitespace from string columns
str_cols = ['area_type', 'availability', 'location', 'size', 'society']
for col in str_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Handle missing locations by filling with 'unknown'
df['location'] = df['location'].fillna('unknown')

# Convert numeric columns to proper types if needed
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Drop rows with missing numeric values (or handle as you want)
df.dropna(subset=['total_sqft', 'bath', 'balcony', 'price'], inplace=True)

# One-hot encode location column
location_dummies = pd.get_dummies(df['location'], prefix='location')

# Prepare final dataframe for model
X = pd.concat([df[['total_sqft', 'bath', 'balcony']], location_dummies], axis=1)
y = df['price']

# Save model features to use in app
model_features = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(model_features, 'model_features.pkl')

print("Model training completed and saved.")

