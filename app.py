import streamlit as st
import numpy as np
import joblib

# Load model and features
model = joblib.load(r"C:\Users\kanim\Desktop\Bengaluru_House_price\house_price_model.pkl")
model_features = joblib.load(r"C:\Users\kanim\Desktop\Bengaluru_House_price\model_features.pkl")

st.title("🏠Bengaluru House Price Prediction")

# Get user inputs
total_sqft = st.number_input("Total Square Feet", min_value=100, max_value=10000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)

# Location selectbox using location features from model
# Extract location names from features starting with 'location_'
locations = [feat.replace('location_', '') for feat in model_features if feat.startswith('location_')]
locations = sorted(locations)

location = st.selectbox("Location", locations)

# Prediction button
if st.button("Predict Price"):
    # Prepare input vector with zeros
    input_data = np.zeros(len(model_features))
    
    # Set numerical values
    try:
        sqft_index = model_features.index('total_sqft')
        input_data[sqft_index] = total_sqft
    except ValueError:
        st.error("Model feature 'total_sqft' not found.")
        st.stop()

    try:
        bath_index = model_features.index('bath')
        input_data[bath_index] = bath
    except ValueError:
        st.error("Model feature 'bath' not found.")
        st.stop()

    try:
        balcony_index = model_features.index('balcony')
        input_data[balcony_index] = balcony
    except ValueError:
        st.error("Model feature 'balcony' not found.")
        st.stop()

    # Set location one-hot
    location_feature = 'location_' + location.lower().strip()
    if location_feature in model_features:
        loc_index = model_features.index(location_feature)
        input_data[loc_index] = 1
    else:
        st.warning(f"Location '{location}' not found in model features. Using base prediction.")

    # Predict price
    prediction = model.predict([input_data])[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} lakhs")




