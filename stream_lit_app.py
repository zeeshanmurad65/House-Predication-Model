# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the Saved Assets ---
# This is a robust way to load the files, with error handling.
try:
    model = joblib.load('final_house_price_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    address_list = joblib.load('address_list.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'final_house_price_model.pkl', 'model_columns.pkl', and 'address_list.pkl' are in the same folder as app.py.")
    st.stop() # Stop the app if files are not found

# --- App Interface ---
# Set the page configuration for a better layout
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title('Iranian House Price Prediction App 🏠')
st.markdown("Enter the details of a house to get a price prediction based on our machine learning model.")

# Create input fields for the user in two columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    area = st.number_input('Area (in square meters)', min_value=30, max_value=1000, value=100, help="Enter the total area of the house.")
    room = st.selectbox('Number of Rooms', [0, 1, 2, 3, 4, 5], index=2, help="Select the number of bedrooms.")
    address = st.selectbox('Address / Neighborhood', address_list, help="Select the neighborhood.")

with col2:
    # Use empty st.write() to add vertical spacing for better alignment
    st.write("")
    st.write("")
    parking = st.checkbox('Has Parking?', value=True)
    warehouse = st.checkbox('Has Warehouse?', value=True)
    elevator = st.checkbox('Has Elevator?', value=True)

# --- Prediction Logic ---
# This block runs when the user clicks the 'Predict Price' button
if st.button('Predict Price', type="primary"):

    # 1. Create a dictionary to hold the user's input
    # The keys must match the feature names your model was trained on
    input_dict = {
        'Area': area,
        'Room': room,
        'Parking': parking,
        'Warehouse': warehouse,
        'Elevator': elevator
    }

    # 2. Handle the 'Address' One-Hot Encoding
    # This part is crucial for making the app work with pd.get_dummies()
    # Start by setting all possible address columns to 0
    for addr in address_list:
        # The column name must be exactly 'Address_Value'
        col_name = f'Address_{addr}'
        input_dict[col_name] = 0
    
    # Now, set the column for the address the user selected to 1
    selected_address_col = f'Address_{address}'
    if selected_address_col in model_columns:
        input_dict[selected_address_col] = 1

    # 3. Create a single-row DataFrame from the dictionary
    input_data = pd.DataFrame([input_dict])

    # 4. Ensure column order matches the model's training data
    # This is a critical step to prevent errors.
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # 5. Make the prediction
    try:
        # The model predicts on the log scale
        log_prediction = model.predict(input_data)
        
        # 6. Convert the prediction from log scale back to the original dollar scale
        price_prediction = np.expm1(log_prediction)

        # Display the final prediction in a nice format
        st.success(f'Predicted Price: ${price_prediction[0]:,.2f} USD')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
