import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Add clinic logo
st.sidebar.image("reign clinic.png", use_container_width=True)

# Define categorical and numerical features
categorical_features = ['Department', 'Doctor', 'Visit Type']
numerical_features = ['Age', 'Previous Visits', 'Appointment Time (24hr)']

# Streamlit App
st.title("Smart Healthcare - Patient Wait Time Predictor")
st.write("Enter patient details to predict total wait time.")

# Input Fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
prev_visits = st.number_input("Previous Visits", min_value=0, value=1)
appt_time = st.number_input("Appointment Time (24hr)", min_value=0, max_value=23, value=10)
department = st.selectbox("Department", ['General', 'Cardiology', 'Orthopedics', 'Dermatology', 'Neurology'])
doctor = st.selectbox("Doctor", ['Dr. Smith', 'Dr. Johnson', 'Dr. Lee', 'Dr. Patel', 'Dr. Gomez'])
visit_type = st.selectbox("Visit Type", ['Routine', 'Emergency', 'Follow-up'])

# Define pre-fitted encoder and scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

# Sample DataFrame to fit encoders (Reduced for speed optimization)
sample_data = pd.DataFrame({
    'Department': ['General', 'Cardiology', 'Orthopedics'],
    'Doctor': ['Dr. Smith', 'Dr. Johnson', 'Dr. Lee'],
    'Visit Type': ['Routine', 'Emergency', 'Follow-up'],
    'Age': [30, 40, 50],
    'Previous Visits': [1, 2, 3],
    'Appointment Time (24hr)': [10, 11, 12]
})

encoder.fit(sample_data[categorical_features])
scaler.fit(sample_data[numerical_features])

# Train a Random Forest model
X_sample_encoded = encoder.transform(sample_data[categorical_features])
X_sample_scaled = scaler.transform(sample_data[numerical_features])
X_sample_final = np.hstack((X_sample_scaled, X_sample_encoded))

y_sample = np.array([15, 20, 25])  # Sample wait times
rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)  # Reduced complexity for faster execution
rf_model.fit(X_sample_final, y_sample)

# Predict Button
if st.button("Predict Wait Time"):
    # Create DataFrame from input
    input_data = pd.DataFrame([[age, prev_visits, appt_time, department, doctor, visit_type]],
                              columns=numerical_features + categorical_features)
    
    # Encode and scale input
    encoded_categorical = encoder.transform(input_data[categorical_features])
    scaled_numerical = scaler.transform(input_data[numerical_features])
    final_input = np.hstack((scaled_numerical, encoded_categorical))
    
    # Predict wait time
    predicted_time = rf_model.predict(final_input)[0]
    
    # Display Prediction
    st.success(f"Predicted Total Wait Time: {predicted_time:.2f} minutes")
