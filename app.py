import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🏥 Diabetes Readmission Risk Predictor")
st.write("Enter patient details in the sidebar and click on 'Predict Risk' to see the result.")

# 2. Input fields - important features
st.sidebar.header("Patient Data Input")

time_in_hospital = st.sidebar.slider("Time in Hospital", 1, 14, 3)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 40)
num_medications = st.sidebar.slider("Medications", 1, 81, 15)
number_outpatient = st.sidebar.slider("Outpatient Visits", 0, 42, 2)
number_emergency = st.sidebar.slider("Emergency Visits", 0, 76, 0)
number_inpatient = st.sidebar.slider("Inpatient Visits", 0, 21, 0)
Total_Medications = st.sidebar.slider("Total Diabetes Meds", 0, 4, 1)
age = st.sidebar.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
race = st.sidebar.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
insulin = st.sidebar.selectbox("Insulin", ['No', 'Up', 'Down', 'Steady'])

# 3. Make dataframe same as training
input_data = pd.DataFrame([{
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'Total_Medications': Total_Medications,
    'age': age,
    'race': race,
    'gender': gender,
    'insulin': insulin
}])

# 4. get_dummies and align columns with training
input_data = pd.get_dummies(input_data, drop_first=True)

# align columns with training data
# 4. get_dummies and align columns with training
input_data = pd.get_dummies(input_data, drop_first=True)

# align columns with training data - YE WALI 2 LINE NAYI HAIN
train_cols = joblib.load('columns.joblib')
input_data = input_data.reindex(columns=train_cols, fill_value=0)

# 5. Scale and Predict
input_scaled = scaler.transform(input_data)
input_data = input_data.reindex(columns=train_cols, fill_value=0)

# 5. Scale and Predict
input_scaled = scaler.transform(input_data)

if st.button("Predict Risk"):
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[:,1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Readmission! \nProbability: {proba[0]*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Readmission \nProbability: {proba[0]*100:.2f}%")