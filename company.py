import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('Company_leaving_employee.pkl')

# Function to encode the input features
def encode_features(data, columns_to_encode):
    # Ordinal encoding for 'salary_level'
    salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
    data['salary_level'] = data['salary_level'].map(salary_mapping)
    
    # One-hot encoding for 'department'
    department_dummies = pd.get_dummies(data['department'], drop_first=True)
    data = pd.concat([data, department_dummies], axis=1)
    data = data.drop(['department'], axis=1)
    
    # Encoding for 'overworked' and 'had_work_accident'
    data['overworked'] = data['overworked'].map({'Yes': 1, 'No': 0})
    data['had_work_accident'] = data['had_work_accident'].map({'Yes': 1, 'No': 0})
    
    # Ensure the encoded DataFrame has the same columns as the trained model
    for col in columns_to_encode:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value 0
    
    # Reorder columns to match training data
    data = data[columns_to_encode]
    
    return data

# Streamlit app title
st.title("Employee Attrition Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

number_of_projects = st.sidebar.slider("Number of Projects", 1, 15, 5)
tenure = st.sidebar.slider("Tenure (in years)", 1, 30, 5)
overworked = st.sidebar.radio("Overworked", ("Yes", "No"))
had_work_accident = st.sidebar.radio("Had Work Accident", ("Yes", "No"))
last_performance_rating = st.sidebar.slider("Last Performance Rating", 1.0, 5.0, 3.5)

# Salary level options
salary_level = st.sidebar.selectbox("Salary Level", ("low", "medium", "high"))

# Department options
department = st.sidebar.selectbox("Department", ("it", "rand", "accounting", "hr", "management", "marketing", "product_mng", "sales", "support", "technical"))

# Prepare the input data for prediction
input_data = {
    'number_of_projects': [number_of_projects],
    'tenure': [tenure],
    'overworked': [overworked],
    'had_work_accident': [had_work_accident],
    'last_performance_rating': [last_performance_rating],
    'salary_level': [salary_level],
    'department': [department]
}

# Convert input data into a DataFrame
input_df = pd.DataFrame(input_data)

# Get the feature names from the model
columns_to_encode = model.feature_names_in_

# Encode the features
input_df = encode_features(input_df, columns_to_encode)

# Predict using the trained model
prediction = model.predict(input_df)

# Display the prediction result
if prediction[0] == 1:
    st.write("### Prediction: Employee might leave the company!")
else:
    st.write("### Prediction: Employee is likely to stay with the company!")
