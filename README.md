# Company_leaving_prediction
This project implements a machine learning model for predicting employee attrition (whether an employee will leave the company or not). The model is trained using a Random Forest Classifier on employee data and has been deployed as a web application using Streamlit.
# Employee Attrition Prediction Model

## Overview

This repository contains a machine learning project that predicts employee attrition (whether an employee is likely to leave the company). The model is built using a **Random Forest Classifier** and can be deployed as a web application using **Streamlit**.

## Key Features

- **Model Type**: Random Forest Classifier
- **Prediction Task**: Predict whether an employee will leave the company or not.
- **Input Features**: 
  - `number_of_projects`: Number of projects the employee has worked on.
  - `tenure`: Number of years the employee has been with the company.
  - `overworked`: Whether the employee is overworked (`Yes` or `No`).
  - `had_work_accident`: Whether the employee had a work-related accident (`Yes` or `No`).
  - `last_performance_rating`: The employee's last performance rating (a score between 1.0 and 5.0).
  - `salary_level`: Employee's salary level (`low`, `medium`, `high`).
  - `department`: Department the employee works in (e.g., `IT`, `HR`, `Sales`, `Marketing`, etc.).

- **Output**:
  - The model predicts if the employee will leave the company (`1` for Yes, `0` for No).

## Features & Capabilities

1. **Data Preprocessing**:
   - **Ordinal Encoding**: The `salary_level` feature is encoded into numerical values (`low = 0`, `medium = 1`, `high = 2`).
   - **One-Hot Encoding**: The `department` feature is one-hot encoded to represent various departments (`IT`, `HR`, `Sales`, etc.).
   - **Binary Encoding**: The `overworked` and `had_work_accident` features are converted to binary values (`Yes = 1`, `No = 0`).

2. **Model Training**:
   - The project uses **Random Forest Classifier** to predict employee attrition.
   - The trained model is saved as a `.pkl` file, which is used in the deployment step.

3. **Deployment**:
   - The model is deployed as a **Streamlit web application**, where users can input employee data and get a prediction on whether the employee is likely to leave the company.

