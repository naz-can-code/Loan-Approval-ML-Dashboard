#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load and clean data
train = pd.read_csv('train.csv')
train = train.dropna()

# Feature engineering
train['TotalApplicantIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train = pd.get_dummies(train, columns=['Gender'], drop_first=True)
train = pd.get_dummies(train, columns=['Married'], drop_first=True)
train = pd.get_dummies(train, columns=['Loan_Status'], drop_first=True)
train = train.rename(columns={'Loan_Status_Y': 'Loan_Approved'})
train['Credit_History'] = train['Credit_History'].astype(int)

# Define features and target
features = ['Gender_Male', 'Married_Yes', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History']
X = train[features]
y = train['Loan_Approved']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

# Train model
forest = RandomForestClassifier(max_depth=4, random_state=10, n_estimators=100, min_samples_leaf=5)
model = forest.fit(x_train, y_train)

# Prediction function
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):
    # Encode inputs
    Gender = 1 if Gender == "Male" else 0
    Married = 1 if Married == "Married" else 0
    Credit_History = 1 if Credit_History == "Has Credit History" else 0
    LoanAmount = LoanAmount / 1000

    # Create input DataFrame with correct column names
    input_df = pd.DataFrame([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]],
                            columns=['Gender_Male', 'Married_Yes', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History'])

    # Make prediction
    pred_inputs = model.predict(input_df)

    # Return result
    if pred_inputs[0] == 0:
        return 'I am sorry, you have been rejected for the loan.'
    elif pred_inputs[0] == 1:
        return 'Congrats! You have been approved for the loan!'
    else:
        return 'Error'

# Streamlit app
def main():
    st.markdown("""
    <div style="background-color:teal;padding:13px">
    <h1 style="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1>
    </div>
    """, unsafe_allow_html=True)

    # User inputs
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    ApplicantIncome = st.number_input("Total Monthly Income (Include Coborrower if Applicable)")
    LoanAmount = st.number_input("Loan Amount (e.g., 125000)")
    Credit_History = st.selectbox('Credit History', ("Has Credit History", "No Credit History"))

    # Prediction trigger
    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        st.success(f'Final Decision: {result}')
        st.write(f"Loan Amount Entered: {LoanAmount}")

if __name__ == '__main__':
    main()