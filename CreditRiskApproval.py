import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('risk_model_xgboost.pkl')

# Load data
df = pd.read_csv("Credit Approval.csv")
column_rename_dict = {
    'A1': 'Gender',
    'A2': 'Age',
    'A3': 'Debt',
    'A4': 'Education Level',
    'A5': 'Employment Status',
    'A6': 'Occupation',
    'A7': 'Marital Status/Family',
    'A8': 'Credit Amount Requested',
    'A9': 'Residential Status',
    'A10': 'Other Credits',
    'A11': 'Housing',
    'A12': 'Job Type',
    'A13': 'Number of Dependents',
    'A14': 'Income',
    'A15': 'Credit Approval Status'
}

# Rename the columns
df.rename(columns=column_rename_dict, inplace=True)

# Dictionaries to map coded values to descriptive names for each column
value_mapping = {
    'Gender': {'a': 'Female', 'b': 'Male'},
    'Education Level': {'u': 'University', 'y': 'High School', 'l': 'None'},
    'Employment Status': {'p': 'Private Sector', 'gg': 'Self-Employed', 'g': 'Government', 'f': 'Freelance'},
    'Occupation': {
        'ff': 'Professional/Executive', 'x': 'Skilled Labor', 'q': 'Semi-Skilled Labor',
        'i': 'Unskilled Labor', 'cc': 'Management', 'c': 'Clerical', 'k': 'Sales',
        'w': 'Service Industry', 'e': 'Agriculture', 'm': 'Manufacturing', 'd': 'Defense',
        'r': 'Retired', 'j': 'Journalism/Media', 'f': 'Freelance', 'aa': 'Temporary Worker'
    },
    'Marital Status/Family': {
        'h': 'Married with Children', 'v': 'Single without Children', 'n': 'Single with Children',
        'o': 'Divorced', 'j': 'Widowed', 'ff': 'Cohabiting', 'dd': 'Separated', 'z': 'Other',
        'g': 'Living with Partner','bb': 'Single with No Dependents'
    },
    'Residential Status': {'t': 'Owns Residence', 'f': 'Rents Residence'},
    'Job Type' : {'t': 'Full-Time','f': 'Part-Time'},
    'Other Credits': {'t': 'Available', 'f': 'Not Available'},
    'Number of Dependents': {'s': 'One', 'p': 'More than One', 'g': 'No Dependents'}
}

# Apply the mappings to the appropriate columns
for col, mapping in value_mapping.items():
    df[col] = df[col].replace(mapping)
df1 = df.copy()

st.title('Risk Model Prediction')

st.sidebar.header("User Information")

def user_input_features():
    residential_status = st.sidebar.selectbox('Residential Status', ('Rents Residence', 'Owns Residence'))
    occupation = st.sidebar.selectbox('Occupation', (
        'Professional/Executive', 'Skilled Labor', 'Semi-Skilled Labor', 'Unskilled Labor',
        'Management', 'Clerical', 'Sales', 'Service Industry', 'Agriculture', 'Manufacturing',
        'Defense', 'Retired', 'Journalism/Media','Temporary Worker','Freelance'
    ))
    marital_status = st.sidebar.selectbox('Marital Status/Family', (
        'Married with Children', 'Cohabiting', 'Single without Children', 'Single with Children',
        'Divorced', 'Widowed', 'Separated', 'Other','Living with Partner','Single with No Dependents'
    ))
    other_credits = st.sidebar.selectbox('Other Credits', ('Available','Not Available')),
    education_level = st.sidebar.selectbox('Education Level', ('High School', 'University','None'))
    employment_status = st.sidebar.selectbox('Employment Status', ('Private Sector', 'Self-Employed','Government', 'Freelance'))
    job_type = st.sidebar.selectbox('Job Type',('Full-Time','Part-Time')),
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    dependents = st.sidebar.selectbox('Number of Dependents', ('One', 'More than One','No Dependents'))

    # Other numerical inputs (like Age, Income, Debt) - add more as necessary
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    income = st.sidebar.number_input('Income', min_value=0, max_value=1000000, value=50000)
    debt = st.sidebar.number_input('Debt', min_value=0, max_value=1000000, value=10000)
    Credit_amount_requested = st.sidebar.number_input('Credit Amount Requested', min_value=0, max_value=1000000, value=10000)
    credit_approval_status = st.sidebar.number_input('Credit Approval Status', min_value=0, max_value=1000000, value=10000)
    housing = st.sidebar.number_input('Housing', min_value=0, max_value=20, value =1)

    # Collect all user input into a DataFrame
    input_data = {
        'Residential Status': [residential_status],
        'Occupation': [occupation],
        'Marital Status/Family': [marital_status],
        'Education Level': [education_level],
        'Employment Status': [employment_status],
        'Gender': [gender],
        'Number of Dependents': [dependents],
        'Age': [age],
        'Job Type': [job_type],
        'Other Credits': [other_credits],
        'Credit Amount Requested': [Credit_amount_requested],
        'Income': [income],
        'Debt': [debt],
        'Credit Approval Status': [credit_approval_status],
        'Housing':[housing],
        
    }
    
    # Convert user input to the model's expected format
    input_df = pd.DataFrame(input_data)
    return input_df

input_df = user_input_features()

def Preprocessed_data(data, model_features):
    def reverse_dataframe(df1):
        column_rename_dict1 = {
            'Gender': 'A1',
            'Age': 'A2',
            'Debt': 'A3',
            'Education Level': 'A4',
            'Employment Status': 'A5',
            'Occupation': 'A6',
            'Marital Status/Family': 'A7',
            'Credit Amount Requested': 'A8',
            'Residential Status': 'A9',
            'Other Credits': 'A10',
            'Housing': 'A11',
            'Job Type': 'A12',
            'Number of Dependents': 'A13',
            'Income': 'A14',
            'Credit Approval Status': 'A15'
        }

        # Rename the columns
        df1.rename(columns=column_rename_dict1, inplace=True)

        # Dictionaries to map descriptive names to coded values for each column
        value_mapping = {
            'A1': {'Female': 'a', 'Male': 'b'},
            'A4': {'University': 'u', 'High School': 'y', 'None': 'l'},
            'A5': {'Private Sector': 'p', 'Self-Employed': 'gg', 'Government': 'g', 'Freelance': 'f'},
            'A6': {
                'Professional/Executive': 'ff', 'Skilled Labor': 'x', 'Semi-Skilled Labor': 'q',
                'Unskilled Labor': 'i', 'Management': 'cc', 'Clerical': 'c', 'Sales': 'k',
                'Service Industry': 'w', 'Agriculture': 'e', 'Manufacturing': 'm', 'Defense': 'd',
                'Retired': 'r', 'Journalism/Media': 'j', 'Freelance': 'f', 'Temporary Worker': 'aa'
            },
            'A7': {
                'Married with Children': 'h', 'Single without Children': 'v', 'Single with Children': 'n',
                'Divorced': 'o', 'Widowed': 'j', 'Cohabiting': 'ff', 'Separated': 'dd', 'Other': 'z',
                'Living with Partner': 'g', 'Single with No Dependents': 'bb'
            },
            'A9': {'Owns Residence': 't', 'Rents Residence': 'f'},
            'A10': {'Available': 't', 'Not Available': 'f'},
            'A12': {'Full-Time': 't', 'Part-Time': 'f'},
            'A13': {'One': 's', 'More than One': 'p', 'No Dependents': 'g'}
        }

        # Apply the mappings to the appropriate columns
        for col, mapping in value_mapping.items():
            if col in df1.columns:
                df1[col] = df1[col].replace(mapping)
        
        return df1

    # Process data
    data = reverse_dataframe(data)
    data = pd.get_dummies(data, drop_first=True).astype(int)

    # Ensure columns align with model features
    

    for col in model_features:
        if col not in data.columns:
            data[col] = 0
    data = data[model_features]
    
    return data
model_features = [
        'A2', 'A3', 'A8', 'A11', 'A14', 'A15', 'A1_b', 'A4_u', 'A4_y',  'A5_gg', 'A5_p', 'A6_c', 'A6_cc', 
        'A6_d', 'A6_e', 'A6_ff', 'A6_i',    'A6_j', 'A6_k', 'A6_m', 'A6_q', 'A6_r', 'A6_w', 'A6_x', 'A7_dd', 
        'A7_ff', 'A7_h', 'A7_j', 'A7_n','A7_o', 'A7_v', 'A7_z', 'A9_t', 'A10_t', 'A12_t', 'A13_p', 'A13_s']

processed_data = Preprocessed_data(input_df, model_features)

st.subheader('Prediction')
try:
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    
    if prediction[0] == 1:
        st.write("The model predicts a high-risk category.")
    else:
        st.write("The model predicts a low-risk category.")
    st.write(f"Probability of High Risk: {prediction_proba[0][1]:.2f}")
except ValueError as e:
    st.write(f"Error in prediction: {e}. Please ensure inputs match model expectations.")