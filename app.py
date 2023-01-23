# import all the app dependencies
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import matplotlib
from IPython import get_ipython
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# load the encoder and model object
model = joblib.load("churn_model_deploy.joblib")
encoder = joblib.load("label_encoder.joblib")
scaler = joblib.load("standard_scaler.joblib")

# 0. Not Churning 1.Churned
st.set_page_config(page_title="Churn Prediction App",
                page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_gender = ["Male", "Female"]
options_senior = ["Yes", "No"]
options_partner = ["Yes", "No"]
options_dependants = ["Yes", "No"]
options_phoneservice = ["Yes", "No"]
options_multiplelines = ['No', 'Yes', 'No phone service']
options_internet = ['Fiber optic', 'DSL', 'No']
options_onlinesecurity = ['No', 'Yes', 'No internet service']
options_backup = ['No', 'Yes', 'No internet service']
options_deviceprotection = ['No', 'Yes', 'No internet service']
options_techsupport = ['No', 'Yes', 'No internet service']
options_streamingtv = ['No', 'Yes', 'No internet service']
options_streamingmovies = ['No', 'Yes', 'No internet service']
options_contract = ['Month-to-month', 'Two year', 'One year']
options_paperlessbilling = ['Yes', 'No']
options_paymentmethod = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)']

features_list = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges']

#title for the webapp
st.markdown("Churn Prediction App", unsafe_allow_html=True)

# main() function to take inputs from user in form based approch
def main():
    with st.form("Prediction form"):
        st.subheader("Please enter the following inputs:")
        gender = st.selectbox("Gender", options=options_gender)
        SeniorCitizen = st.selectbox("SeniorCitizen", options = options_senior)
        Partner = st.selectbox("Partner", options = options_partner)
        Dependents = st.selectbox("Dependants", options = options_dependants)
        tenure = st.number_input("Tenure")
        PhoneService = st.selectbox("PhoneService", options = options_phoneservice)
        MultipleLines = st.selectbox("MultipleLines", options = options_multiplelines)
        InternetService = st.selectbox("InternetService", options = options_internet)
        OnlineSecurity = st.selectbox("OnlineSecurity", options = options_onlinesecurity)
        OnlineBackup = st.selectbox("OnlineBackup", options = options_backup)
        DeviceProtection = st.selectbox("DeviceProtection", options = options_deviceprotection)
        TechSupport = st.selectbox("TechSupport", options = options_techsupport)
        StreamingTV = st.selectbox("StreamingTV", options = options_streamingtv)
        StreamingMovies = st.selectbox("StreamingMovies", options = options_streamingmovies)
        Contract = st.selectbox("Contract", options = options_contract)
        PaperlessBilling = st.selectbox("PaperlessBilling", options = options_paperlessbilling)
        PaymentMethod = st.selectbox("PaymentMethod", options = options_paymentmethod)
        MonthlyCharges = st.number_input("MonthlyCharges")
        TotalCharges = st.number_input("TotalCharges")
        
        submit = st.form_submit_button("Predict")
        
    if submit:
        input_array1 = np.array([gender, SeniorCitizen, Partner, Dependents])
        input_array2 = np.array([PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                                 DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                                 PaymentMethod])
        #encoded_arr1 = list(encoder.transform(input_array1).ravel())
        #encoded_arr2 = list(encoder.transform(input_array2).ravel())
        input_array = np.array([gender, Partner, Dependents, PhoneService, MultipleLines,
                                 InternetService, OnlineSecurity, OnlineBackup,
                                 DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                                PaymentMethod], ndmin = 2)

        #for i in input_array:
        encoded_arr = list(encoder.transform(input_array).ravel())
        if SeniorCitizen == 'Yes':
            senior = 1
        else:
            senior = 0
        num_arr1= [tenure, senior]
        num_arr2 = [MonthlyCharges, TotalCharges]
        #pred_arr = np.array(encoded_arr1 + num_arr1 + encoded_arr2 + num_arr2).reshape(1,-1)
        pred_arr = np.array(encoded_arr + num_arr1 + num_arr2).reshape(1,-1)

        # predict the target from all the input features
        prediction = model.predict(pred_arr)
            
        if prediction == 0:
            st.write(f"The Customer will not Churn")
        else:
            st.write(f"The Customer will Churn immediately")
            

# run the main function               
if __name__ == '__main__':
   main()
