{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "20e6998e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import all the app dependencies\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "import streamlit as st\n",
    "import joblib\n",
    "import shap\n",
    "import matplotlib\n",
    "from IPython import get_ipython\n",
    "from PIL import Image\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fc2aa203",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load the encoder and model object\n",
    "model = joblib.load(\"churn_model_deploy.joblib\")\n",
    "encoder = joblib.load(\"label_encoder.joblib\")\n",
    "scaler = joblib.load(\"standard_scaler.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b4630683",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0. Not Churning 1.Churned\n",
    "st.set_page_config(page_title=\"Churn Prediction App\",\n",
    "                page_icon=\"🚧\", layout=\"wide\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "616e921e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#creating option list for dropdown menu\n",
    "options_gender = [\"Male\", \"Female\"]\n",
    "options_senior = [\"Yes\", \"No\"]\n",
    "options_partner = [\"Yes\", \"No\"]\n",
    "options_dependants = [\"Yes\", \"No\"]\n",
    "options_phoneservice = [\"Yes\", \"No\"]\n",
    "options_multiplelines = ['No', 'Yes', 'No phone service']\n",
    "options_internet = ['Fiber optic', 'DSL', 'No']\n",
    "options_onlinesecurity = ['No', 'Yes', 'No internet service']\n",
    "options_backup = ['No', 'Yes', 'No internet service']\n",
    "options_deviceprotection = ['No', 'Yes', 'No internet service']\n",
    "options_techsupport = ['No', 'Yes', 'No internet service']\n",
    "options_streamingtv = ['No', 'Yes', 'No internet service']\n",
    "options_streamingmovies = ['No', 'Yes', 'No internet service']\n",
    "options_contract = ['Month-to-month', 'Two year', 'One year']\n",
    "options_paperlessbilling = ['Yes', 'No']\n",
    "options_paymentmethod = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',\n",
    "       'Credit card (automatic)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1d65ba09",
   "metadata": {},
   "outputs": [],
   "source": [
    "features_list = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',\n",
    "       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',\n",
    "       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',\n",
    "       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',\n",
    "       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',\n",
    "       'TotalCharges']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "bd396034",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#title for the webapp\n",
    "st.markdown(\"Churn Prediction App\", unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "fe2f025d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# main() function to take inputs from user in form based approch\n",
    "def main():\n",
    "    with st.form(\"Prediction form\"):\n",
    "        st.subheader(\"Please enter the following inputs:\")\n",
    "        gender = st.selectbox(\"Gender\", options=options_gender)\n",
    "        SeniorCitizen = st.selectbox(\"Senior Citizen\", options = options_senior)\n",
    "        Partner = st.selectbox(\"Partner\", options = options_partner)\n",
    "        Dependents = st.selectbox(\"Dependants\", options = options_dependants)\n",
    "        tenure = st.number_input(\"Tenure\")\n",
    "        PhoneService = st.selectbox(\"PhoneService\", options = options_phoneservice)\n",
    "        MultipleLines = st.selectbox(\"MultipleLines\", options = options_multiplelines)\n",
    "        InternetService = st.selectbox(\"InternetService\", options = options_internet)\n",
    "        OnlineSecurity = st.selectbox(\"OnlineSecurity\", options = options_onlinesecurity)\n",
    "        OnlineBackup = st.selectbox(\"OnlineBackup\", options = options_options_backup)\n",
    "        DeviceProtection = st.selectbox(\"DeviceProtection\", options = options_deviceprotection)\n",
    "        TechSupport = st.selectbox(\"TechSupport\", options = options_techsupport)\n",
    "        StreamingTV = st.selectbox(\"StreamingTV\", options = options_streamingtv)\n",
    "        StreamingMovies = st.selectbox(\"StreamingMovies\", options = options_streamingmovies)\n",
    "        Contract = st.selectbox(\"Contract\", options = options_contract)\n",
    "        PaperlessBilling = st.selectbox(\"PaperlessBilling\", options = options_paperlessbilling)\n",
    "        PaymentMethod = st.selectbox(\"PaymentMethod\", options = options_paymentmethod)\n",
    "        MonthlyCharges = st.number_input(\"MonthlyCharges\")\n",
    "        TotalCharges = st.number_input(\"TotalCharges\")\n",
    "        \n",
    "        submit = st.form_submit_button(\"Predict\")\n",
    "        \n",
    "    if submit:\n",
    "        input_array1 = np.array([gender, SeniorCitizen, Partner, Dependents])\n",
    "        input_array2 = np.array([PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,\n",
    "                                 DeviceProtection,\n",
    "                                 TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod])\n",
    "        encoded_arr1 = list(encoder.transform(input_array1).ravel())\n",
    "        encoded_arr2 = list(encoder.transform(input_array2).ravel())\n",
    "        num_arr1= [tenure]\n",
    "        num_arr2 = [MonthlyCharges, TotalCharges]\n",
    "        pred_arr = np.array(encoded_arr1 + num_arr1 + encoded_arr2 + num_arr2).reshape(1,-1)\n",
    "        \n",
    "        # predict the target from all the input features\n",
    "        prediction = model.predict(pred_arr)\n",
    "            \n",
    "        if prediction == 0:\n",
    "            st.write(f\"The Customer will not Churn\")\n",
    "        else:\n",
    "            st.write(f\"The Customer will Churn immediately\")\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f034dff5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de788bd0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94ac612f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76bdf719",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0df8e590",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9e7095c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e3dcb27",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "612af4f7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "029e4fbf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "645f28b0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "399f2cc9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "39bf226d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20796325",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e6d582a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4319ded7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bac7e7d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04159e1d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0920497",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfba3872",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a40480c5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb4bce3a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a0b4bad",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12da7c2e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13eae25c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10221bf6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dce130ed",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b6cf0e3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9419cfa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
