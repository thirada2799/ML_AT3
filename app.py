#!/usr/bin/env python
# coding: utf-8

# !pip install streamlit

import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained KNN model
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

# Preprocessing steps
def preprocess_data(data):

	# Calculate transaction amount per account
    amount_per_acct = data.groupby('acct_num')['amt'].transform('sum')
    data['amt_per_acct'] = amount_per_acct

    # Calculate transaction frequency per merchant
    freq_per_merch = data.groupby('merchant')['trans_num'].transform('count')
    data['freq_per_merch'] = freq_per_merch

    # Select features
    selected_features = ['acct_num', 'zip', 'category', 'age', 'amt_per_acct', 'freq_per_merch']
    data = data[selected_features]

    # Label encoding
    le = LabelEncoder()
    data['acct_num'] = le.fit_transform(data['acct_num'])
    data['zip'] = le.fit_transform(data['zip'])
    data['category'] = le.fit_transform(data['category'])

    return data
    
def predict(transaction_data):
    # Preprocess the transaction data
    preprocessed_data = preprocess_data(pd.DataFrame([transaction_data]))

    # Scale the preprocessed data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(preprocessed_data)

    # Perform prediction using the loaded model
    prediction = knn_model.predict(scaled_data)[0]

    # Convert prediction to "fraud" or "not fraud"
    if prediction == 1:
        return "<span style='color:red;font-weight:bold;'>The customer has a risk of fraudulent behavior</span>"
    else:
        return "<span style='color:green;font-weight:bold;'>The customer does not have a risk of fraudulent behavior</span>"

    return prediction

def main():
    st.title(':bank: Fraud Detection App')

    # Get the transaction data from the user
    transaction_data = {}
    transaction_data['acct_num'] = st.text_input('Account Number', key='acct_num')
    transaction_data['zip'] = st.text_input('ZIP Code', key='zip')
    transaction_data['category'] = st.selectbox('Category', ['shopping_pos', 'home', 'grocery_pos', 'kids_pets', 'gas_transport', 'food_dining', 'entertainment', 'shopping_net', 'personal_care', 'misc_pos', 'health_fitness', 'misc_net', 'grocery_net', 'travel'], key='category')
    transaction_data['age'] = st.slider('Age', 18, 100, 30, key='age')
    transaction_data['merchant'] = st.text_input('Merchant', key='merchant')
    transaction_data['amt'] = st.number_input('Transaction Amount', key='amt')
    transaction_data['trans_num'] = st.text_input('Transaction Number', key='trans_num')
   

    # Predict fraud
    if st.button('Predict', key='predict_button'):
        prediction = predict(transaction_data)
        st.markdown('Prediction: {}'.format(prediction), unsafe_allow_html=True)
        st.write('36106 Machine Learning Algorithms and Applications - Autumn 2023') 
        st.write('Thirada Tiamklang 14337188')
        
        


if __name__ == '__main__':
    main()










