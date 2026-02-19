import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# load the trained model
model = tf.keras.models.load_model('model.h5')

# load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load the label encoder
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

st.title("Customer Churn Prediction")

# user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=16, max_value=69, value=30)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=1000, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
number_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the features
input_data_scaled = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Predicted probability of churn: {prediction_proba:.2f}")

# Display the prediction
if prediction_proba > 0.5:
    st.write("The model predicts that this customer is likely to churn.")
else:
    st.write("The model predicts that this customer is not likely to churn.")