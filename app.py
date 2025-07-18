import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_data
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return pickle.load(f)
    
with open('pipeline.pkl', 'rb') as f:
    pp= pickle.load(f)

model=load_model()

# Title of the app
st.title("Online Payment Fraud Detection")
st.subheader("Enter feature values:")


f1 = st.number_input("Transaction Amount", value=0.0)
f2 = st.number_input("Time of Transaction", value=0.0)
f3 = float(st.selectbox("Previous Fradaulent Transactions", options=[0,1,2,3,4]))
f4 = st.slider("Account Age",0,140)
f5 = st.number_input("No. of Transactions Last 24h", value=0.0)
f6=st.selectbox("Transaction Type",options=['Bill Payment','ATM Withdrawal','Bank Transfer','Online Purchase','POS Payment'])
f7=st.selectbox("Location",options=['Chicago','New York','San Francisco','Seattle','Los Angeles','Miami','Houston','Boston'])
f8=st.selectbox("Device Used",options=['Mobile','Tablet','Desktop','Unknown Device'])
f9=st.selectbox("Payment Method",options=['UPI','Debit Card','Net Banking','Credit Card','Invalid Method'])

input_df = pd.DataFrame([{
    "Transaction_Amount": f1,
    "Time_of_Transaction": f2,
    "Previous_Fraudulent_Transactions":f3,
    "Account_Age": f4,
    "Number_of_Transactions_Last_24H":f5,
    "Transaction_Type":f6,
    "Location":f7,
    "Device_Used":f8,
    "Payment_Method":f9
}])

data_trans=pp.transform(input_df)
# Prediction button
if st.button("Predict"):
    pred=model.predict(data_trans)
    st.write("Prediction")
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    if pred ==1:
        st.error("**Fraudulent Transaction Detected!**")
        st.write("The transaction is a Fraud")
    else:
        st.success("**Transaction Appears Legitimate**")
        st.write("The transaction is not a Fraud")   
