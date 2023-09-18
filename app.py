# import readymade dependencies
import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU
from tensorflow.keras.models import Sequential
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import regex as re


# import custom dependencies
from utils  import submit


# Set title
st.title("Earthquake Intensity Prediction")





# Sidebar for user input
st.sidebar.subheader("User Input")
# model selection
model_type = st.sidebar.selectbox("Select a model type", ["GRU", "LSTM", "RNN"])
# receiver's email input
email = st.sidebar.text_input("Enter your email address")

# Check if the email is valid
if email:
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        st.sidebar.write("Please enter a valid email address")
        address = "invalid"
    else:
        address = "valid"









# Load data using file_uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)




# Submit Button! When the button is clicked, the submit function is executed!!
if st.button("Submit"):
    try:
        if email:
            if address == "valid":
                submit(model_type, df, email)
            else:
             st.write("Enter a valid reciever's email from the sidebar on the left")   
        else:
            st.write("Enter a valid reciever's email from the sidebar on the left")
    except:
        st.write("Please uopload a valid csv having fields: Timestamp, X, Y, Z, Intensity")



pwd = st.secrets["AUTH_PASSWORD"]\
st.write(pwd)
