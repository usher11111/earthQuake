# import dependencies
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
from tensorflow.keras.models import Sequential
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Function to convert the raw data to timeseries format
def get_timeseries_data(data):
    data.set_index('Timestamp', inplace=True)
    data_ts = data.resample('0.05S').mean()
    data_ts = data_ts.ffill()
    data_ts['Intensity'] =  np.ceil(data_ts['Intensity']).astype(int)
    return data_ts


# Function to prepare data
def prepareData(data, lookback):
    data = data.values
    X_ = []
    Y_ = []
    for i in range(lookback, len(data)):
        X_.append(data[i - lookback: i])
        Y_.append(data[i][-1])
    X_ = np.array(X_)
    Y_ = np.array(Y_)
    return X_, Y_


# Model train, evaluate, and save function for GRU
def trainGRU(Xtrain, Ytrain, epoch):
    model = Sequential()
    model.add(GRU(10, activation='relu', return_sequences=True))
    model.add(GRU(5, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='RMSProp', loss='mse')
    r = model.fit(Xtrain, Ytrain, epochs=epoch, validation_split=0.3, batch_size=10, verbose=0)
    return model


# Model train, evaluate, and save function for LSTM
def trainLSTM(Xtrain, Ytrain, epoch):
    model = Sequential()
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(5, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='RMSProp', loss='mse')
    r = model.fit(Xtrain, Ytrain, epochs=epoch, validation_split=0.3, batch_size=10, verbose=0)
    return model


# Model train, evaluate, and save function for RNN
def trainRNN(Xtrain, Ytrain, epoch):
    model = Sequential()
    model.add(SimpleRNN(10, activation='relu', return_sequences=True))
    model.add(SimpleRNN(5, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='RMSProp', loss='mse')
    r = model.fit(Xtrain, Ytrain, epochs=epoch, validation_split=0.3, batch_size=10, verbose=0)
    return model


# Define a function for data preprocessing
def preprocess_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df_ts = get_timeseries_data(df)
    return df_ts


# Define a function for model training
def train_model(model_type, x_train, y_train):
    if model_type == 'GRU':
        model = trainGRU(x_train, y_train, 20)
    elif model_type == 'LSTM':
        model = trainLSTM(x_train, y_train, 20)
    elif model_type == 'RNN':
        model = trainRNN(x_train, y_train, 20)
    return model


# Define a function for performance evaluation
def evaluate_model(model, x_test, y_test):
    y_pred = np.round(model.predict(x_test))
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    mse = mean_squared_error(y_pred, y_test)
    mae = mean_absolute_error(y_pred, y_test)
    return y_pred, rmse, mse, mae


# Define a function for sending emails
def send_email(alert_data, email_, password_):

    # Email configuration
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server address
    smtp_port = 587  # Replace with your SMTP server's port number (587 for TLS)
    sender_email = "ushersense@gmail.com"  # Replace with your email address
    receiver_email = email_  # Replace with the recipient's email address
    password = password_  # Replace with your email password

    # Create the email content
    subject = 'Earthquake alert'
    message = 'Hello, this email includes x, y, and z locations when the earthquake intensity is predicted as 2'
    message_html = f'<html><body><p>{message}</p>{alert_data.to_html(index=False)}</body></html>'

    # Create a MIMEText object to represent the email text (plain text)
    email_text = MIMEText(message, 'plain')

    # Create a MIMEText object to represent the email HTML content
    email_html = MIMEText(message_html, 'html')

    # Create a MIMEMultipart object to represent the email
    email = MIMEMultipart()
    email['Subject'] = subject
    email['From'] = sender_email
    email['To'] = receiver_email

    # Attach both the plain text and HTML content to the email
    email.attach(email_text)
    email.attach(email_html)

    # Send the email
    with st.spinner("\n\nSending email..."):

        try:

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # Use TLS (Transport Layer Security) for encryption
            server.login(sender_email, password)  # Login to your email account
            server.sendmail(sender_email, receiver_email, email.as_string())
            server.quit()
            
            st.success(f'Email sent successfully to {email_}')

        except Exception as e:
            st.error(f'Error: {str(e)}')


# Define the submit function
def submit(model_type, df, email, password):
    try: 
        df_ts = preprocess_data(df)
        train, test = train_test_split(df_ts, test_size=0.2, shuffle=False)
        x_train, y_train = prepareData(train, 2)
        x_test, y_test = prepareData(test, 2)

        with st.spinner(f"\n\nTraining the {model_type} model..."):
            model = train_model(model_type, x_train, y_train)

        # Evaluate the model
        y_pred, rmse, mse, mae = evaluate_model(model, x_test, y_test)
        
        st.write("\n\nPerformance evaluation:")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Create DataFrame for alert data
        response_data = test.tail(-2)
        response_data['prediction'] = y_pred

        st.dataframe(response_data, height=400)
        alert_data = response_data[response_data['prediction'] > 1]

        # Send email
        send_email(alert_data, email, password)

    except:
        st.write("Please upload a valid file")
