import streamlit as st
import pandas as pd
import smtplib
from email.message import EmailMessage
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.warning("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def show_sidebar():
    st.sidebar.title("Model Selection")
    st.sidebar.info("Choose a model to apply or explore dataset")

def reset_session():
    st.session_state.uploaded_data = None
    st.session_state.active_models = []
    st.session_state.model_results = {}



def send_otp_email(receiver_email, otp):
    msg = EmailMessage()
    msg.set_content(f"Your OTP for registration is: {otp}")
    msg['Subject'] = "Your OTP Code"
    msg['From'] = "komal0mallaram@gmail.com"
    msg['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login("komal0mallaram@gmail.com", "wvma uvep ocew kodq")  # Use env vars in real apps
        smtp.send_message(msg)
