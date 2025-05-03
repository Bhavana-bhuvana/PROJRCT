import streamlit as st
import pandas as pd

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
