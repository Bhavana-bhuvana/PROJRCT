import streamlit as st
from model_manager import run_selected_model
from ui_manager import display_upload_ui, display_cleaning_ui
from db import log_user_action  # MongoDB logger
from login_signup import login_signup_ui

# Ensure user is logged in before using the app
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_signup_ui()
    st.stop()

# Page Config
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("Machine Learning Playground")
st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")

# Session State Init
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# Sidebar Navigation
page = st.session_state.page

st.sidebar.title("Navigation")

if st.sidebar.button("Upload Data"):
    st.session_state.page = "upload"
    st.experimental_rerun()

if st.sidebar.button("Data Cleaning"):
    st.session_state.page = "Data Cleaning"
    st.experimental_rerun()

if "raw_data" in st.session_state:
    with st.sidebar.expander("Original Dataset"):
        st.dataframe(st.session_state.raw_data.head(), use_container_width=True)

if "cleaned_data" in st.session_state:
    with st.sidebar.expander("Cleaned Dataset"):
        st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)

if st.session_state.selected_models:
    st.sidebar.subheader("Selected Models:")
    for model in st.session_state.selected_models:
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            if st.button(f"{model}", key=f"goto_{model}"):
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Switched to model: {model}")
                st.session_state.page = model
                st.experimental_rerun()
        with col2:
            if st.button("❌", key=f"remove_{model}"):
                st.session_state.selected_models.remove(model)
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Removed model: {model}")
                if st.session_state.page == model:
                    st.session_state.page = "model_selection"
                st.experimental_rerun()

# Only show model options after data is uploaded
submodels = {
    "Regression": [
        "Linear Regression", "Polynomial Regression", "Multiple Linear Regression",
        "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression"
    ],
    "Classification": [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "SVM", "KNN", "Naive Bayes"
    ],
    "Clustering": [
        "K-Means", "DBSCAN", "Gaussian Mixture Model", "Hierarchical"
    ]
}

st.sidebar.subheader("Add Models")
main_models = ["Regression", "Classification", "Clustering"]
selected_main_model = st.sidebar.selectbox("Choose Model Type", options=main_models)

# Filter already added submodels
added_submodels = {m.split(": ")[1] for m in st.session_state.selected_models if m.startswith(selected_main_model)}
available_submodels = [s for s in submodels[selected_main_model] if s not in added_submodels]

if available_submodels:
    selected_submodels = st.sidebar.multiselect(
        f"Select {selected_main_model} algorithms",
        options=available_submodels,
        key=f"select_{selected_main_model}"
    )

    if selected_submodels and st.sidebar.button("Add Model(s)"):
        for sub in selected_submodels:
            model_key = f"{selected_main_model}: {sub}"
            if model_key not in st.session_state.selected_models:
                st.session_state.selected_models.append(model_key)
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Added model: {model_key}")
        st.session_state.page = f"{selected_main_model}: {selected_submodels[0]}"
        st.experimental_rerun()
else:
    st.sidebar.info("All models added.")

st.sidebar.markdown("---")

if st.sidebar.button("Upload New Data"):
    log_user_action(st.session_state.get("user_email", "anonymous"), "Started new data upload")
    st.session_state.page = "upload"
    st.session_state.selected_models = []
    st.session_state.model_results = {}
    st.experimental_rerun()

# Main Area Routing
if st.session_state.page == "upload":
    display_upload_ui()
elif st.session_state.page == "Data Cleaning":
    display_cleaning_ui()
elif st.session_state.page in st.session_state.selected_models:
    run_selected_model(st.session_state.page, st.session_state.get("cleaned_data", st.session_state.get("raw_data")))
else:
    st.error("Invalid page state.")
