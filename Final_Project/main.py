import streamlit as st
from model_manager import run_selected_model
from ui_manager import display_upload_ui
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
with st.sidebar:
    st.title("Model Navigation")

    if "data" in st.session_state:
        with st.expander("📄 Preview Dataset"):
            st.dataframe(st.session_state.data.head(), use_container_width=True)

    if st.session_state.selected_models:
        st.subheader("Selected Models:")
        for model in st.session_state.selected_models:
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"{model}", key=f"goto_{model}"):
                    log_user_action(st.session_state["user_email"], f"Switched to model: {model}")
                    st.session_state.page = model
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{model}"):
                    st.session_state.selected_models.remove(model)
                    log_user_action(st.session_state["user_email"], f"Removed model: {model}")
                    if st.session_state.page == model:
                        st.session_state.page = "model_selection"
                    st.rerun()

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

    st.subheader("Choose Model Type")
    main_models = ["Regression", "Classification", "Clustering"]
    selected_main_model = st.selectbox("Add another model type", options=main_models)

    # Filter already added submodels
    added_submodels = {m.split(": ")[1] for m in st.session_state.selected_models if m.startswith(selected_main_model)}
    available_submodels = [s for s in submodels[selected_main_model] if s not in added_submodels]

    if available_submodels:
        selected_submodels = st.multiselect(
            f"Select {selected_main_model} algorithms",
            options=available_submodels,
            key=f"select_{selected_main_model}"
        )

        if selected_submodels and st.button("Add Model(s)"):
            for sub in selected_submodels:
                model_key = f"{selected_main_model}: {sub}"
                if model_key not in st.session_state.selected_models:
                    st.session_state.selected_models.append(model_key)
                    log_user_action(st.session_state["user_email"], f"Added model: {model_key}")
            st.session_state.page = f"{selected_main_model}: {selected_submodels[0]}"
            st.rerun()
    else:
        st.info("All models added.")

    st.markdown("---")
    if st.button("Upload New Data"):
        log_user_action(st.session_state["user_email"], "Started new data upload")
        st.session_state.page = "upload"
        st.session_state.selected_models = []
        st.session_state.model_results = {}
        st.rerun()

# Main Area Routing
if st.session_state.page == "upload":
    display_upload_ui()
elif st.session_state.page in st.session_state.selected_models:
    run_selected_model(st.session_state.page, st.session_state.data)
else:
    st.error("Invalid page state.")
