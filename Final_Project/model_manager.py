# # model_manager.py
# import streamlit as st
# from models.regression import regression_page
# from models.classification import classification_page
# from models.clustering import clustering_page

# MODEL_FUNCTIONS = {
#     "Regression": regression_page,
#     "Classification": classification_page,
#     "Clustering": clustering_page
# }

# def run_selected_model(model_name, data):
#     st.markdown(f"## 📌 {model_name} Module")
#     if model_name in MODEL_FUNCTIONS:
#         MODEL_FUNCTIONS[model_name](data)
#     else:
#         st.warning(f"No handler defined for model: {model_name}")
# import streamlit as st
# from models.classification import classification_page
# from models.clustering import clustering_page
# from models.regression import regression_page
# from ui_manager import show_upload_ui, show_model_selection_ui

# # Initialize session state
# if "page" not in st.session_state:
#     st.session_state.page = "upload"

# if "selected_models" not in st.session_state:
#     st.session_state.selected_models = []

# if "data" not in st.session_state:
#     st.session_state.data = None

# # Sidebar Navigation
# with st.sidebar:
#     st.header("Model Selection")
#     if st.session_state.page == "upload":
#         st.info("Upload a dataset to get started.")
#     elif st.session_state.page == "model_selection":
#         st.button("Choose a model to apply or explore dataset", on_click=lambda: None)
#     else:
#         for model in st.session_state.selected_models:
#             if st.button(model):
#                 st.session_state.page = model
#                 st.rerun()

# # Page Routing
# def run_selected_model(model_name, data):
#     if model_name == "Regression":
#         from regression import run_regression
#         return run_regression(data)
#     elif model_name == "Classification":
#         from classification import run_classification
#         return run_classification(data)
#     elif model_name == "Clustering":
#         from clustering import run_clustering
#         return run_clustering(data)
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")
import streamlit as st
from models.classification import classification_page
from models.clustering import clustering_page
from models.regression import regression_page


def run_selected_model(model_name, data):
    if model_name == "Regression":
        return regression_page(data)
    elif model_name == "Classification":
        return classification_page(data)
    elif model_name == "Clustering":
        return clustering_page(data)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
