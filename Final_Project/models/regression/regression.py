import streamlit as st
from models.regression.linear_regression import linear_regression_page
from models.regression.polynomial_regression import polynomial_regression_page
from models.regression.multiple_linear_regression import multiple_linear_regression_page
from models.regression.decision_tree_regression import decision_tree_regression_page
from models.regression.random_forest_regression import random_forest_regression_page
from models.regression.svr_regression import svr_regression_page

def regression_page(model_name, data):
    st.subheader("Select Features and Target")
    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    if model_name == "Linear Regression":
        linear_regression_page(data) 
    elif model_name == "Polynomial Regression":
        polynomial_regression_page(data)
    elif model_name == "Multiple Linear Regression":
        multiple_linear_regression_page(data) 
    elif model_name == "Decision Tree Regression":
        decision_tree_regression_page(data )
    elif model_name == "Random Forest Regression":
        random_forest_regression_page(data)
    elif model_name == "Support Vector Regression":
        svr_regression_page(data)
    else:
        st.error("Unsupported Regression Model")

