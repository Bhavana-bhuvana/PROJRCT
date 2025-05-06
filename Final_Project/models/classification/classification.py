import streamlit as st
from models.classification.logistic_regression import logistic_regression_page
from models.classification.decision_tree_classification import decision_tree_classification_page
from models.classification.random_forest_classification import random_forest_classification_page
from models.classification.svm_classification import svm_classification_page
from models.classification.knn_classification import knn_classification_page
from models.classification.naive_bayes_classification import naive_bayes_classification_page

def classification_page(model_name, data):
    if model_name == "Logistic Regression":
        logistic_regression_page(data)
    elif model_name == "Decision Tree":
        decision_tree_classification_page(data)
    elif model_name == "Random Forest":
        random_forest_classification_page(data)
    elif model_name == "SVM":
        svm_classification_page(data)
    elif model_name == "KNN":
        knn_classification_page(data)
    elif model_name == "Naive Bayes":
        naive_bayes_classification_page(data)
    else:
        st.error("Unsupported Classification Model")

