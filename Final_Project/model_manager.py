# import streamlit as st
# import streamlit as st
# #
# # from models.classification.classification import classification_page
# from models.clustering.clustering import clustering_page
# from models.regression.regression import regression_page

# def run_selected_model(model_name, data):
#     st.sidebar.subheader("Choose Model Type")
#     model_type = st.sidebar.selectbox("Model Type", ["Regression", "Classification", "Clustering"])

#     if model_type == "Regression":
#         regression_model = st.sidebar.selectbox("Select Regression Algorithm", [
#             "Linear Regression",
#             "Polynomial Regression",
#             "Multiple Linear Regression",
#             "Decision Tree Regression",
#             "Random Forest Regression",
#             "Support Vector Regression"
#         ])
#         regression_page(regression_model, data)

#     # elif model_type == "Classification":
#     #     classification_model = st.sidebar.selectbox("Select Classification Algorithm", [
#     #         "Logistic Regression",
#     #         "Decision Tree",
#     #         "Random Forest",
#     #         "SVM",
#     #         "KNN",
#     #         "Naive Bayes"
#     #     ])
#     #     classification_page(classification_model, data)

#     elif model_type == "Clustering":
#         clustering_model = st.sidebar.selectbox("Select Clustering Algorithm", [
#             "K-Means",
#             "DBSCAN",
#             "Gaussian Mixture Model",
#             "Hierarchical"
#         ])
#         clustering_page(clustering_model, data) 
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")



import streamlit as st
from models.regression.regression import regression_page
from models.classification.classification import classification_page
from models.clustering.clustering import clustering_page

def run_selected_model(model_name, data):
    # Expecting model_name format: "Regression: Linear Regression"
    try:
        model_type, algorithm = model_name.split(": ")
    except ValueError:
        st.error("Invalid model name format.")
        return

    if model_type == "Regression":
        regression_page(algorithm, data)

    elif model_type == "Classification":
        classification_page(algorithm, data)

    elif model_type == "Clustering":
        clustering_page(algorithm, data)

    else:
        st.error(f"Unsupported model type: {model_type}")
