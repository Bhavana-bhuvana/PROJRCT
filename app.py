# import streamlit as st
# import os
# import pandas as pd
# import numpy as np 
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# import streamlit as st
# import os
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split

# # Session State to keep track of navigation and data
# if "page" not in st.session_state:
#     st.session_state.page = "upload"
# if "model_data" not in st.session_state:
#     st.session_state.model_data = {}

# # Navigation buttons
# if st.session_state.page == "results":
#     if st.button(" Go Back to Upload Page"):
#         st.session_state.page = "upload"

# if st.session_state.page == "upload":
#     st.title(" File Upload & Data Preview")

#     col1, col2 = st.columns([1, 2])

#     with col1:
#         st.markdown("### Upload Your Dataset")
#         uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"],
#                                          help="Choose a file to upload.")

#         UPLOAD_FOLDER = "uploaded_files"
#         os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#         if uploaded_file:
#             file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             st.success(f"✔ File saved successfully: {file_path}")

#     with col2:
#         st.markdown("### Preview Uploaded Data")
#         if uploaded_file:
#             def load_data(file):
#                 if file.name.endswith(".csv"):
#                     return pd.read_csv(file)
#                 elif file.name.endswith(".xlsx"):
#                     return pd.read_excel(file)
#                 elif file.name.endswith(".json"):
#                     return pd.read_json(file)

#             data = load_data(uploaded_file)
#             st.dataframe(data)

#             st.markdown("### Select Model Type")
#             model_type = st.selectbox("Choose a model:", ["Linear Regression", "Multiple Regression", "Clustering"])

#             st.markdown("### Select Columns for Processing")
#             columns = st.multiselect("Select feature columns:", data.columns)

#             target_column = None
#             if model_type in ["Linear Regression", "Multiple Regression"]:
#                 target_column = st.selectbox("Select target column:", data.columns)

#             if st.button("Run Model") and columns:
#                 X = data[columns]
#                 y = data[target_column] if target_column else None

#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                 model = LinearRegression()
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_test)

#                 st.session_state.model_data = {
#                     "X_test": X_test,
#                     "y_test": y_test,
#                     "predictions": predictions,
#                     "intercept": model.intercept_,
#                     "coefficients": model.coef_,
#                     "Error": y_test.values - predictions,
#                     "feature_columns": columns,
#                     "data": data
#                 }
#                 st.session_state.page = "results"
#                 st.rerun()

# elif st.session_state.page == "results":
#     st.title(" Model Results and Customization")
#     model_data = st.session_state.model_data

#     intercept = model_data["intercept"]
#     coefficients = model_data["coefficients"]
#     error=model_data["Error"]
#     feature_columns = model_data["feature_columns"]
#     X_test = model_data["X_test"]
#     y_test = model_data["y_test"]
#     predictions = model_data["predictions"]

#     # Display original coefficients and predictions
#     st.write("###  Original Intercept:", intercept)
#     st.write("###  Original Coefficients:")
#     for i, col in enumerate(feature_columns):
#         st.write(f"{col}: {coefficients[i]}")
#     st.write("### ERROR ",error)
#     st.write("###  Original Predictions:")
#     st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))

#     # Editable fields
#     st.markdown("---")
#     st.markdown("###  Customize Parameters")
#     custom_intercept = st.number_input("Custom Intercept", value=float(intercept), format="%.6f")

#     custom_coefs = []
#     st.markdown("#### Coefficients for Each Feature")
#     for i, col in enumerate(feature_columns):
#         coef_val = st.number_input(
#             f"Coefficient for **{col}**",
#             value=float(coefficients[i]),
#             format="%.6f",
#             key=f"coef_{col}"
#         )
#         custom_coefs.append(coef_val)

#     if st.button("Update Model with Custom Values"):
#         # Recalculate predictions with custom coefficients
#         X_df = X_test.copy()
#         custom_preds = custom_intercept + X_df.dot(custom_coefs)

#         # Save to session for traceability
#         if "custom_runs" not in st.session_state:
#             st.session_state.custom_runs = []

#         run_df = pd.DataFrame({
#             "Actual": y_test,
#             "Custom Predicted": custom_preds
#         })
#         st.session_state.custom_runs.append({
#             "intercept": custom_intercept,
#             "coefficients": custom_coefs,
#             "results": run_df
#         })

#     if "custom_runs" in st.session_state:
#         st.markdown("###  Previous Custom Runs")
#         for i, run in enumerate(st.session_state.custom_runs[::-1]):
#             st.markdown(f"#### Run #{len(st.session_state.custom_runs) - i}")
#             st.write(f"Intercept: {run['intercept']}")
#             for j, coef in enumerate(run['coefficients']):
#                 st.write(f"{feature_columns[j]}: {coef}")
#             st.dataframe(run["results"])
# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
# import numpy as np

# st.set_page_config(page_title=" ML Playground", layout="wide")
# st.title(" Machine Learning Playground")
# st.caption("Upload a dataset and apply Regression, Classification, or Clustering")

# # Initialize session state
# if "page" not in st.session_state:
#     st.session_state.page = "upload"
# if "data" not in st.session_state:
#     st.session_state.data = None
# if "model_data" not in st.session_state:
#     st.session_state.model_data = {}

# # Sidebar Menu
# with st.sidebar:
#     st.title(" Settings")
#     if st.session_state.page == "upload":
#         st.subheader("Step 1: Upload Data")
#     elif st.session_state.page == "select_model":
#         st.subheader("Step 2: Configure Model")
#     elif st.session_state.page == "results":
#         st.subheader("Step 3: View Results")
#     st.markdown("---")
#     st.caption("Made with  using Streamlit")

# # Upload Page
# if st.session_state.page == "upload":
#     uploaded_file = st.file_uploader(" Upload CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

#     if uploaded_file:
#         file_type = uploaded_file.name.split(".")[-1]
#         if file_type == "csv":
#             data = pd.read_csv(uploaded_file)
#         elif file_type in ["xlsx", "xls"]:
#             data = pd.read_excel(uploaded_file)
#         elif file_type == "json":
#             data = pd.read_json(uploaded_file)
#         else:
#             st.error("Unsupported file format!")
#             st.stop()

#         st.success(f" Uploaded Successfully: {uploaded_file.name}")
#         st.session_state.data = data
#         st.session_state.page = "select_model"
#         st.rerun()

# # Model Selection Page
# elif st.session_state.page == "select_model":
#     st.header(" Model Configuration")
#     data = st.session_state.data
#     st.dataframe(data, use_container_width=True)

#     with st.sidebar:
#         st.subheader("Choose Model")
#         model_type = st.radio("Select model type:", ["Linear Regression", "Multiple Regression", "Clustering", "Classification"])

#         columns = st.multiselect("Select input features (X):", data.columns.tolist())

#         target_column = None
#         if model_type in ["Linear Regression", "Multiple Regression", "Classification"]:
#             target_column = st.selectbox("Select target column (y):", data.columns.tolist())

#         cluster_count = 3
#         if model_type == "Clustering":
#             cluster_count = st.number_input("Select number of clusters (K)", min_value=2, max_value=10, value=3, step=1)

#         st.markdown("---")
#         run_model = st.button(" Run Model")

#     # Instructions
#     st.info(f"**Instructions:** Select input columns (X) and a target (y) if applicable, then click Run.")

#     if run_model and columns:
#         X = data[columns]
#         y = data[target_column] if target_column else None

#         if model_type in ["Linear Regression", "Multiple Regression"]:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LinearRegression()
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)

#             st.session_state.model_data = {
#                 "model_type": "regression",
#                 "X_test": X_test,
#                 "y_test": y_test,
#                 "predictions": predictions,
#                 "intercept": model.intercept_,
#                 "coefficients": model.coef_,
#                 "feature_columns": columns,
#                 "data": data
#             }

#         elif model_type == "Clustering":
#             model = KMeans(n_clusters=int(cluster_count), random_state=42)
#             model.fit(X)
#             cluster_labels = model.labels_

#             st.session_state.model_data = {
#                 "model_type": "clustering",
#                 "cluster_labels": cluster_labels,
#                 "feature_columns": columns,
#                 "cluster_count": cluster_count,
#                 "data": data
#             }

#         elif model_type == "Classification":
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LogisticRegression()
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)
#             acc = accuracy_score(y_test, predictions)
#             conf_mat = confusion_matrix(y_test, predictions)

#             st.session_state.model_data = {
#                 "model_type": "classification",
#                 "X_test": X_test,
#                 "y_test": y_test,
#                 "predictions": predictions,
#                 "accuracy": acc,
#                 "confusion_matrix": conf_mat,
#                 "feature_columns": columns,
#                 "data": data
#             }

#         st.session_state.page = "results"
#         st.rerun()

# # Results Page
# elif st.session_state.page == "results":
#     st.header(" Model Results")
#     model_data = st.session_state.model_data
#     model_type = model_data.get("model_type")

#     if model_type == "regression":
#         st.subheader(" Regression Analysis")

#         intercept = st.number_input("Intercept", value=float(model_data["intercept"]))
#         coefficients = []

#         st.markdown("### Adjust Coefficients if needed")
#         for i, col in enumerate(model_data["feature_columns"]):
#             coef = st.number_input(f"Coefficient for **{col}**", value=float(model_data["coefficients"][i]))
#             coefficients.append(coef)

#         X_test = model_data["X_test"]
#         y_test = model_data["y_test"]

#         updated_predictions = intercept + np.dot(X_test, coefficients)

#         rmse = mean_squared_error(y_test, updated_predictions, squared=False)
#         mae = mean_absolute_error(y_test, updated_predictions)

#         st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
#         st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")

#         st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": updated_predictions}))

#     elif model_type == "clustering":
#         st.subheader("🔹 Clustering Analysis")
#         data_with_clusters = model_data["data"].copy()
#         data_with_clusters["Cluster"] = model_data["cluster_labels"]
#         st.dataframe(data_with_clusters, use_container_width=True)

#         st.success(f"Formed {model_data['cluster_count']} Clusters")

#         st.markdown("### Cluster Distribution")
#         st.bar_chart(data_with_clusters["Cluster"].value_counts())

#     elif model_type == "classification":
#         st.subheader("🔹 Classification Analysis")

#         st.metric(label="Accuracy", value=f"{model_data['accuracy']:.4f}")

#         st.markdown("### Confusion Matrix")
#         st.dataframe(pd.DataFrame(
#             model_data["confusion_matrix"],
#             columns=[f"Predicted {i}" for i in range(model_data["confusion_matrix"].shape[1])],
#             index=[f"Actual {i}" for i in range(model_data["confusion_matrix"].shape[0])]
#         ))

#         st.markdown("### Actual vs Predicted")
#         st.dataframe(pd.DataFrame({
#             "Actual": model_data["y_test"].values,
#             "Predicted": model_data["predictions"]
#         }))

#     if st.button(" Upload New Data"):
#         st.session_state.page = "upload"
#         st.session_state.data = None
#         st.session_state.model_data = {}
#         st.rerun()
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

# # Streamlit Page Configuration
# st.set_page_config(page_title="ML Playground", layout="wide")
# st.title(" Machine Learning Playground")
# st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")

# # Initialize session states
# if "page" not in st.session_state:
#     st.session_state.page = "upload"
# if "data" not in st.session_state:
#     st.session_state.data = None
# if "model_data" not in st.session_state:
#     st.session_state.model_data = {}

# # Sidebar
# with st.sidebar:
#     st.title(" Settings")
#     step_titles = {"upload": "Step 1: Upload Data", "select_model": "Step 2: Configure Model", "results": "Step 3: View Results"}
#     st.subheader(step_titles.get(st.session_state.page, ""))
#     st.markdown("---")
#     st.caption("Made with  using Streamlit")

# # Upload Page
# if st.session_state.page == "upload":
#     uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

#     if uploaded_file:
#         file_type = uploaded_file.name.split(".")[-1]
#         try:
#             if file_type == "csv":
#                 data = pd.read_csv(uploaded_file)
#             elif file_type in ["xlsx", "xls"]:
#                 data = pd.read_excel(uploaded_file)
#             elif file_type == "json":
#                 data = pd.read_json(uploaded_file)
#             else:
#                 st.error("Unsupported file format!")
#                 st.stop()
#         except Exception as e:
#             st.error(f"Error reading file: {e}")
#             st.stop()

#         st.success(f"Uploaded Successfully: {uploaded_file.name}")
#         st.session_state.data = data
#         st.session_state.page = "select_model"
#         st.rerun()

# # Model Selection Page
# elif st.session_state.page == "select_model":
#     st.header("🔧 Model Configuration")
#     data = st.session_state.data
#     st.dataframe(data, use_container_width=True)

#     with st.sidebar:
#         st.subheader("Choose Model Type")
#         model_type = st.radio("Model Type:", ["Linear Regression", "Multiple Regression", "Clustering", "Classification"])

#         columns = st.multiselect("Select input features (X):", data.columns.tolist())

#         target_column = None
#         if model_type in ["Linear Regression", "Multiple Regression", "Classification"]:
#             target_column = st.selectbox("Select Target Column (y):", data.columns.tolist())

#         cluster_count = 3
#         if model_type == "Clustering":
#             cluster_count = st.number_input("Number of Clusters (K):", min_value=2, max_value=10, value=3, step=1)

#         st.markdown("---")
#         run_model = st.button("Run Model")

#     st.info("**Instructions:** Select input features (X) and a target (y) if required, then click **Run Model**.")

#     if run_model and columns:
#         X = data[columns]
#         y = data[target_column] if target_column else None

#         if model_type in ["Linear Regression", "Multiple Regression"]:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LinearRegression()
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)

#             st.session_state.model_data = {
#                 "model_type": "regression",
#                 "X_test": X_test,
#                 "y_test": y_test,
#                 "predictions": predictions,
#                 "intercept": model.intercept_,
#                 "coefficients": model.coef_,
#                 "feature_columns": columns,
#                 "data": data
#             }

#         elif model_type == "Clustering":
#             model = KMeans(n_clusters=int(cluster_count), random_state=42)
#             model.fit(X)
#             cluster_labels = model.labels_

#             st.session_state.model_data = {
#                 "model_type": "clustering",
#                 "cluster_labels": cluster_labels,
#                 "feature_columns": columns,
#                 "cluster_count": cluster_count,
#                 "data": data
#             }

#         elif model_type == "Classification":
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LogisticRegression()
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)
#             acc = accuracy_score(y_test, predictions)
#             conf_mat = confusion_matrix(y_test, predictions)

#             st.session_state.model_data = {
#                 "model_type": "classification",
#                 "X_test": X_test,
#                 "y_test": y_test,
#                 "predictions": predictions,
#                 "accuracy": acc,
#                 "confusion_matrix": conf_mat,
#                 "feature_columns": columns,
#                 "data": data
#             }

#         st.session_state.page = "results"
#         st.rerun()

# # Results Page
# elif st.session_state.page == "results":
#     st.header("📊 Model Results")
#     model_data = st.session_state.model_data
#     model_type = model_data.get("model_type")

#     if model_type == "regression":
#         st.subheader("📈 Regression Analysis")
#         intercept = st.number_input("Intercept:", value=float(model_data["intercept"]))
#         coefficients = []

#         st.markdown("### 🔄 Adjust Coefficients (Optional)")
#         for i, col in enumerate(model_data["feature_columns"]):
#             coef = st.number_input(f"Coefficient for **{col}**:", value=float(model_data["coefficients"][i]))
#             coefficients.append(coef)

#         X_test = model_data["X_test"]
#         y_test = model_data["y_test"]

#         updated_predictions = intercept + np.dot(X_test, coefficients)

#         rmse = mean_squared_error(y_test, updated_predictions, squared=False)  # RMSE (Root Mean Squared Error)
#         mae = mean_absolute_error(y_test, updated_predictions)  # MAE (Mean Absolute Error)
#         r2_score_value = 1 - (np.sum((y_test - updated_predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))  # R² Score

#         col1, col2, col3 = st.columns(3)
#         col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
#         col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
#         col3.metric("R² Score", f"{r2_score_value:.4f}")

#         st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": updated_predictions}))

#     elif model_type == "clustering":
#         st.subheader("🔹 Clustering Analysis")
#         data_with_clusters = model_data["data"].copy()
#         data_with_clusters["Cluster"] = model_data["cluster_labels"]
#         st.dataframe(data_with_clusters, use_container_width=True)

#         st.success(f"Formed {model_data['cluster_count']} clusters!")

#         st.markdown("### 📊 Cluster Distribution")
#         st.bar_chart(data_with_clusters["Cluster"].value_counts())

#     elif model_type == "classification":
#         st.subheader("🔹 Classification Analysis")
#         st.metric("Accuracy", f"{model_data['accuracy']:.4f}")

#         st.markdown("### 📉 Confusion Matrix")
#         st.dataframe(pd.DataFrame(
#             model_data["confusion_matrix"],
#             columns=[f"Predicted {i}" for i in range(model_data["confusion_matrix"].shape[1])],
#             index=[f"Actual {i}" for i in range(model_data["confusion_matrix"].shape[0])]
#         ))

#         st.markdown("### 📋 Actual vs Predicted")
#         st.dataframe(pd.DataFrame({
#             "Actual": model_data["y_test"].values,
#             "Predicted": model_data["predictions"]
#         }))

#     st.markdown("---")
#     if st.button("📤 Upload New Data"):
#         st.session_state.page = "upload"
#         st.session_state.data = None
#         st.session_state.model_data = {}
#         st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Configuration
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("Machine Learning Playground")
st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")
st.markdown("### Instructions:")
st.markdown("- **Step 1:** Upload a dataset (CSV, Excel, or JSON).")
st.markdown("- **Step 2:** Choose a model type (Regression, Classification, or Clustering).")
st.markdown("- **Step 3:** Modify model parameters and observe results interactively.")

# Initialize session states
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "data" not in st.session_state:
    st.session_state.data = None
if "model_data" not in st.session_state:
    st.session_state.model_data = {}

# Sidebar
with st.sidebar:
    st.title("Settings")
    step_titles = {
        "upload": "Step 1: Upload Data",
        "select_model": "Step 2: Configure Model",
        "results": "Step 3: View Results"
    }
    st.subheader(step_titles.get(st.session_state.page, ""))
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

# Upload Page
if st.session_state.page == "upload":
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]
        try:
            if file_type == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                data = pd.read_excel(uploaded_file)
            elif file_type == "json":
                data = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format!")
                st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.success(f"Uploaded Successfully: {uploaded_file.name}")
        st.session_state.data = data
        st.session_state.page = "select_model"
        st.rerun()

# Model Selection Page
elif st.session_state.page == "select_model":
    st.header("🔧 Model Configuration")
    data = st.session_state.data
    st.dataframe(data, use_container_width=True)

    with st.sidebar:
        st.subheader("Choose Model Type")
        model_type = st.radio("Model Type:", ["Linear Regression", "Multiple Regression", "Clustering", "Classification"])

        columns = st.multiselect("Select input features (X):", data.columns.tolist())

        target_column = None
        if model_type in ["Linear Regression", "Multiple Regression", "Classification"]:
            target_column = st.selectbox("Select Target Column (y):", data.columns.tolist())

        cluster_count = 3
        if model_type == "Clustering":
            cluster_count = st.number_input("Number of Clusters (K):", min_value=2, max_value=10, value=3, step=1)

        st.markdown("---")
        run_model = st.button("Run Model")

    st.info("**Instructions:** Select input features (X) and a target (y) if required, then click **Run Model**.")
    st.info("You can adjust model settings such as the number of clusters or coefficients to observe how changes affect the results.")

    if run_model and columns:
        X = data[columns]
        y = data[target_column] if target_column else None

        if model_type in ["Linear Regression", "Multiple Regression"]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.session_state.model_data = {
                "model_type": "regression",
                "X_test": X_test,
                "y_test": y_test,
                "predictions": predictions,
                "intercept": model.intercept_,
                "coefficients": model.coef_,
                "feature_columns": columns,
                "data": data
            }

        elif model_type == "Clustering":
            model = KMeans(n_clusters=int(cluster_count), random_state=42)
            model.fit(X)
            cluster_labels = model.labels_

            st.session_state.model_data = {
                "model_type": "clustering",
                "cluster_labels": cluster_labels,
                "feature_columns": columns,
                "cluster_count": cluster_count,
                "data": data
            }

        elif model_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            conf_mat = confusion_matrix(y_test, predictions)

            st.session_state.model_data = {
                "model_type": "classification",
                "X_test": X_test,
                "y_test": y_test,
                "predictions": predictions,
                "accuracy": acc,
                "confusion_matrix": conf_mat,
                "feature_columns": columns,
                "data": data
            }

        st.session_state.page = "results"
        st.rerun()

# Results Page
elif st.session_state.page == "results":
    st.header("📊 Model Results")
    model_data = st.session_state.model_data
    model_type = model_data.get("model_type")

    if model_type == "regression":
        st.subheader("📈 Regression Analysis")
        intercept = st.number_input("Intercept:", value=float(model_data["intercept"]))
        coefficients = []

        st.markdown("### 🔄 Adjust Coefficients (Optional)")
        for i, col in enumerate(model_data["feature_columns"]):
            coef = st.number_input(f"Coefficient for **{col}**:", value=float(model_data["coefficients"][i]))
            coefficients.append(coef)

        X_test = model_data["X_test"]
        y_test = model_data["y_test"]

        updated_predictions = intercept + np.dot(X_test, coefficients)
        
        mse = mean_squared_error(y_test, updated_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, updated_predictions)
        r2_score_value = 1 - (np.sum((y_test - updated_predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        col1, col2, col3 = st.columns(3)
        col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
        col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        col3.metric("R² Score", f"{r2_score_value:.4f}")

        st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": updated_predictions}))

    elif model_type == "clustering":
        st.subheader("🔹 Clustering Analysis")
        data_with_clusters = model_data["data"].copy()
        data_with_clusters["Cluster"] = model_data["cluster_labels"]
        st.dataframe(data_with_clusters, use_container_width=True)

        st.success(f"Formed {model_data['cluster_count']} clusters!")

        st.markdown("### 📈 Visualize Clusters")

        feature_cols = model_data["feature_columns"]

        if len(feature_cols) >= 2:
            x_axis = st.selectbox("Select X-axis feature:", feature_cols, index=0)
            y_axis = st.selectbox("Select Y-axis feature:", feature_cols, index=1)

            fig, ax = plt.subplots()
            sns.scatterplot(
                data=data_with_clusters,
                x=x_axis,
                y=y_axis,
                hue="Cluster",
                palette="tab10",
                s=60,
                ax=ax
            )
            plt.title("Cluster Visualization")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend(title="Cluster")
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 features to plot clusters.")

    elif model_type == "classification":
        st.subheader("🔹 Classification Analysis")
        st.metric("Accuracy", f"{model_data['accuracy']:.4f}")

        st.markdown("### 📉 Confusion Matrix")
        st.dataframe(pd.DataFrame(
            model_data["confusion_matrix"],
            columns=[f"Predicted {i}" for i in range(model_data["confusion_matrix"].shape[1])],
            index=[f"Actual {i}" for i in range(model_data["confusion_matrix"].shape[0])]
        ))

        st.markdown("### 📋 Actual vs Predicted")
        st.dataframe(pd.DataFrame({
            "Actual": model_data["y_test"].values,
            "Predicted": model_data["predictions"]
        }))

    st.markdown("---")
    if st.button("📤 Upload New Data"):
        st.session_state.page = "upload"
        st.session_state.data = None
        st.session_state.model_data = {}
        st.rerun()
