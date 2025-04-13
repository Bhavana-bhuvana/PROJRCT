import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

'''st.title("📂 File Upload & Data Preview")

# Layout: Sidebar + Main
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"],
                                     help="Choose a file to upload.")

    # File Save
    UPLOAD_FOLDER = "uploaded_files"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✔ File saved successfully: {file_path}")

# Data Display Section
with col2:
    st.markdown("### Preview Uploaded Data")
    if uploaded_file:
        def load_data(file):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                return pd.read_excel(file)
            elif file.name.endswith(".json"):
                return pd.read_json(file)


        data = load_data(uploaded_file)
        st.dataframe(data)

        # Model Selection
        st.markdown("### Select Model Type")
        model_type = st.selectbox("Choose a model:", ["Linear Regression", "Multiple Regression", "Clustering"])

        # Column Selection
        st.markdown("### Select Columns for Processing")
        columns = st.multiselect("Select feature columns:", data.columns)
        target_column = None
        if model_type in ["Linear Regression", "Multiple Regression"]:
            target_column = st.selectbox("Select target column:", data.columns)
        
        # Scatter plot: Feature vs Target (only if 1 feature is selected)
        if len(columns) == 1:
            st.markdown(f"###  Feature vs Target Plot: {columns[0]} vs {target_column}")
            fig2, ax2 = plt.subplots()
            ax2.scatter(X[columns[0]], y, color='green')
            ax2.set_xlabel(columns[0])
            ax2.set_ylabel(target_column)
            st.pyplot(fig2)


        # Process Data
        if st.button("Run Model") and columns:
            X = data[columns]
            if model_type in ["Linear Regression", "Multiple Regression"] and target_column:
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Output coefficients and intercept
                st.markdown("###  Model Parameters")
                st.write(f"**Intercept:** {model.intercept_}")
                coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
                st.dataframe(coef_df)

                # Output predictions vs actual
                results_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": predictions,
                    "Error": y_test.values - predictions
                })
                st.markdown("###  Prediction Results")
                st.dataframe(results_df)

                # Scatter plot: Actual vs Predicted
                st.markdown("### Actual vs Predicted Plot")
                fig1, ax1 = plt.subplots()
                ax1.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
                ax1.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Prediction')
                ax1.set_xlabel("Actual")
                ax1.set_ylabel("Predicted")
                ax1.legend()
                st.pyplot(fig1)'''
import streamlit as st
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Session State to keep track of navigation and data
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "model_data" not in st.session_state:
    st.session_state.model_data = {}

# Navigation buttons
if st.session_state.page == "results":
    if st.button("⬅ Go Back to Upload Page"):
        st.session_state.page = "upload"

if st.session_state.page == "upload":
    st.title("📂 File Upload & Data Preview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"],
                                         help="Choose a file to upload.")

        UPLOAD_FOLDER = "uploaded_files"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        if uploaded_file:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✔ File saved successfully: {file_path}")

    with col2:
        st.markdown("### Preview Uploaded Data")
        if uploaded_file:
            def load_data(file):
                if file.name.endswith(".csv"):
                    return pd.read_csv(file)
                elif file.name.endswith(".xlsx"):
                    return pd.read_excel(file)
                elif file.name.endswith(".json"):
                    return pd.read_json(file)

            data = load_data(uploaded_file)
            st.dataframe(data)

            st.markdown("### Select Model Type")
            model_type = st.selectbox("Choose a model:", ["Linear Regression", "Multiple Regression", "Clustering"])

            st.markdown("### Select Columns for Processing")
            columns = st.multiselect("Select feature columns:", data.columns)

            target_column = None
            if model_type in ["Linear Regression", "Multiple Regression"]:
                target_column = st.selectbox("Select target column:", data.columns)

            if st.button("Run Model") and columns:
                X = data[columns]
                y = data[target_column] if target_column else None

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.session_state.model_data = {
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": predictions,
                    "intercept": model.intercept_,
                    "coefficients": model.coef_,
                    "feature_columns": columns,
                    "data": data
                }
                st.session_state.page = "results"
                st.experimental_rerun()

elif st.session_state.page == "results":
    st.title(" Model Results and Customization")
    model_data = st.session_state.model_data

    intercept = model_data["intercept"]
    coefficients = model_data["coefficients"]
    feature_columns = model_data["feature_columns"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    predictions = model_data["predictions"]

    # Display original coefficients and predictions
    st.write("### 🔹 Original Intercept:", intercept)
    st.write("### 🔹 Original Coefficients:")
    for i, col in enumerate(feature_columns):
        st.write(f"{col}: {coefficients[i]}")

    st.write("###  Original Predictions:")
    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))

    # Editable fields
    st.markdown("---")
    st.markdown("###  Customize Parameters")
    custom_intercept = st.number_input("Custom Intercept", value=float(intercept), format="%.6f")

    custom_coefs = []
    st.markdown("#### 🔧 Coefficients for Each Feature")
    for i, col in enumerate(feature_columns):
        coef_val = st.number_input(
            f"Coefficient for **{col}**",
            value=float(coefficients[i]),
            format="%.6f",
            key=f"coef_{col}"
        )
        custom_coefs.append(coef_val)

    if st.button("Update Model with Custom Values"):
        # Recalculate predictions with custom coefficients
        X_df = X_test.copy()
        custom_preds = custom_intercept + X_df.dot(custom_coefs)

        # Save to session for traceability
        if "custom_runs" not in st.session_state:
            st.session_state.custom_runs = []

        run_df = pd.DataFrame({
            "Actual": y_test,
            "Custom Predicted": custom_preds
        })
        st.session_state.custom_runs.append({
            "intercept": custom_intercept,
            "coefficients": custom_coefs,
            "results": run_df
        })

    if "custom_runs" in st.session_state:
        st.markdown("###  Previous Custom Runs")
        for i, run in enumerate(st.session_state.custom_runs[::-1]):
            st.markdown(f"#### Run #{len(st.session_state.custom_runs) - i}")
            st.write(f"Intercept: {run['intercept']}")
            for j, coef in enumerate(run['coefficients']):
                st.write(f"{feature_columns[j]}: {coef}")
            st.dataframe(run["results"])


                