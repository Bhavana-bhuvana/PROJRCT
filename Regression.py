import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

st.title("Model Training")

if "data" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

data = st.session_state.data
columns = data.columns.tolist()

features = st.multiselect("Select Feature Columns", columns)
target = st.selectbox("Select Target Column", columns)

if st.button("Train Model"):
    if not features or not target:
        st.error("Please select features and target.")
        st.stop()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    error_type = "RMSE" if rmse < mae else "MAE"
    error_value = min(mae, rmse)

    st.session_state.model = {
        "intercept": model.intercept_,
        "coefficients": model.coef_,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "features": features,
        "error_type": error_type,
        "error_value": error_value
    }

    st.success("Model trained successfully ")

    st.markdown(f"**Intercept:** {model.intercept_}")
    for f, c in zip(features, model.coef_):
        st.markdown(f"- {f}: {c}")
    
    st.markdown(f"**{error_type}** = {error_value:.4f}")
    st.dataframe(
        pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions,
            "Error": y_test.values - predictions
        })
    )
