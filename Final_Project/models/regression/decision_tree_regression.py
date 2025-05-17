import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
from datetime import datetime


def decision_tree_regression_page(data):
    st.header("Decision Tree Regressor")

    if "dtr_history" not in st.session_state:
        st.session_state.dtr_history = []

    st.markdown("""
    #### Feature Selection
    <span style='font-size: 14px;'>
     Choose feature columns (independent variables) and one target column.
    </span>
    """, unsafe_allow_html=True)

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if len(features) == 0:
        st.warning("Please select at least one feature column.")
        return
    if target in features:
        st.warning("Target column must not be in features.")
        return

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_depth = st.slider("Max Depth of Tree:", min_value=1, max_value=20, value=5)

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.subheader("Model Performance")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("MAE", f"{mae:.4f}")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")

    st.session_state.dtr_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MaxDepth": max_depth,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    st.subheader("Actual vs Predicted")
    scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig = px.scatter(scatter_df, x="Actual", y="Predicted", title="Actual vs Predicted Values")
    fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Predict New Data")
    new_input = []
    for feature in features:
        val = st.number_input(f"Enter value for {feature}", key=f"dtr_{feature}")
        new_input.append(val)

    if st.button("Predict with New Input"):
        prediction = model.predict([new_input])[0]
        st.success(f"Predicted {target}: {prediction:.4f}")

    if st.session_state.dtr_history:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.dtr_history)
        st.dataframe(hist_df, use_container_width=True)

    with st.expander("📘 What is a Decision Tree Regressor?"):
        st.markdown("""
        - **Decision Tree** models split the dataset based on feature values to form a tree structure.
        - **Max Depth** controls how deep the tree can grow. Deeper trees may overfit, shallow trees may underfit.
        - **R² Score**: Measures how well the model explains variance in target.
        - **MAE, MSE, RMSE**: Error metrics to evaluate prediction accuracy.
        - **Actual vs Predicted**: The closer the points are to the red line, the better the model.
        """)
