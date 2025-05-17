import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
from datetime import datetime


def random_forest_regression_page(data):
    st.header("Random Forest Regressor")

    if "rfr_history" not in st.session_state:
        st.session_state.rfr_history = []

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

    n_estimators = st.slider("Number of Trees (n_estimators):", min_value=10, max_value=200, value=100, step=10)
    max_depth = st.slider("Max Depth of Each Tree:", min_value=1, max_value=30, value=10)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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

    st.session_state.rfr_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Trees": n_estimators,
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
        val = st.number_input(f"Enter value for {feature}", key=f"rfr_{feature}")
        new_input.append(val)

    if st.button("Predict with New Input"):
        prediction = model.predict([new_input])[0]
        st.success(f"Predicted {target}: {prediction:.4f}")

    if st.session_state.rfr_history:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.rfr_history)
        st.dataframe(hist_df, use_container_width=True)

    with st.expander(" What is a Random Forest Regressor?"):
        st.markdown("""
        - **Random Forest** uses many decision trees to improve prediction accuracy and reduce overfitting.
        - **n_estimators**: Number of trees in the forest.
        - **max_depth**: Depth of each tree to control complexity.
        - **R² Score**: How well predictions match actual outcomes.
        - **Error Metrics**: MAE, MSE, RMSE measure prediction differences.
        - **Scatter Plot**: Red line shows perfect prediction reference.
        """)
