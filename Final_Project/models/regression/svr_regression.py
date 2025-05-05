import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import numpy as np

def svr_regression_page(data):
    st.header("Support Vector Regression (SVR)")

    features = st.multiselect("Select feature column(s):", options=data.columns)
    target = st.selectbox("Select target column:", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid features and a target column.")
        return

    # 🧠 Expandable Tooltip: Feature selection guide
    with st.expander("🔍 How to Choose Good Features"):
        st.markdown("""
        - **SVR is sensitive to feature scaling**, so numerical features are best.
        - Avoid categorical columns unless they are encoded.
        - Choose features that correlate well with the target.
        - Consider checking pairwise plots or correlation matrices before final selection.
        """)

    # Train-Test Split Slider
    test_size = st.slider("Select test size ratio:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # Custom hyperparameters
    with st.expander("⚙️ Customize SVR Parameters"):
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        C = st.slider("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Split & Train
    X = data[features]
    y = data[target]

    # Normalize features (SVR sensitive to scale)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size)

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Metrics
    st.subheader("📊 Model Performance")
    mse = mean_squared_error(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")

    # 📘 Expandable tooltip for metric explanations
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        - **Mean Squared Error (MSE)**: Penalizes large errors more than small ones. Lower is better.
        - **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and actual values.
        - **R² Score**: Proportion of variance in target explained by model. Closer to 1 is better.
        """)

    # Visualization
    st.subheader("📈 Actual vs Predicted")
    results_df = pd.DataFrame({
        "Actual": y_test_actual,
        "Predicted": y_pred
    })
    fig = px.scatter(results_df, x="Actual", y="Predicted", title="Actual vs Predicted", trendline="ols")
    st.plotly_chart(fig)

    # Optional: Predict on new input
    with st.expander("🔮 Predict with Custom Input"):
        input_data = {}
        for feature in features:
            value = st.number_input(f"Input value for {feature}:", value=float(data[feature].mean()))
            input_data[feature] = value

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler_X.transform(input_df)
            pred_scaled = model.predict(input_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
            st.success(f"Predicted value: **{pred:.4f}**")
