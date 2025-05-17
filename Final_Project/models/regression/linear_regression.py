import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def linear_regression_page(data):
    st.header("Linear Regression Model")

    if "regression_history" not in st.session_state:
        st.session_state.regression_history = []

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)
    if data is None or data.empty or data.columns.empty:
        st.warning(" Please upload and preprocess a dataset with valid columns before using this model.")
        return

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    X = data[features]
    y = data[target]

    # Train-test split
    with st.expander("Train-Test Split Explanation"):
        st.markdown("""
        Adjust the ratio to control how much data is used to train vs. test the model.
        A high test split can reveal overfitting issues.
        """)
    split_ratio = st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    intercept = model.intercept_
    coefficients = model.coef_

    # Model coefficients
    st.subheader("Model Coefficients")
    st.write("**Intercept:**", round(intercept, 4))
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": [round(c, 4) for c in coefficients]
    })
    st.dataframe(coef_df, use_container_width=True)

    # Underfitting/Overfitting check
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    st.subheader("Performance Summary")
    st.metric("Train R²", f"{r2_train:.4f}")
    st.metric("Test R²", f"{r2_test:.4f}")
    
    if r2_train > 0.9 and r2_test < 0.6:
        st.warning(" Possible Overfitting: Model performs well on training but poorly on test data.")
    elif r2_train < 0.5 and r2_test < 0.5:
        st.warning(" Possible Underfitting: Model is not learning the pattern from training data.")

    # Interactive coefficient customization
    st.subheader("Customize Coefficients & Intercept")
    new_intercept = st.number_input("Intercept", value=float(intercept), format="%.4f", step=0.1)
    new_coeffs = []
    for i, feature in enumerate(features):
        coeff = st.number_input(f"Coefficient for {feature}", value=float(coefficients[i]), format="%.4f", step=0.1)
        new_coeffs.append(coeff)

    if st.button("Apply Custom Parameters"):
        X_array = X.values
        y_pred_custom = np.dot(X_array, new_coeffs) + new_intercept

        r2 = r2_score(y, y_pred_custom)
        mae = mean_absolute_error(y, y_pred_custom)
        mse = mean_squared_error(y, y_pred_custom)
        rmse = np.sqrt(mse)

        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Intercept": new_intercept,
            **{f: c for f, c in zip(features, new_coeffs)},
            "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
        }

        # Avoid duplicate entries based on timestamp
        if result not in st.session_state.regression_history:
            st.session_state.regression_history.append(result)

        st.success("Custom coefficients applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

    if st.session_state.regression_history:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.regression_history)
        st.dataframe(hist_df, use_container_width=True)

    # Predicting new data points
    st.markdown("---")
    st.subheader("Predict New Data Point")

    new_data = {}
    for feature in features:
        val = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.4f", key=f"predict_{feature}")
        new_data[feature] = val

    if st.button("Predict Target"):
        input_array = np.array([list(new_data.values())])
        use_custom = st.checkbox("Use custom coefficients for prediction", value=False)
        if use_custom:
            prediction = np.dot(input_array, new_coeffs) + new_intercept
        else:
            prediction = model.predict(input_array)
        st.success(f"Predicted {target}: {round(prediction[0], 4)}")

    # Plotting
    st.markdown("---")
    st.subheader("Regression Plot (Scatter + Line)")
    fig = px.scatter(x=X_test[features[0]], y=y_test, labels={'x': features[0], 'y': target}, title="Regression Fit")
    fig.add_scatter(x=X_test[features[0]], y=test_pred, mode='lines', name='Prediction Line', line=dict(color='red'))
    st.plotly_chart(fig)

    # Correlation heatmap
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    with st.expander("Show correlation between features and target"):
        corr = data[[*features, target]].corr()
        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

    # Educational metrics info
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Intercept**: Predicted value when all features are zero.
        - **Coefficient**: Effect of one-unit increase in a feature on the target.
        - **R² Score**: Fraction of variance explained (1 is perfect).
        - **MAE**: Average of absolute prediction errors.
        - **MSE**: Average of squared prediction errors.
        - **RMSE**: Square root of MSE – same units as target.
        - **Overfitting**: High train performance, poor test performance.
        - **Underfitting**: Poor performance on both train and test.
        """)
