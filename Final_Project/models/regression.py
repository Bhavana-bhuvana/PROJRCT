import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score 
from datetime import datetime

# def regression_page(data):
#     st.header(" Regression Analysis")

#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     target = st.selectbox("Select target variable:", numeric_cols)
#     features = st.multiselect("Select feature columns:", [col for col in numeric_cols if col != target])

#     if not features or not target:
#         st.warning("Select both target and features to continue.")
#         return

#     X = data[features]
#     y = data[target]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     st.session_state.model_data = {
#                 "model_type": "regression",
#                 "X_test": X_test,
#                 "y_test": y_test,
#                 "predictions": predictions,
#                 "intercept": model.intercept_,
#                 "coefficients": model.coef_,
#                 "feature_columns": features,
#                 "data": data
#             }



#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, predictions)
#     r2_score_value = model.score(X_test, y_test)

#     col1, col2, col3 = st.columns(3)
#     col1.metric("RMSE", f"{rmse:.4f}")
#     col2.metric("MAE", f"{mae:.4f}")
#     col3.metric("R² Score", f"{r2_score_value:.4f}")

#     st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}))
#*************************************************************************************************

def regression_page(data):
    st.header("Regression Model")

    if "regression_history" not in st.session_state:
        st.session_state.regression_history = []

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    intercept = model.intercept_
    coefficients = model.coef_

    st.subheader("Initial Model Coefficients")
    st.write("**Intercept:**", round(intercept, 4))
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": [round(c, 4) for c in coefficients]
    })
    st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.subheader(" Customize Coefficients & Intercept")

    # Editable fields
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

        st.session_state.regression_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Intercept": new_intercept,
            **{f: c for f, c in zip(features, new_coeffs)},
            "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
        })

        st.success("Custom coefficients applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

    if st.session_state.regression_history:
        st.markdown("---")
        st.subheader(" Change History")
        hist_df = pd.DataFrame(st.session_state.regression_history)
        st.dataframe(hist_df, use_container_width=True)

    with st.expander(" What do these metrics mean?"):
        st.markdown("""
        - **Intercept**: Expected value of target when all features are 0.
        - **Coefficient**: Change in target for one-unit change in the feature.
        - **R² Score**: Fraction of variance explained by the model (1 is perfect).
        - **MAE**: Average absolute difference between predictions and true values.
        - **MSE**: Same as MAE but squared – penalizes larger errors more.
        - **RMSE**: Square root of MSE – interpretable in same units as target.
        """)
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}))


