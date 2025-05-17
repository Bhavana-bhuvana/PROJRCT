import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import plotly.express as px

# Helper function for tooltips
def show_tooltip(tooltip_type):
    tooltips = {
        "intro": """
            **Polynomial Regression** is an extension of linear regression that allows us to model non-linear relationships by introducing polynomial terms (squared, cubed, etc.) of the features.
            This approach fits a polynomial curve instead of a straight line, providing more flexibility in capturing complex relationships.
        """,
        "technical_intro": """
            Polynomial Regression fits the data to a polynomial equation of degree n, where n represents the highest exponent of the feature. This allows for capturing curves and non-linearities in the data that linear regression can't handle.
        """,
        "feature": """
            Features are the independent variables used to predict the target. In polynomial regression, we transform the features by adding polynomial terms (e.g., x², x³) to model non-linear relationships.
        """,
        "parameter": """
            - **Intercept**: The expected value of the target when all features are zero.
            - **Coefficient**: The impact of each feature (or transformed feature) on the target variable.
        """,
        "robot_assistant": """
            The Robot Assistant can help explain how polynomial regression works by visualizing how different degrees of the polynomial curve fit the data. It can also explain what the various hyperparameters control.
        """
    }

    # Display the appropriate tooltip
    st.info(tooltips[tooltip_type])

def polynomial_regression_page(data):
    st.header("Polynomial Regression Model")

    if "regression_history" not in st.session_state:
        st.session_state.regression_history = []

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    # Train-test split slider with tooltip
    st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05, key="split_ratio", help="Adjust the ratio for the training data and testing data.")
    split_ratio = st.session_state.split_ratio

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    # Polynomial degree slider with tooltip
    degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=2, step=1, help="Increase the polynomial degree to capture more complex relationships.")
    show_tooltip("intro")  # Intro tooltip for Polynomial Regression

    # Transform features to include polynomial terms
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    predictions = model.predict(X_poly_test)

    intercept = model.intercept_
    coefficients = model.coef_

    st.subheader("Model Coefficients")
    st.write("**Intercept:**", round(intercept, 4))
    coef_df = pd.DataFrame({
        "Feature": ['1'] + [f"{feature}^{i}" for feature in features for i in range(1, degree+1)],
        "Coefficient": [round(c, 4) for c in coefficients]
    })
    st.dataframe(coef_df, use_container_width=True)

    # Interactive sliders for coefficient and intercept customization
    st.subheader("Customize Coefficients & Intercept")
    new_intercept = st.number_input("Intercept", value=float(intercept), format="%.4f", step=0.1)
    new_coeffs = []
    for i, feature in enumerate(features):
        coeff = st.number_input(f"Coefficient for {feature}", value=float(coefficients[i]), format="%.4f", step=0.1)
        new_coeffs.append(coeff)

    if st.button("Apply Custom Parameters"):
        X_poly_array = X.values
        y_pred_custom = np.dot(X_poly_array, new_coeffs) + new_intercept

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
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.regression_history)
        st.dataframe(hist_df, use_container_width=True)

    # Predicting new data points
    st.markdown("---")
    st.subheader("Predict New Data Point")

    new_data = {}
    for feature in features:
        val = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.4f")
        new_data[feature] = val

    if st.button("Predict Target"):
        input_array = np.array([list(new_data.values())])

        # Choose which coefficients to use
        use_custom = st.checkbox("Use custom coefficients for prediction", value=False)
        if use_custom:
            prediction = np.dot(input_array, new_coeffs) + new_intercept
        else:
            prediction = model.predict(input_array)

        st.success(f"Predicted {target}: {round(prediction[0], 4)}")

    # Scatter plot with polynomial regression curve using Plotly
    st.markdown("---")
    st.subheader("Polynomial Regression Plot (Scatter + Curve)")
    fig = px.scatter(x=X_test[features[0]], y=y_test, labels={'x': features[0], 'y': target}, title="Polynomial Regression Fit")
    sorted_idx = X_test[features[0]].argsort()
    sorted_X = X_test[features[0]].values[sorted_idx]
    poly_predictions = model.predict(poly.transform(sorted_X.reshape(-1, 1)))

    fig.add_scatter(x=sorted_X, y=poly_predictions, mode='lines', name='Prediction Curve', line=dict(color='red'))
    st.plotly_chart(fig)

    # Educational section with an expandable box
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Intercept**: Expected value of target when all features are 0.
        - **Coefficient**: Change in target for one-unit change in the feature.
        - **R² Score**: Fraction of variance explained by the model (1 is perfect).
        - **MAE**: Average absolute difference between predictions and true values.
        - **MSE**: Same as MAE but squared – penalizes larger errors more.
        - **RMSE**: Square root of MSE – interpretable in same units as target.
        """)



# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from datetime import datetime
# import plotly.express as px


# def polynomial_regression_page(data):
#     st.header(" Polynomial Regression")

#     if "poly_history" not in st.session_state:
#         st.session_state.poly_history = []

#     st.markdown("###  Feature & Target Selection")
#     with st.expander("ℹ Why only 1 feature in Polynomial Regression?"):
#         st.info("""
#         - Polynomial Regression is an extension of Linear Regression.
#         - We typically use **1 feature** to visualize the curve clearly.
#         - The algorithm creates new features by raising the selected feature to higher powers.
#         """)

#     feature = st.selectbox("Select a single Feature (X):", options=data.columns)
#     target = st.selectbox("Select Target Column (y):", options=data.columns)

#     if not feature or not target or feature == target:
#         st.warning("Please select different columns for feature and target.")
#         return

#     degree = st.slider("Select Polynomial Degree:", 2, 10, 2)

#     X = data[[feature]].values
#     y = data[target].values

#     poly = PolynomialFeatures(degree=degree)
#     X_poly = poly.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)

#     intercept = model.intercept_
#     coefficients = model.coef_

#     st.subheader(" Model Coefficients")
#     st.write("**Intercept:**", round(intercept, 4))
#     coef_names = poly.get_feature_names_out([feature])
#     coef_df = pd.DataFrame({
#         "Term": coef_names,
#         "Coefficient": [round(c, 4) for c in coefficients]
#     })
#     st.dataframe(coef_df, use_container_width=True)

#     st.subheader("🛠 Customize Coefficients")
#     new_intercept = st.number_input("Intercept", value=float(intercept), format="%.4f", step=0.1)
#     new_coeffs = []
#     for i, name in enumerate(coef_names):
#         val = st.number_input(f"Coefficient for {name}", value=float(coefficients[i]), format="%.4f", step=0.1)
#         new_coeffs.append(val)

#     if st.button("Apply Custom Polynomial"):
#         y_pred_custom = np.dot(poly.transform(X), new_coeffs)
#         r2 = r2_score(y, y_pred_custom)
#         mae = mean_absolute_error(y, y_pred_custom)
#         mse = mean_squared_error(y, y_pred_custom)
#         rmse = np.sqrt(mse)

#         st.session_state.poly_history.append({
#             "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "Intercept": new_intercept,
#             **{name: c for name, c in zip(coef_names, new_coeffs)},
#             "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
#         })

#         st.success(" Custom polynomial applied and evaluated!")

#         st.subheader(" Updated Performance Metrics")
#         st.metric("R² Score", f"{r2:.4f}")
#         st.metric("MAE", f"{mae:.4f}")
#         st.metric("MSE", f"{mse:.4f}")
#         st.metric("RMSE", f"{rmse:.4f}")

#     st.subheader(" Predict New Target Value")
#     x_val = st.number_input(f"Enter value for {feature}", format="%.4f")
#     if st.button("Predict for New Value"):
#         x_poly = poly.transform([[x_val]])
#         pred = np.dot(x_poly, new_coeffs)
#         st.success(f" Predicted Target: **{round(pred[0], 4)}**")

#     st.subheader(" Plot Polynomial Curve")
#     with st.expander("ℹ Plot Info"):
#         st.info("This plot shows the fitted polynomial curve against actual data points.")

#     sorted_indices = X[:, 0].argsort()
#     X_sorted = X[sorted_indices]
#     y_sorted = y[sorted_indices]
#     y_pred_sorted = np.dot(poly.transform(X_sorted), new_coeffs)

#     plot_df = pd.DataFrame({
#         feature: X_sorted.flatten(),
#         "Actual": y_sorted,
#         "Predicted": y_pred_sorted
#     })
#     fig = px.scatter(plot_df, x=feature, y="Actual", title="Polynomial Fit", labels={"value": target})
#     fig.add_scatter(x=plot_df[feature], y=plot_df["Predicted"], mode='lines', name="Fitted Curve")
#     st.plotly_chart(fig, use_container_width=True)

#     if st.session_state.poly_history:
#         st.subheader(" Change History")
#         hist_df = pd.DataFrame(st.session_state.poly_history)
#         st.dataframe(hist_df, use_container_width=True)

#     with st.expander(" Glossary of Terms"):
#         st.markdown("""
#         - **Polynomial Features**: New features like x², x³ added from original x.
#         - **Intercept**: Model output when all features = 0.
#         - **R²**: Goodness of fit (1.0 is perfect).
#         - **MAE/MSE/RMSE**: Measures of prediction error.
#         """)
