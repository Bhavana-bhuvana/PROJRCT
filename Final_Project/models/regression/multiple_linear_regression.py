import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import plotly.express as px

# Helper function for tooltips
def show_tooltip(tooltip_type):
    tooltips = {
        "intro": """
            **Multiple Linear Regression** is a statistical technique used to model the relationship between one dependent variable (target) and multiple independent variables (features).
            It assumes a linear relationship between the dependent and independent variables. The model tries to find the best-fitting line by adjusting the coefficients of the features.
        """,
        "technical_intro": """
            Multiple Linear Regression (MLR) assumes that the target variable is a linear combination of the independent variables. The coefficients of the features represent the influence of each feature on the target.
            It minimizes the residual sum of squares between the observed targets in the dataset and the targets predicted by the linear approximation.
        """,
        "feature": """
            Features are the independent variables that are used to predict the target variable. The right selection of features is crucial for a good model, as irrelevant features can decrease accuracy.
            You want features that have a significant relationship with the target and exclude those with little or no effect.
        """,
        "parameter": """
            - **Intercept**: The expected value of the target when all the feature values are 0.
            - **Coefficient**: The rate of change in the target variable for a one-unit change in the corresponding feature.
            Each feature has a coefficient that represents its influence on the prediction. Larger values of the coefficients indicate stronger influence.
        """,
        "robot_assistant": """
            The Robot Assistant can help you understand each step of the process. From explaining the relationship between features and target to guiding you through model selection and evaluation, the assistant provides educational support for beginners and advanced users alike.
        """
    }

    # Display the appropriate tooltip
    st.info(tooltips[tooltip_type])

def multiple_linear_regression_page(data):
    st.header("Multiple Linear Regression Model")

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

    # Train-test split slider with tooltip
    st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05, key="split_ratio", help="Adjust the ratio for the training data and testing data.")
    split_ratio = st.session_state.split_ratio

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    intercept = model.intercept_
    coefficients = model.coef_

    # Displaying educational tooltip
    show_tooltip("intro")  # Intro tooltip for MLR

    st.subheader("Model Coefficients")
    st.write("**Intercept:**", round(intercept, 4))
    coef_df = pd.DataFrame({
        "Feature": features,
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

    # Scatter plot with regression line using Plotly
    st.markdown("---")
    st.subheader("Regression Plot (Scatter + Line)")
    fig = px.scatter(x=X_test[features[0]], y=y_test, labels={'x': features[0], 'y': target}, title="Multiple Regression Fit")
    fig.add_scatter(x=X_test[features[0]], y=predictions, mode='lines', name='Prediction Line', line=dict(color='red'))
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
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.model_selection import train_test_split
# from datetime import datetime
# import plotly.express as px

# def multiple_linear_regression_page(data):
#     st.header("📊 Multiple Linear Regression")

#     if "mlr_history" not in st.session_state:
#         st.session_state.mlr_history = []

#     st.markdown("### 🎯 Feature & Target Selection")
#     with st.expander("ℹ️ What are features and target?"):
#         st.info("""
#         - **Features (X)** are the independent variables you use to predict the outcome.
#         - **Target (y)** is the dependent variable you want to predict.
#         - For Multiple Linear Regression, select **2 or more features**.
#         """)

#     features = st.multiselect("Select Feature Columns (X):", options=data.columns)
#     target = st.selectbox("Select Target Column (y):", options=data.columns)

#     if len(features) < 2 or not target or target in features:
#         st.warning("Please select at least 2 valid feature columns and a different target column.")
#         return

#     X = data[features]
#     y = data[target]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)

#     intercept = model.intercept_
#     coefficients = model.coef_

#     st.subheader("🧠 Model Coefficients")
#     st.write("**Intercept:**", round(intercept, 4))
#     coef_df = pd.DataFrame({
#         "Feature": features,
#         "Coefficient": [round(c, 4) for c in coefficients]
#     })
#     st.dataframe(coef_df, use_container_width=True)

#     st.markdown("---")
#     st.subheader("🛠 Customize Coefficients & Intercept")
#     new_intercept = st.number_input("Intercept", value=float(intercept), format="%.4f", step=0.1)
#     new_coeffs = []
#     for i, feature in enumerate(features):
#         val = st.number_input(f"Coefficient for {feature}", value=float(coefficients[i]), format="%.4f", step=0.1)
#         new_coeffs.append(val)

#     if st.button("Apply Custom Parameters"):
#         X_array = X.values
#         y_pred_custom = np.dot(X_array, new_coeffs) + new_intercept

#         r2 = r2_score(y, y_pred_custom)
#         mae = mean_absolute_error(y, y_pred_custom)
#         mse = mean_squared_error(y, y_pred_custom)
#         rmse = np.sqrt(mse)

#         st.session_state.mlr_history.append({
#             "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "Intercept": new_intercept,
#             **{f: c for f, c in zip(features, new_coeffs)},
#             "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
#         })

#         st.success("✅ Custom coefficients applied and evaluated!")

#         st.subheader("📈 Updated Performance Metrics")
#         st.metric("R² Score", f"{r2:.4f}")
#         st.metric("MAE", f"{mae:.4f}")
#         st.metric("MSE", f"{mse:.4f}")
#         st.metric("RMSE", f"{rmse:.4f}")

#     st.markdown("---")
#     st.subheader("🔍 Predict New Target Value")

#     new_data = []
#     for f in features:
#         val = st.number_input(f"Enter value for {f}", format="%.4f")
#         new_data.append(val)

#     if st.button("Predict New Value"):
#         pred = np.dot(np.array(new_data), new_coeffs) + new_intercept
#         st.success(f"🎯 Predicted Target: **{round(pred, 4)}**")

#     st.markdown("---")
#     st.subheader("📊 Interactive Scatter Plot")

#     with st.expander("ℹ️ What does this plot show?"):
#         st.info("""
#         - This plot visualizes the relationship between the actual and predicted values.
#         - Helps assess how well your model fits the data.
#         """)

#     pred_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
#     fig = px.scatter(pred_df, x="Actual", y="Predicted", title="Actual vs Predicted", trendline="ols")
#     st.plotly_chart(fig, use_container_width=True)

#     if st.session_state.mlr_history:
#         st.subheader("📜 Change History")
#         hist_df = pd.DataFrame(st.session_state.mlr_history)
#         st.dataframe(hist_df, use_container_width=True)

#     with st.expander("📘 Glossary of Metrics"):
#         st.markdown("""
#         - **Intercept**: Predicted value when all features are 0.
#         - **Coefficient**: How much the target changes with 1 unit of the feature.
#         - **R² Score**: Proportion of target variance explained by the model.
#         - **MAE**: Average absolute error.
#         - **MSE**: Mean squared error (sensitive to large errors).
#         - **RMSE**: Root mean squared error (same unit as target).
#         """)
