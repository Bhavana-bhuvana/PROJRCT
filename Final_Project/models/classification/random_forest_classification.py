from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import plotly.express as px
def random_forest_classification_page(data):
    st.header("Random Forest Classification")

    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    # Tooltips for Feature Selection
    with st.expander("How to Select Features"):
        st.markdown("""
        - **Good Features**: Random Forest can handle both categorical and numerical features. However, it works best with a mix of both.
        - **Numeric Features**: Continuous features (e.g., age, income) are well handled.
        - **Categorical Features**: Random Forest can handle categorical features but make sure to encode them correctly.
        - **Feature Importance**: Random Forest can assess feature importance. You should focus on the most important features, especially for large datasets.
        """)

    # Train-test split slider
    st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05, key="split_ratio")
    split_ratio = st.session_state.split_ratio

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    st.subheader("Model Parameters")
    st.write(model.get_params())

    # Interactive sliders for customization
    st.subheader("Customize Hyperparameters")
    if st.button("Apply Custom Hyperparameters"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        st.session_state.classification_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1
        })

        st.success("Custom hyperparameters applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

    # Visualizing feature importances
    st.markdown("---")
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    st.write(importance_df)

    # Visualize with Plotly
    st.markdown("---")
    st.subheader("Random Forest Prediction")
    fig = px.scatter(x=X_test[features[0]], y=X_test[features[1]], color=predictions, labels={'x': features[0], 'y': features[1]}, title="Random Forest Classification")
    st.plotly_chart(fig)

    # Educational section with an expandable box
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Proportion of correctly predicted instances.
        - **Precision**: Fraction of relevant instances among the retrieved instances.
        - **Recall**: Fraction of relevant instances that have been retrieved.
        - **F1 Score**: The harmonic mean of precision and recall.
        """)
