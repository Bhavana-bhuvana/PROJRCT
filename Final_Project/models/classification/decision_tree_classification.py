from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt


def decision_tree_classification_page(data):
    st.header("Decision Tree Classification")

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
        - **Good Features**: Decision Trees work well with both categorical and numerical features. However, categorical features should be properly encoded (one-hot encoding or label encoding).
        - **Numerical Features**: Continuous features are ideal for decision trees.
        - **Categorical Features**: Can be handled easily by Decision Trees, but ensure proper encoding to avoid errors.
        - **Feature Selection**: Overfitting can occur if you use too many irrelevant features. Focus on meaningful and relevant features.
        """)

    # Train-test split slider
    st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05, key="split_ratio")
    split_ratio = st.session_state.split_ratio

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    # Decision Tree Model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')

    st.subheader("Model Parameters")
    st.write(model.get_params())

    # Interactive sliders for customization
    st.subheader("Customize Hyperparameters")
    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=5, step=1)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)

    if st.button("Apply Custom Hyperparameters"):
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary')
        recall = recall_score(y_test, predictions, average='binary')
        f1 = f1_score(y_test, predictions, average='binary')

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

    # Visualizing Feature Importance
    st.markdown("---")
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    st.write(importance_df)

    # Visualize the Decision Tree using Plotly
    st.markdown("---")
    st.subheader("Decision Tree Prediction Visualization")
    fig = px.scatter(x=X_test[features[0]], y=X_test[features[1]], color=predictions, labels={'x': features[0], 'y': features[1]}, title="Decision Tree Classification")
    st.plotly_chart(fig)

    # Educational section with an expandable box
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Proportion of correctly predicted instances.
        - **Precision**: Fraction of relevant instances among the retrieved instances.
        - **Recall**: Fraction of relevant instances that have been retrieved.
        - **F1 Score**: The harmonic mean of precision and recall. A higher score indicates better model performance.
        """)

    # Visualizing the decision tree structure (optional)
    if st.button("Show Decision Tree Diagram"):
        from sklearn.tree import plot_tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=features, class_names=model.classes_, ax=ax)
        st.pyplot(fig)
