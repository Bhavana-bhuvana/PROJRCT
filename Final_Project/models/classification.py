import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def classification_page(data):
    st.subheader("🔹 Classification Analysis")

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    target = st.selectbox("Select target variable (must be categorical):", cat_cols)
    features = st.multiselect("Select numeric features:", num_cols)

    if not features or not target:
        st.warning("Select target and feature columns.")
        return

    X = data[features]
    y = data[target]

    y = pd.factorize(y)[0]  # Convert to numeric labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, preds)))

    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": preds}))
