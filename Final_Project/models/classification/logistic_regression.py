import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff
from datetime import datetime

def logistic_regression_page(df):
    st.title("Logistic Regression Classifier")
    MODEL_KEY = "classification_logistic"

    # Initialize session state
    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = {
            "features": [],
            "target": None,
            "test_size": 0.2,
            "random_state": 42,
            "history": [],
            "last_prediction": None,
        }

    if df is None or df.empty:
        st.warning("Please upload a valid dataset to continue.")
        return

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Feature/Target Selection
    features = st.multiselect(
        "Select Feature Columns:",
        options=df.columns.tolist(),
        default=st.session_state[MODEL_KEY]["features"]
    )

    target = st.selectbox(
        "Select Target Column:",
        options=df.columns.tolist(),
        index=df.columns.get_loc(st.session_state[MODEL_KEY]["target"]) if st.session_state[MODEL_KEY]["target"] in df.columns else 0
    )

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and a target column.")
        return

    st.session_state[MODEL_KEY]["features"] = features
    st.session_state[MODEL_KEY]["target"] = target

    X = df[features]
    y = df[target]

    # Train-Test Split
    test_size = st.slider("Test Size (%)", 10, 50, int(st.session_state[MODEL_KEY]["test_size"] * 100), step=5)
    random_state = st.number_input("Random State", value=st.session_state[MODEL_KEY]["random_state"], step=1)

    st.session_state[MODEL_KEY]["test_size"] = test_size / 100
    st.session_state[MODEL_KEY]["random_state"] = random_state

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=random_state)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # Evaluation
    st.subheader("Evaluation Metrics")
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")

    # Save in history
    history_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Features": ", ".join(features),
        "Target": target
    }
    st.session_state[MODEL_KEY]["history"].append(history_entry)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    cm_fig = ff.create_annotated_heatmap(
        z=cm,
            x=list(np.unique(y_test)),
            y=list(np.unique(y_test)),
        colorscale="Blues",
        showscale=True
    )
    st.plotly_chart(cm_fig)

    # History
    if st.session_state[MODEL_KEY]["history"]:
        st.markdown("---")
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state[MODEL_KEY]["history"])
        st.dataframe(history_df, use_container_width=True)

    # Reset Option
    if st.button("Reset Session"):
        st.session_state.pop(MODEL_KEY, None)
        st.rerun()
