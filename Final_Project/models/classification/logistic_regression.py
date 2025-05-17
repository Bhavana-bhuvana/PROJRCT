import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

def logistic_regression_page(df):
    st.title("Logistic Regression")

    st.write("### Dataset Preview")
    st.write(df.head())

    # Feature and target selection
    st.write("### Feature Selection")
    features = st.multiselect("Select Feature Columns", options=df.columns.tolist())
    target = st.selectbox("Select Target Column", options=df.columns.tolist())

    if not features or not target:
        st.warning("Please select both features and a target to proceed.")
        return

    X = df[features]
    y = df[target]

    # Train-test split
    st.write("### Train-Test Split")
    test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5)
    random_state = st.number_input("Random State (for reproducibility)", value=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    # --- Metrics ---
    st.write("### Model Evaluation Metrics")
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    st.metric("Accuracy (from sklearn)", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    labels = list(np.unique(y_test))

    st.text("Rows = Actual, Columns = Predicted")
    st.write(pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                             columns=[f"Predicted {l}" for l in labels]))

    # Accuracy from Confusion Matrix
    cm_accuracy = np.trace(cm) / np.sum(cm)
    st.metric("Accuracy (from Confusion Matrix)", f"{cm_accuracy:.4f}")

    # Plot Confusion Matrix Heatmap
    z = cm
    x = [str(l) for l in labels]
    y = [str(l) for l in labels]
    z_text = [[str(val) for val in row] for row in z]

    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale="Blues")
    fig_cm.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cm)

    # --- Scatter Plot for 2D Feature Visualization ---
    if len(features) == 2:
        st.subheader("Prediction Scatter Plot (2 Features Only)")

        plot_df = X_test.copy()
        plot_df["Predicted"] = predictions
        plot_df["Actual"] = y_test.reset_index(drop=True)
        plot_df["Correct"] = plot_df["Predicted"] == plot_df["Actual"]
        plot_df["Correctness"] = plot_df["Correct"].map({True: "Correct", False: "Incorrect"})
        plot_df["Class Label"] = "Class " + plot_df["Predicted"].astype(str)

        fig = px.scatter(
            plot_df,
            x=features[0],
            y=features[1],
            color="Class Label",
            symbol="Correctness",
            hover_data=["Predicted", "Actual"],
            title="Prediction Scatter Plot (Correct vs Incorrect)"
        )
        fig.update_layout(
            legend_title="Prediction Details",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig)
    else:
        st.info("To see a 2D scatter plot, please select exactly 2 feature columns.")
