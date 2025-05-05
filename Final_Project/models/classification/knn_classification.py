import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def knn_classification_page(data):
    st.header("K-Nearest Neighbors Classification")

    if "knn_history" not in st.session_state:
        st.session_state.knn_history = []

    # Feature selection: user can select multiple features
    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    # Educating the user about feature selection
    with st.expander("Why feature selection matters?"):
        st.markdown("""
        - **Selecting 2 features**: When you select two features, we can visualize the classification boundaries using a 2D plot. This is helpful for understanding how KNN decision boundaries work.
        - **Selecting more than 2 features**: KNN can handle multiple features, but it is harder to visualize the decision boundary in higher dimensions. The model will still classify based on proximity, but visual interpretation may not be as clear.
        - **Feature importance**: In KNN, all features are treated equally in distance calculation. Thus, choosing the most relevant features can improve model performance.
        """)

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model customization: Set K value for KNN
    k_value = st.slider("Select number of neighbors (K):", min_value=1, max_value=20, value=5)

    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.subheader(f"Model Accuracy (K={k_value})")
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    st.markdown("---")
    st.subheader(" Customize KNN Model")

    # Editable fields for custom K
    new_k_value = st.number_input("Custom K Value", value=k_value, min_value=1, max_value=20)

    if st.button("Apply Custom K Value"):
        custom_model = KNeighborsClassifier(n_neighbors=new_k_value)
        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)

        custom_accuracy = accuracy_score(y_test, custom_predictions)
        st.success(f"Custom KNN model applied with accuracy: {custom_accuracy * 100:.2f}%")

        st.subheader("Updated Performance")
        st.metric("Accuracy", f"{custom_accuracy * 100:.2f}%")

        # Save to session state for history tracking
        st.session_state.knn_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "K Value": new_k_value,
            "Accuracy": custom_accuracy
        })

    if st.session_state.knn_history:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.knn_history)
        st.dataframe(hist_df, use_container_width=True)

    with st.expander(" What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: The proportion of correct predictions out of total predictions.
        - **K Value**: The number of neighbors to consider for classification.
        """)

    # Predicting new data points
    st.markdown("---")
    st.subheader("Predict New Data Point")

    new_data = {}
    for feature in features:
        val = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.4f")
        new_data[feature] = val

    if st.button("Predict Target"):
        input_array = np.array([list(new_data.values())])
        
        # Predict using the custom or original KNN model
        use_custom_knn = st.checkbox("Use custom KNN model for prediction", value=False)
        if use_custom_knn:
            custom_knn_model = KNeighborsClassifier(n_neighbors=new_k_value)
            custom_knn_model.fit(X_train, y_train)
            prediction = custom_knn_model.predict(input_array)
        else:
            prediction = model.predict(input_array)

        st.success(f"Predicted {target}: {prediction[0]}")

    # Visualization: Confusion Matrix
    st.markdown("---")
    st.subheader("Confusion Matrix")

    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
    st.dataframe(confusion_matrix, use_container_width=True)

    # Plot confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Interactive Plot (optional for feature-based classification)
    st.markdown("---")
    st.subheader("Feature Plot")

    if len(features) == 2:  # Works best with 2 features for visualization
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_test[features[0]], X_test[features[1]], c=predictions, cmap='viridis')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("Feature Scatter Plot (KNN Predictions)")
        fig.colorbar(scatter)
        st.pyplot(fig)
    elif len(features) > 2:
        st.warning("For visualizing the decision boundary, it is recommended to select only 2 features.")
