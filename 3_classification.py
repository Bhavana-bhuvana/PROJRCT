import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.title("Classification Playground")

if 'data' in st.session_state:
    data = st.session_state['data']
    st.write("###  Data Preview")
    st.dataframe(data)

    st.subheader(" Select Features and Target")
    features = st.multiselect("Choose features (X):", data.columns)
    target = st.selectbox("Choose target (y):", data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Encode categorical target
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Train-Test Split
        test_size = st.slider(" Test size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        # Model Selection
        st.subheader(" Choose Classification Model")
        model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"])

        if model_name == "Logistic Regression":
            with st.expander("ℹ What is Logistic Regression?"):
                st.markdown("Logistic Regression is used for binary classification like yes/no, spam/not spam, etc.")
            c = st.slider("C (Regularization strength)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Max Iterations", 100, 1000, 300)
            model = LogisticRegression(C=c, max_iter=max_iter)

        elif model_name == "Decision Tree":
            with st.expander(" What is a Decision Tree?"):
                st.markdown("A Decision Tree splits data based on feature conditions to classify observations.")
            max_depth = st.slider("Max Depth", 1, 20, 5)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

        elif model_name == "Random Forest":
            with st.expander(" What is Random Forest?"):
                st.markdown("Random Forest combines many decision trees to improve accuracy and reduce overfitting.")
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        elif model_name == "SVM":
            with st.expander(" What is SVM?"):
                st.markdown("Support Vector Machine finds the best boundary to separate classes.")
            c = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVC(C=c, kernel=kernel)

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader(" Model Evaluation")

        acc = accuracy_score(y_test, y_pred)
        st.success(f" Accuracy: {acc:.2f}")

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("###  Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # View Predictions
        st.markdown("###  View Predictions")
        preds_df = X_test.copy()
        preds_df['Actual'] = y_test
        preds_df['Predicted'] = y_pred
        st.dataframe(preds_df)

        # Download Predictions
        csv = preds_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Predictions", csv, "predictions.csv", "text/csv")

        # Save model
        if st.button(" Save Trained Model"):
            joblib.dump(model, f"{model_name.replace(' ', '_')}_model.pkl")
            st.success(f"{model_name} model saved successfully!")
    else:
        st.info("Please select both features and a target.")
else:
    st.warning(" Upload a dataset first in the main page.")
