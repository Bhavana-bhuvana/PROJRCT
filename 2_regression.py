# pages/2_regression.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("📈 Linear Regression Model")

# Step 1: Load Data
if "cleaned_data" not in st.session_state:
    st.warning("⚠️ Please upload and clean your data in Page 1 first.")
    st.stop()

df = st.session_state.cleaned_data

# Step 2: Column Selection
st.markdown("### 🔧 Select Variables")
independent_var = st.selectbox("Select Independent Variable (X)", df.columns)
dependent_var = st.selectbox("Select Dependent Variable (y)", df.columns)

if independent_var == dependent_var:
    st.error("❌ X and y cannot be the same")
    st.stop()

# Step 3: Model Training
X = df[[independent_var]]
y = df[dependent_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 4: Coefficient and Score
st.markdown("### 📊 Model Summary")
st.write(f"**Coefficient (slope)**: {model.coef_[0]:.4f}")
st.write(f"**Intercept**: {model.intercept_:.4f}")
st.write(f"**R² Score**: {model.score(X_test, y_test):.4f}")

# Save model or predictions if needed
st.session_state.regression_results = {
    "coefficient": model.coef_[0],
    "intercept": model.intercept_,
    "score": model.score(X_test, y_test),
}

# Step 5: Plotting
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label="Actual")
ax.plot(X_test, y_pred, color='red', label="Predicted Line")
ax.set_xlabel(independent_var)
ax.set_ylabel(dependent_var)
ax.legend()
st.pyplot(fig)
