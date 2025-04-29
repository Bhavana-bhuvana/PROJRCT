import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
salary_df = pd.read_csv("D:\PROJRCT\zero-code-ml-model\zero-code-ml-model\Salary_Data (1).csv")
x = salary_df["YearsExperience"].values  # use correct column name
y = salary_df["Salary"].values

# Plot actual data
plt.scatter(x, y, label="Actual Salary", color="blue")

# Define custom intercept and coefficient
y_intercept = 23000
co_ef = 10000  # e.g., 10,000 per year

# Generate predictions using the actual x values (not index!)
pred_sal = co_ef * x + y_intercept

# Plot predicted line
plt.plot(x, pred_sal, color="green", label="Predicted Salary (Custom)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Actual vs Predicted Salary")
plt.legend()
plt.grid(True)
plt.show()

# Print results
for i in range(len(x)):
    print(f"Actual salary = {y[i]} | Predicted salary = {pred_sal[i]} | Diff = {y[i] - pred_sal[i]}")
