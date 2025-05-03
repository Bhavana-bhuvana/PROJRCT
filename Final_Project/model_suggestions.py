# model_suggestions.py
import pandas as pd

def suggest_models(data: pd.DataFrame) -> list:
    suggestions = []
    if data is None or data.empty:
        return suggestions

    numeric_cols = data.select_dtypes(include="number").shape[1]
    object_cols = data.select_dtypes(include="object").shape[1]
    unique_targets = data.nunique().max()

    if numeric_cols >= 2:
        suggestions.append("Regression")
        suggestions.append("Clustering")

    if object_cols >= 1 or unique_targets <= 20:
        suggestions.append("Classification")

    return suggestions
