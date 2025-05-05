import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

def gmm_clustering_page(data):
    st.subheader("Gaussian Mixture Model Clustering")

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = st.multiselect("Select features for clustering:", numeric_cols, key="gmm_features")

    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        return

    n_components = st.slider("Number of Clusters (Gaussian Components)", 2, 10, 3)

    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    cluster_labels = gmm.predict(X_scaled)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = cluster_labels

    st.markdown("### Clustered Data")
    st.dataframe(data_with_clusters)
    st.success(f"Formed {n_components} soft clusters using GMM!")

    # Predict new point
    st.markdown("### Predict Cluster for a New Data Point")
    new_data = {}
    for feature in features:
        new_data[feature] = st.number_input(f"Enter value for {feature}", key="gmm_input_" + feature)

    show_new_point = st.button("Predict Cluster for New Data Point", key="gmm_predict_btn")

    new_point_original = None
    predicted_cluster = None

    if show_new_point:
        new_point_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_point_df)
        probs = gmm.predict_proba(new_scaled)
        predicted_cluster = np.argmax(probs)
        new_point_original = new_point_df[features].values[0]
        new_point_df["Predicted Cluster"] = predicted_cluster
        st.success(f"The new data point most likely belongs to cluster {predicted_cluster} (with probability {probs[0][predicted_cluster]:.2f})")
        st.dataframe(new_point_df)

    # Visualization
    st.markdown("### Cluster Visualization (2D)")
    if len(features) > 2:
        x_axis = st.selectbox("X-axis", features, index=0, key="gmm_xaxis")
        y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0, key="gmm_yaxis")
    else:
        x_axis, y_axis = features[0], features[1]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=data_with_clusters[x_axis],
        y=data_with_clusters[y_axis],
        hue=data_with_clusters["Cluster"],
        palette="tab10",
        ax=ax
    )

    if new_point_original is not None:
        x_idx = features.index(x_axis)
        y_idx = features.index(y_axis)
        ax.scatter(
            new_point_original[x_idx],
            new_point_original[y_idx],
            s=150, c='red', marker='D', label='New Point'
        )

    plt.title("GMM Clustering (Elliptical Gaussian Distributions)")
    plt.legend()
    st.pyplot(fig)

    st.markdown("### How Gaussian Mixture Model Works")
    st.info("""
    - **GMM** models clusters as multivariate **Gaussian distributions**.
    - It assigns **probabilities** to each point belonging to each cluster (soft clustering).
    - Based on Expectation-Maximization (EM) algorithm:
      - Estimates Gaussian means and covariances.
    - Great for data with **elliptical clusters** and overlapping distributions.
    - Prediction outputs the **most probable** cluster for a new data point.
    """)
