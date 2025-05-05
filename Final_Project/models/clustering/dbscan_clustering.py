import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

def dbscan_clustering_page(data):
    st.subheader("DBSCAN Clustering Analysis")

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = st.multiselect("Select features for clustering:", numeric_cols, key="dbscan_features")

    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        return

    eps = st.slider("Epsilon (Radius of Neighborhood)", 0.1, 5.0, 0.5, step=0.1)
    min_samples = st.slider("Minimum Samples per Cluster", 1, 10, 5)

    # Scale the selected data
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = cluster_labels

    st.markdown("### Clustered Data")
    st.dataframe(data_with_clusters)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    st.success(f"Formed {n_clusters} clusters (noise labeled as -1)")

    # Predict new point cluster (approximate)
    st.markdown("### Predict Cluster for a New Data Point")
    new_data = {}
    for feature in features:
        new_data[feature] = st.number_input(f"Enter value for {feature}", key="dbscan_input_" + feature)

    show_new_point = st.button("Estimate Cluster for New Data Point (approximate)", key="dbscan_predict_btn")

    new_point_original = None
    predicted_cluster = None

    if show_new_point:
        new_point_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_point_df)
        distances = np.linalg.norm(X_scaled - new_scaled, axis=1)
        neighbors = distances < eps
        if np.sum(neighbors) >= min_samples:
            predicted_cluster = cluster_labels[neighbors].max()
        else:
            predicted_cluster = -1
        new_point_original = new_point_df[features].values[0]
        new_point_df["Estimated Cluster"] = predicted_cluster
        st.success(f"Estimated Cluster: {predicted_cluster}")
        st.dataframe(new_point_df)

    # Visualization
    st.markdown("### Cluster Visualization (2D)")
    if len(features) > 2:
        x_axis = st.selectbox("X-axis", features, index=0, key="dbscan_xaxis")
        y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0, key="dbscan_yaxis")
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

    plt.title("DBSCAN Cluster Visualization (Density-based Clustering)")
    plt.legend()
    st.pyplot(fig)

    # Educational explanation
    st.markdown("### How DBSCAN Clusters Data")
    st.info("""
    - **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) clusters data based on density, not centroids.
    - It defines clusters as areas of high point density, using:
      - `ε (eps)`: maximum distance between neighbors.
      - `min_samples`: minimum number of points to form a dense region (core point).
    - Points near core points are added to the cluster.
    - Points too far from any core are labeled as **noise** (Cluster -1).
    - This method works well for irregularly shaped clusters and outlier detection.
    """)
