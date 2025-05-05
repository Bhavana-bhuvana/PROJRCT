import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def hierarchical_clustering_page(data):
    st.subheader("Hierarchical Clustering Analysis")

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = st.multiselect("Select features for clustering:", numeric_cols, key="hierarchical_features")

    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        return

    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(X_scaled)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = cluster_labels

    st.markdown("### Clustered Data")
    st.dataframe(data_with_clusters)
    st.success(f"Formed {n_clusters} clusters!")

    # New point input
    st.markdown("### New Data Point (No prediction possible here)")
    for feature in features:
        st.number_input(f"Enter value for {feature} (for reference only)", key="hierarchical_input_" + feature)

    # Visualize
    st.markdown("### Cluster Visualization (2D)")
    if len(features) > 2:
        x_axis = st.selectbox("X-axis", features, index=0, key="hierarchical_xaxis")
        y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0, key="hierarchical_yaxis")
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
    plt.title("Agglomerative Clustering (Linkage-Based)")
    plt.legend()
    st.pyplot(fig)

    # Dendrogram
    st.markdown("### Dendrogram")
    linkage_matrix = linkage(X_scaled, method='ward')
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=20, leaf_rotation=45., leaf_font_size=10., show_contracted=True, ax=ax2)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    st.pyplot(fig2)

    st.markdown("### How Hierarchical Clustering Works")
    st.info("""
    - **Agglomerative Clustering** is a bottom-up hierarchical approach.
    - Starts with each data point as its own cluster and **merges the closest pairs** based on a linkage criterion (e.g., **Ward** minimizes variance).
    - The result is a **dendrogram** showing how points are merged at each step.
    - Clusters are formed by cutting the dendrogram at a specific level.
    - No direct prediction for new points — it's based on precomputed distance merges.
    """)
