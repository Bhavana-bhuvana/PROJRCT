import streamlit as st
from models.clustering.kmeans_clustering import kmeans_clustering_page
# from models.clustering.dbscan_clustering import dbscan_clustering_page
# from models.clustering.gmm_clustering import gmm_clustering_page
# from models.clustering.hierarchical_clustering import hierarchical_clustering_page

def clustering_page(model_name, data):
    # st.subheader("Select Features for Clustering")
    # features = st.multiselect("Select feature columns (X):", options=data.columns)

    # if not features:
    #     st.warning("Please select feature(s) for clustering.")
    #     return

    # X = data[features]

    if model_name == "K-Means":
      # n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10)
        kmeans_clustering_page(data)
    # elif model_name == "DBSCAN":
    #     eps = st.slider("Select epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1)
    #     min_samples = st.slider("Select min samples", min_value=2, max_value=10)
    #     dbscan_clustering_page(X, eps, min_samples)
    # elif model_name == "Gaussian Mixture Model":
    #     n_components = st.slider("Select number of components", min_value=2, max_value=10)
    #     gmm_clustering_page(X, n_components)
    # elif model_name == "Hierarchical":
    #     n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10)
    #    hierarchical_clustering_page(X, n_clusters)
    else:
        st.error("Unsupported Clustering Model")

