import streamlit as st
from models.clustering.kmeans_clustering import kmeans_clustering_page
from models.clustering.dbscan_clustering import dbscan_clustering_page
from models.clustering.gmm_clustering import gmm_clustering_page
from models.clustering.hierarchical_clustering import hierarchical_clustering_page

def clustering_page(model_name, data):
    # st.subheader("Select Features for Clustering")
    # features = st.multiselect("Select feature columns (X):", options=data.columns)

    # if not features:
    #     st.warning("Please select feature(s) for clustering.")
    #     return

    # X = data[features]

    if model_name == "K-Means":
        kmeans_clustering_page(data)
    elif model_name == "DBSCAN":
        dbscan_clustering_page(data)
    elif model_name == "Gaussian Mixture Model":
        gmm_clustering_page(data)
    elif model_name == "Hierarchical":
      hierarchical_clustering_page(data)
    else:
        st.error("Unsupported Clustering Model")

