# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# st.title("🔗 Clustering (KMeans)")

# if 'data' in st.session_state:
#     data = st.session_state['data']
#     st.write("### Data Preview")
#     st.dataframe(data)

#     st.subheader("Feature Selection for Clustering")
#     features = st.multiselect("Select features:", data.columns)

#     if features:
#         X = data[features]

#         k = st.slider("Number of Clusters (k):", 2, 10, 3)

#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(X)
#         labels = kmeans.labels_
#         data['Cluster'] = labels

#         st.write("### Clustered Data")
#         st.dataframe(data)

#         if len(features) >= 2:
#             st.subheader("Cluster Visualization")
#             fig, ax = plt.subplots()
#             scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
#             legend = ax.legend(*scatter.legend_elements(), title="Clusters")
#             ax.add_artist(legend)
#             ax.set_xlabel(features[0])
#             ax.set_ylabel(features[1])
#             st.pyplot(fig)
#         else:
#             st.info("Select at least two features to visualize clusters.")
#     else:
#         st.warning("Please select features for clustering.")
# else:
#     st.warning("Upload a dataset from the main dashboard.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("🌀 Clustering with K-Means")

if 'data' in st.session_state:
    data = st.session_state['data']
    st.write("### 📊 Data Preview")
    st.dataframe(data)

    st.subheader("🔍 Select Features for Clustering")
    features = st.multiselect("Pick features to cluster on", data.columns)

    if len(features) >= 2:
        X = data[features]

        # Handle categorical data
        if X.select_dtypes(include='object').shape[1] > 0:
            st.warning("⚠️ Please select only numeric features for clustering.")
        else:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            st.markdown("### 🔢 Choose Number of Clusters (K)")
            k = st.slider("K (Number of clusters)", min_value=2, max_value=10, value=3)

            # Fit KMeans
            model = KMeans(n_clusters=k, random_state=42)
            clusters = model.fit_predict(X_scaled)

            data['Cluster'] = clusters
            st.success("✅ Clustering Complete!")

            st.markdown("### 📈 Cluster Visualization (PCA 2D Plot)")

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X_scaled)
            reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
            reduced_df["Cluster"] = clusters

            fig, ax = plt.subplots()
            sns.scatterplot(data=reduced_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
            st.pyplot(fig)

            st.markdown("### 📁 Clustered Data")
            st.dataframe(data)

            # Download
            csv = data.to_csv(index=False).encode()
            st.download_button("⬇️ Download Clustered Data", csv, "clustered_data.csv", "text/csv")

            # Elbow method
            with st.expander("📐 Find Optimal K (Elbow Method)"):
                distortions = []
                K_range = range(1, 11)
                for i in K_range:
                    km = KMeans(n_clusters=i, random_state=42)
                    km.fit(X_scaled)
                    distortions.append(km.inertia_)

                fig2, ax2 = plt.subplots()
                plt.plot(K_range, distortions, 'bo-')
                plt.xlabel("K")
                plt.ylabel("Inertia (Distortion)")
                plt.title("Elbow Method For Optimal K")
                st.pyplot(fig2)

            with st.expander("ℹ️ What is K-Means Clustering?"):
                st.markdown("""
                - **K-Means** is an *unsupervised learning algorithm* that groups data into K clusters.
                - It minimizes the distance between points in a cluster and the cluster center.
                - The algorithm:
                    1. Randomly assigns K cluster centers
                    2. Assigns each point to the nearest center
                    3. Updates centers to the average of their cluster
                    4. Repeats until convergence
                """)
    else:
        st.info("Please select at least two numeric features.")
else:
    st.warning("⚠️ Please upload and clean data first.")
