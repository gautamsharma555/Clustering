import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Clustering App", layout="wide")

# Title
st.title("EastWest Airlines Clustering App")

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")
    return df

# Optional: file uploader for Hugging Face
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="data")
else:
    df = load_data()

st.subheader("Raw Data")
st.write(df.head())

# Preprocessing
df1 = df.drop(['ID#'], axis=1)

# Normalize data
scaler = StandardScaler()
df_norm = scaler.fit_transform(df1)

# PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(df_norm)
pca_norm = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])

st.subheader("PCA Reduced Data")
st.write(pca_norm.head())

# Copy for clustering results
cluster_results = pca_norm.copy()

# ----------------- KMeans --------------------
st.subheader("KMeans Clustering")

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_norm)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(2, 11), wcss, marker='o')
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

k_val = st.slider("Select number of clusters for KMeans", 2, 10, 4)
kmeans = KMeans(n_clusters=k_val, random_state=42)
k_labels = kmeans.fit_predict(pca_norm)
cluster_results['KMeans_Cluster'] = k_labels

k_score = silhouette_score(pca_norm, k_labels)
st.success(f"Silhouette Score for KMeans: {k_score:.3f}")
st.write("KMeans Cluster Distribution:", np.bincount(k_labels))

# ----------------- Hierarchical --------------------
st.subheader("Hierarchical Clustering")

linkage_matrix = linkage(pca_norm, method='ward')
fig2, ax2 = plt.subplots(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=10, ax=ax2)
st.pyplot(fig2)

h_val = st.slider("Select number of clusters for Hierarchical", 2, 10, 4)
h_labels = fcluster(linkage_matrix, h_val, criterion='maxclust')
cluster_results['HCluster'] = h_labels

h_score = silhouette_score(pca_norm, h_labels)
st.success(f"Silhouette Score for Hierarchical: {h_score:.3f}")
st.write("Hierarchical Cluster Distribution:", np.bincount(h_labels))

# ----------------- DBSCAN --------------------
st.subheader("DBSCAN Clustering")

eps_val = st.slider("Select eps value for DBSCAN", 0.1, 5.0, 1.0, step=0.1)
min_samples_val = st.slider("Select min_samples for DBSCAN", 2, 20, 5)

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
db_labels = dbscan.fit_predict(pca_norm)
cluster_results['DBSCAN_Cluster'] = db_labels

# Count valid clusters (excluding noise)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
if n_clusters >= 2:
    db_score = silhouette_score(pca_norm, db_labels)
    st.success(f"Silhouette Score for DBSCAN: {db_score:.3f}")
    st.write("DBSCAN Cluster Distribution:", np.bincount(db_labels[db_labels != -1]))
else:
    st.warning("DBSCAN did not find at least 2 clusters. Silhouette Score cannot be calculated.")

# ----------------- Final Data --------------------
st.subheader("Clustered Data Preview")
st.write(cluster_results.head())

# ----------------- Visualization --------------------
st.subheader("3D PCA Cluster Plot (KMeans)")

fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111, projection='3d')
scatter = ax3.scatter(cluster_results['PC1'], cluster_results['PC2'], cluster_results['PC3'],
                      c=cluster_results['KMeans_Cluster'], cmap='tab10')
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel("PC3")
plt.title("3D PCA with KMeans Clustering")
st.pyplot(fig3)
st.subheader("3D PCA Cluster Plot (Hierarchical)")
fig4 = plt.figure(figsize=(10, 6))