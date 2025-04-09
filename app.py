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

st.set_page_config(page_title="Clustering App", layout="wide")

# Title
st.title("EastWest Airlines Clustering App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("EastWestAirlines.xlsx", sheet_name="data")
    return df

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

# ----------------- KMeans --------------------
st.subheader("KMeans Clustering")

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_norm)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
st.pyplot(plt)

k_val = st.slider("Select number of clusters for KMeans", 2, 10, 4)
kmeans = KMeans(n_clusters=k_val, random_state=42)
k_labels = kmeans.fit_predict(pca_norm)
pca_norm['KMeans_Cluster'] = k_labels

k_score = silhouette_score(pca_norm.iloc[:, :-1], k_labels)
st.success(f"Silhouette Score for KMeans: {k_score:.3f}")

# ----------------- Hierarchical --------------------
st.subheader("Hierarchical Clustering")

linkage_matrix = linkage(pca_norm.iloc[:, :-1], method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=10)
st.pyplot(plt)

h_val = st.slider("Select number of clusters for Hierarchical", 2, 10, 4)
h_labels = fcluster(linkage_matrix, h_val, criterion='maxclust')
pca_norm['HCluster'] = h_labels

h_score = silhouette_score(pca_norm.iloc[:, :-2], h_labels)
st.success(f"Silhouette Score for Hierarchical: {h_score:.3f}")

# ----------------- DBSCAN --------------------
st.subheader("DBSCAN Clustering")

eps_val = st.slider("Select eps value for DBSCAN", 0.1, 5.0, 1.0, step=0.1)
min_samples_val = st.slider("Select min_samples for DBSCAN", 2, 20, 5)

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
db_labels = dbscan.fit_predict(pca_norm.iloc[:, :-2])
pca_norm['DBSCAN_Cluster'] = db_labels

# Check if at least 2 clusters are found (excluding noise -1)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
if n_clusters >= 2:
    db_score = silhouette_score(pca_norm.iloc[:, :-3], db_labels)
    st.success(f"Silhouette Score for DBSCAN: {db_score:.3f}")
else:
    st.warning("DBSCAN did not find at least 2 clusters. Silhouette Score cannot be calculated.")

# ----------------- Final Data --------------------
st.subheader("Clustered Data Preview")
st.write(pca_norm.head())

# ----------------- Visualization --------------------
st.subheader("3D PCA Cluster Plot (KMeans)")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_norm['PC1'], pca_norm['PC2'], pca_norm['PC3'], 
                     c=pca_norm['KMeans_Cluster'], cmap='tab10')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA with KMeans Clustering")
st.pyplot(fig)
