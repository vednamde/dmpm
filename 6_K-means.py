import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# 1. Load and preprocess Titanic dataset
file_path = "Titanic.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.lower().str.strip()

# 2. Select numeric + useful features and clean data
df = df[['pclass', 'age', 'fare', 'sex']].dropna()
df['sex'] = LabelEncoder().fit_transform(df['sex'])  # male=1, female=0

# 3. Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 4. Determine optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 5. Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Build KMeans model with k=3 (or use the elbow plot suggestion)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# 7. Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# 8. Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', palette='Set1', data=df, s=80)
plt.title(f'K-Means Clustering with k={k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()