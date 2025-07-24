# sklearn K-means wine dataset- Chemical analysis data for 3 different wine grape types grown in Italy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
from sklearn.datasets import load_wine
data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

# Preprocess Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_new = scaler.fit_transform(df)

# Implement K-Means Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_new)


# See how many samples per clusters
print(df['Cluster'].value_counts())       # Wine 0 has the most value counts near its centroid

# Look into characteristics of each cluster
print(df.groupby('Cluster').mean)
