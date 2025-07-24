# Principal Component Analysis, World Happiness Report: data on happiness scores, GDP, social support, freedom, corruption, etc.

import pandas as pd
import matplotlib.pyplot as plt

# load dataset
url = "https://raw.githubusercontent.com/carogaltier/world-happiness-report/main/WHR_2020_2024.csv"
df=pd.read_csv(url)

print(df.head())
print(df.columns)

# Filter for past year
df_past= df[df['Year'] == df['Year'].max()].reset_index(drop=True)

# Select optimal features
features = df_past[['Explained by: Log GDP per capita',
               'Explained by: Social support',
               'Explained by: Healthy life expectancy',
               'Explained by: Freedom to make life choices',
               'Explained by: Generosity',
               'Explained by: Perceptions of corruption']]
countries = df_past['Country Name']

features_filled = features.fillna(features.mean())
print(features_filled.isnull().sum())

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_scaled = scaler.fit_transform(features_filled)

# Apply PCA to reduce features from 7D to 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# Run PCA on scaled data
x_pca = pca.fit_transform(x_scaled)

# Scatter plot
plt.scatter(x_pca[:,0], x_pca[:,1])

plt.xlabel("Principal Comp 1")
plt.ylabel("Principal Comp 2")

plt.title("World Happiness Report")

plt.show()


