# KNN for the Iris Dataset. Y: 0 = Setosa, 1 = Versicolor, 2 = Virginica

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

# Train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

# Accuracy score and Prediction
from sklearn.metrics import accuracy_score
Y_pred = knn.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
print(Y_pred)              # Our result is 1.0 so our iris is classified as Versicolor


