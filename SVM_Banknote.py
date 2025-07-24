# SVM with Kernel Trick for Banknote Authentication Dataset

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

df=pd.read_csv(url, names=columns)
print(df.head())

# Preprocessing
x = df.drop('class', axis=1)
y = df['class']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

# Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)

# RBF Kernel
from sklearn.svm import SVC
rbf = SVC(kernel='rbf')  # radical basis function (non-linear)

rbf.fit(x_train, y_train)

# Predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = rbf.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
