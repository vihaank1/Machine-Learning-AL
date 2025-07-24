# Decision Tree breast cancer dataset, medical data for classifying a tumor as benign or malignant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

df=pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target

print(df.head())

# Features: measurements of tumors (radius, texture, area, etc.)
# Target: 0 = malignant, 1 = benign


# Train_test
from sklearn.model_selection import train_test_split

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Create Decision Tree Classifier & train
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, random_state=30)

clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)

# Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

# Visualize model
from sklearn.tree import plot_tree

plt.figure(figsize=(10,20))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Breast Cancer status")
plt.show()


