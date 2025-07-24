# Random Forest for Digits dataset, for handwritten digit images of digits 0 through 9

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
from sklearn.datasets import load_digits
digits = load_digits()

# DataFrame
df = pd.DataFrame(digits.data)

df['target'] = digits.target

print(df.head())


# features and labels
x = df.drop('target', axis=1)  # input data (pixel brightness values)
y = df['target']        # actual digit (0–9)

# training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=36)

# Build RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=7, random_state=36)

# Train
clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)


# Accuracy score, confusion matrix, classification report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy score is: ", accuracy_score(y_test, y_pred))
print("Confusion matrix :", confusion_matrix(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))

# Show first image in dataset
plt.imshow(x_test.iloc[0].values.reshape(8,8), cmap='gray')
plt.title("Digit Predictions")
plt.axis("off")
plt.show()


