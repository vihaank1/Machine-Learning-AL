# Logistic Regression to Predict if a person will subscribe to a term bank deposit, options (yes/no)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("bank/bank-full.csv", sep=";")

print(df.shape)
print(df.head())
print(df.info)

print(df.columns.tolist())


# Preprocessing
# Convert 'y' into binary
df["y"] = df["y"].map({'yes': 1, 'no': 0})

# One-hot encoding for categorical features we have in csv
df_new = pd.get_dummies(df.drop("y", axis=1))
df_new["y"] = df["y"]

print(df_new)

# Split into training and testing data
from sklearn.model_selection import train_test_split

X = df_new.drop("y", axis=1)
Y = df_new["y"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale our features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Perform Logistic Regression
from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()

LogR.fit(X_train_scaled, Y_train)


# Make our Predictions
Y_pred = LogR.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy score: ", accuracy_score(Y_test, Y_pred))
print("Confusion matrix: ", confusion_matrix(Y_test, Y_pred))
