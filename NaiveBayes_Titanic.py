# Naive Bayes for Titanic dataset, predicts whether a passenger survived based on features like age, sex, fare, etc.

import pandas as pd
import matplotlib.pyplot as plt

# load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())


# Choose important columns so it's simple
df = df[['Survived', 'Sex', 'Pclass', 'Age']]
df = df.dropna()

# Make sex numeric (0 = male, 1 = female)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features and target
x = df[['Sex', 'Pclass', 'Age']]
y = df['Survived']

# Split into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Train Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

# Predict
y_pred = gaussian.predict(x_test)

# Important data summaries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy score- ", accuracy_score(y_test, y_pred))
print("Confusion matrix ", confusion_matrix(y_test, y_pred))
print("Classification report ", classification_report(y_test, y_pred))

