# Linear Regression that uses median income to predict the median house value for California housing with 1990 census data.
# target column is median_house_value

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load California housing data
df=pd.read_csv("california_housing_train.csv")


# Define our X and Y
X = df["median_income"]
Y = df["median_house_value"]


# Test train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40)


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train.values.reshape(-1,1), Y_train.values)

Y_pred = LR.predict(X_test.values.reshape(-1,1))
                    
plt.plot(X_test.values, Y_pred, label="Linear Regression", color='orange')
plt.scatter(X_test, Y_pred)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()

plt.show()        



