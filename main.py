# Simple Linear Regression

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state=0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting 
y_predict = regressor.predict(x_test)

# Showing the data
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="green")
plt.show()
