# Simple Linear Regression using Scikit Learn

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# creating data frame
dataset = pd.read_csv("Salary_Data.csv")

# Matrix of attributes and vector of DV
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.head()
dataset.tail()
dataset.info()
dataset.describe()
dataset.columns

# EDA
sns.pairplot(dataset)
sns.distplot(dataset['Salary'])
# sns.distplot(dataset['YearsExperience'])
sns.heatmap(dataset.corr())

# Split into train test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=101)

# Creating and fitting linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.coef_, regressor.intercept_

# Predicting test results
y_pred = regressor.predict(X_test)

# Visualizing the resultset
plt.scatter(y_test, y_pred)
# Residuals
sns.distplot(y_test - y_pred)

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Using statsmodel to get statistics output
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

regressor_2 = sm.OLS(y_train, X_train).fit()
# R kind of statistical output
regressor_2.summary()

