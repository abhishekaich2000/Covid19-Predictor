# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:00:29 2021

@author: ABHISHEK AICH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


covid_data = pd.read_csv("covid-data.csv")

# Taking only India
covid_India = covid_data[covid_data["location"] == "India"]

# Slicing unccesary columns
covid_India.drop(["iso_code", "continent"], axis=1, inplace=True)

covid_India = covid_India.reset_index()
covid_India = covid_India.drop("index", axis=1)

# Making date column
covid_India["date"] = pd.to_datetime(covid_India["date"])
covid_India.insert(2, "day", covid_India["date"].dt.day)
covid_India.insert(3, "month", covid_India["date"].dt.month)
covid_India.insert(4, "year", covid_India["date"].dt.year)
covid_India.drop("date", axis=1, inplace=True)

# Adding day count
covid_India.insert(5, "day count", np.arange(1, (len(covid_India) + 1)))

# Analysis for total cases

# Getting X and y
X = covid_India["day count"].values.reshape(-1,1)
y = covid_India["total_cases"].values


# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y ,test_size=0.2)


# Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Graph
plt.scatter(X, y, color="red")
plt.plot(X, reg.predict(X_poly), color="black");
plt.title("Total Cases Predict")
plt.xlabel("Days passed")
plt.ylabel("No. of total cases (in million)")
plt.text(0,23000000, "Black line shows the prediction of total cases")
plt.show()

op = reg.predict(X_test)

# getting x2, y2
X2 = covid_India["day count"].values.reshape(-1, 1)
y2 = covid_India["new_cases"].values


# Analysis for New Cases

# PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
X_poly2 = poly.fit_transform(X2)

# Train test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_poly2, y2, test_size=0.2)


# Regression
reg2 = LinearRegression()
reg2.fit(X_train2, y_train2)

# Graph
plt.scatter(X2, y2, color="red")
plt.plot(X2, reg2.predict(X_poly2), color="black");
plt.title("New Cases Predict")
plt.xlabel("Days passed")
plt.ylabel("No. of new cases")
plt.text(0,450000, "Black line shows the prediction of new cases")
plt.show()

# Accuracy of our model

#For the total cases 
print("Total cases :" + str(reg.score(X_test, y_test)*100) +  " %")

#For the new cases
print("New cases :" + str(reg2.score(X_test2, y_test2)*100) + " %")

# Predict on the basis of days
predict = np.array([468, 496]).reshape(-1, 1)
print(reg2.predict(poly.transform(predict)))


# Saving the covid_India data as csv file
covid_India.to_csv("covid-India.csv")








