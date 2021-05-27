# Importing the necessary modules

import pandas as pd

import numpy

from sklearn.linear_model import LinearRegression

# Loading the dataset

ds = pd.read_csv('SalaryData.csv')

# Assigning dependent and independent feature to varibles y and x respectively

y = ds['Salary']

x = ds['YearsExperience'].values.reshape(30,1)

# Creating mind

model = LinearRegression()

# Providing experience to the mind / model

model.fit(x,y)

years = input('Enter the years of experience : ')

output = model.predict([[float(years)]])

print('Expected salary for the given experience : {}'.format(output))
