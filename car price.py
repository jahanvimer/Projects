# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:31:28 2022

@author: Jahanvi Mer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


#reading the data
cars = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\car_price.csv - car_price.csv.csv")
cars.head()
cars.shape
cars.info()

#to check null values
cars.isnull().sum()

#make copy of the dataset to work on it
df = cars.copy()

#carname has a lot of different stuff so we will clean it
df['CarName']
df['CarName'] = [x.split()[0] for x in df['CarName']]
df['CarName'] = df['CarName'].replace({'maxda': 'Mazda','mazda': 'Mazda', 
                                     'nissan': 'Nissan', 
                                     'porcshce': 'Porsche','porsche':'Porsche', 
                                     'toyouta': 'Toyota', 'toyota':'Toyota',
                            'vokswagen': 'Volkswagen', 'vw': 'Volkswagen', 'volkswagen':'Volkswagen'})

df = df.drop(['car_ID'], axis=1)

#differentiating between numerical and categorical columns
numerical= df.drop(['price'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')

df['price'].describe()
print( f"Skewness: {df['price'].skew()}")

df[numerical].describe()

df = df.drop('citympg',axis=1)
df[categorical].head()

df2 = pd.get_dummies(df, columns=categorical, drop_first=True)
df2.head()

X= df2.drop('price', axis=1)
y= df2['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mean_squared_error(y_test, y_pred)

r2_score(y_test, y_pred)









