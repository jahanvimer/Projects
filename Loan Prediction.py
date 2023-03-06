# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:27:03 2022

@author: Jahanvi Mer
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_data = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\Loan Perdiction Train Data.csv")
test_data = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\Loan Perdiction Test Data.csv")

train_data.shape
train_data.info()

train_data.isnull().sum()

corr_matrix = train_data.corr()
sns.heatmap(data=corr_matrix , cmap='BrBG', annot=True, linewidth=0.2)


train_data = train_data.drop(columns=['Loan_ID'])
'''
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train_data,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col,data=train_data,x='Loan_Status',ax=axes[idx])

plt.subplots_adjust(hspace=1)
'''

train_data["Gender"].fillna(train_data["Gender"].mode()[0],inplace=True)
train_data['Married'] = train_data['Married'].fillna(train_data['Married'].mode()[0])
train_data["Dependents"].fillna(train_data["Dependents"].mode()[0],inplace=True)
train_data["Self_Employed"].fillna(train_data["Self_Employed"].mode()[0],inplace=True)
train_data["Credit_History"].fillna(train_data["Credit_History"].mode()[0],inplace=True)

train_data.isnull().sum()

train_data['Loan_Amount_Term'].value_counts()
train_data["Loan_Amount_Term"].fillna(train_data["Loan_Amount_Term"].mode()[0],inplace=True)

train_data['LoanAmount'].fillna(train_data['LoanAmount'].mode()[0],inplace=True)

train_data.isnull().sum()

train_data['Gender'] = train_data['Gender'].replace({'Male':0,'Female':1})
train_data['Married'] = train_data['Married'].replace({'Yes':0,'No':1})
train_data['Dependents'] = train_data['Dependents'].replace({'3+':3})
train_data['Education'] = train_data['Education'].replace({'Graduate':0,'Not Graduate':1})
train_data['Self_Employed'] = train_data['Self_Employed'].replace({'Yes':0,'No':1})
train_data['Property_Area'] = train_data['Property_Area'].replace({'Semiurban':0,'Urban':1,'Rural':2})


#In loan status Y is loan granted which i changed to 1 and N is loan not granted which i changed to 0
train_data['Loan_Status'] = train_data['Loan_Status'].replace({'N':0,'Y':1})


corr_matrix = train_data.corr()
sns.heatmap(data=corr_matrix , cmap='BrBG', annot=True, linewidth=0.2)

train_data['Total_income'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']
train_data.drop(['ApplicantIncome','CoapplicantIncome'], axis='columns', inplace = True)

train_data1 = train_data.copy()
x_train = train_data1.drop(columns='Loan_Status')
y_train = train_data1[['Loan_Status']]

Scaled_data = scale(x_train)

data = pd.DataFrame(Scaled_data, columns = x_train.columns)

x_train.shape
x_training, x_valid, y_training, y_valid = train_test_split(x_train,y_train, test_size = 0.2, random_state=100)

Coef = LogisticRegression()
Coef.fit(x_training,y_training)

prediction = Coef.predict(x_valid)

accuracy_score(y_valid,prediction)

confusion = confusion_matrix(y_valid, prediction,labels=[1,0])
confusion
report = classification_report(y_valid, prediction)
report



















