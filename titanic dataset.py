# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:24:40 2022

@author: Jahanvi Mer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


train_data = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\train - train.csv")
test_data = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\test - test.csv")
test_results = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\gender_submission - gender_submission.csv")

train_data.head()
train_data.describe()
train_data.isnull().sum()


train_data.Survived.value_counts()
plt = train_data.Survived.value_counts().plot(kind='bar')
plt.set_xlabel('Survived or not')
plt.set_ylabel('Passenger Count')

plt = train_data.Pclass.value_counts().sort_index().plot(kind='bar', title='')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')

train_data[['Pclass', 'Survived']].groupby('Pclass').count()
train_data[['Pclass', 'Survived']].groupby('Pclass').sum()


plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot(kind='bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')


plt = train_data.Sex.value_counts().sort_index().plot(kind='bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Passenger count')

plt = train_data[['Sex','Survived']].groupby('Sex').mean().Survived.plot(kind='bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Probability')

plt = train_data.Embarked.value_counts().plot(kind='bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probabilty')

plt = train_data[['Embarked','Survived']].groupby('Embarked').mean().Survived.plot(kind='bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probabilty')


#combine sibsp and parch to create family size and drop sibsp and parch
train_data.columns
train_data['family_size'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data = train_data.drop(columns=['SibSp','Parch'])

# drop uncesscary columns. passenger Id, cabin(cause it has alot of null values) and ticket
train_data = train_data.drop(columns=['PassengerId','Ticket','Cabin'])

label = LabelEncoder()
label_model = label.fit(train_data['Sex'])
train_data['Sex'] = label_model.transform(train_data['Sex'])
train_data['Embarked'] = train_data['Embarked'].map({'C':0,'Q':1,'S':2})


train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data = train_data.drop(columns='Name')
train_data.Title.unique()

plt = train_data.Title.value_counts().plot(kind='bar')

train_data['Title'] = train_data['Title'].replace(['Dr','Rev','Major','Col','Countess','Capt','Sir','Lady','Don','Jonkheer'],'Others')
train_data['Title'] = train_data['Title'].replace('Ms','Miss')
train_data['Title'] = train_data['Title'].replace('Mme','Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle','Miss')

plt = train_data['Title'].value_counts().sort_index().plot(kind='bar')
plt.set_xlabel('Title')
plt.set_ylabel('Surviavl Probabilty')

plt = train_data[['Title','Survived']].groupby('Title').mean().Survived.plot(kind='bar')
plt.set_xlabel('Title')
plt.set_ylabel('Survived Probabilty')

train_data['Title'] = train_data['Title'].replace({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Others':4})
train_data.head()
train_data.isnull().sum()

train_data['Embarked'] = train_data['Embarked'].fillna(2)
train_data['Embarked'].isnull().sum()
train_data['Embarked'] = train_data['Embarked'].astype(int)
train_data.info()

age_median = train_data.Age.median()
train_data['Age'] = train_data['Age'].fillna(age_median)
train_data['Age'].isnull().sum()

train_data.isnull().sum()


train_data1 = train_data.copy()
x_train = train_data1.drop(columns='Survived')
y_train = train_data1[['Survived']]

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





'''
train_data1 = train_data.copy()
Scaled_data = scale(train_data1)

data = pd.DataFrame(Scaled_data, columns = train_data.columns)


x_train = data.drop(columns = 'Survived')
y_train = data[['Survived']]

x_train.shape

x_training, x_valid, y_training, y_valid = train_test_split(x_train,y_train, test_size = 0.2, random_state=100)

Coef = LogisticRegression()
Coef.fit(x_training,y_training)
#for the error, the y _train needs to be in class format, need to remove it from scaled data. scale the data after removing the prediction output.
'''


















