# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:07:56 2023

@author: Jahanvi Mer
"""
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.stattools import 
X = pd.read_csv(r"C:\Users\Jahanvi Mer\Downloads\AirPassengers.csv")

X.info()

#.to_datetime is used to change object to date time
X['Month'] = pd.to_datetime(X['Month'],infer_datetime_format=True)
X.info()
indexedDataset = X.set_index(['Month'])
indexedDataset.head()
X.info()

#Graphs

plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.plot(indexedDataset)

#checking mathematically if the data is stationary or not
#Rolling mean/Window is where we give a window side, it will take an average 
#of the window you have decided. 

rolmean = indexedDataset.rolling(window=12).mean()
rolstd = indexedDataset.rolling(window=12).std()
rolmean
rolstd

#plotting rolling stats
orig = plt.plot(indexedDataset, color='Blue', label='Original')
mean = plt.plot(rolmean, color='Red',label='Rolling Mean')
std = plt.plot(rolstd, color='Black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.show(block=False)

#using ADCF test to see if our data is stationary or not
#for a time series to be stationary, the ADCF test result should be
# p-value should be lower and critical values at 1%, 5% and 10% should be as close as possible to Test Statistics

dftest = adfuller(indexedDataset['#Passengers'],autolag='AIC')
dftest

dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','#lags used','Number of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
dfoutput
    
#to make data stationary
#log scale transformation
indexedDataset_logscale = np.log(indexedDataset)
plt.plot(indexedDataset_logscale)

#to make data stationary
movingAverage = indexedDataset_logscale.rolling(window=12).mean()
movingStd = indexedDataset_logscale.rolling(window=12).std()
plt.plot(indexedDataset_logscale)
plt.plot(movingAverage, color='Red')

#to remove trend
datasetlogscaleMinusMovingAverage = indexedDataset_logscale - movingAverage
datasetlogscaleMinusMovingAverage.head()

#to remove Nan values
datasetlogscaleMinusMovingAverage.dropna(inplace=True)
datasetlogscaleMinusMovingAverage.head(10)

#df and adcf test for now stationary data
dftest = adfuller(datasetlogscaleMinusMovingAverage['#Passengers'],autolag='AIC')
dftest

dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','#lags used','Number of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
dfoutput
#as per this test we can see that the data is now stationary.




























