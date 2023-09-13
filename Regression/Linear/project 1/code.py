# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:28:20 2023

@author: Habiba ELmahdy
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

#print(data.head(10))
#print(data.shape)
#print(data.isnull().sum())nums
        #delelte nuls
data = data.dropna()

#print(data.isnull().sum())0

        #split data for x, y
        
x = data.iloc[:,:-1]

y = data.iloc[:,-1]

        #split data for Train & Test

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


m = LinearRegression()

m.fit(x_train, y_train)

            #accuracy
            
print(m.score(x_train, y_train))

print(m.score(x_test, y_test))


#plt.scatter(x, y)

plt.scatter(x_train, y_train)
plt.plot(x_train, m.predict(x_train), color = "red")
plt.show()

plt.scatter(x_test, y_test, color = "grey")
plt.plot(x_test, m.predict(x_test), color = "green")
plt.show()
