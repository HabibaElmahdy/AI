# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 01:10:09 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('train.csv')

data = data.dropna()

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.90)

m = MLPRegressor()#default

#trying
m = MLPRegressor(activation='tanh',
                 max_iter=2500)


m.fit(x_train, y_train)

print(m.score(x_train, y_train))
print(m.score(x_test, y_test))
