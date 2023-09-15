# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 20:06:54 2023

@author: Habiba ELmahdy
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('polynomial-regression.csv')

        #split x > input y > output

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

        #draw

plt.scatter(x, y)
plt.show()

        #as linear

'''m = LinearRegression()

m.fit(x,y)

print(m.score(x,y)) #baaaaad'''

        #nonlinear
        
poly = PolynomialFeatures(degree=4)

p_x = poly.fit_transform(x)

m = LinearRegression()

m.fit(p_x,y)

print(m.score(p_x,y))

        #draw
        
plt.scatter(x, y)
plt.plot(x,m.predict(p_x))
plt.show()
