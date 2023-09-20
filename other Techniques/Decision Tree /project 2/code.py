"""
Created on Wed Sep 20 21:05:11 2023

@author: Habiba ELmahdy
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
data = data.dropna()

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.95)



m =  DecisionTreeRegressor(max_depth=10)
m.fit(x_tr, y_tr)

print(m.score(x_tr,y_tr))
print(m.score(x_ts,y_ts))
