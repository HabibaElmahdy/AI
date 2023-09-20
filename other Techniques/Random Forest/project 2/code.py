"""
Created on Wed Sep 20 22:37:36 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('train.csv')
data = data.dropna()

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.95)

m = RandomForestRegressor()


m.fit(x_tr, y_tr)

print(m.score(x_tr, y_tr))
print(m.score(x_ts, y_ts))
