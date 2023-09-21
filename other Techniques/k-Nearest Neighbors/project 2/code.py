"""
Created on Thu Sep 21 21:19:15 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('train.csv')
data = data.dropna()

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.50)

m = KNeighborsRegressor(#n_neighbors=10
                         )

m.fit(x_tr, y_tr)

print(m.score(x_tr,y_tr))
print(m.score(x_ts,y_ts))
