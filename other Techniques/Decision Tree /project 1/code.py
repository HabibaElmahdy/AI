"""
Created on Tue Sep 19 20:50:34 2023

@author: Habiba ELmahdy
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Iris.csv')

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.95)

m = DecisionTreeClassifier(max_depth=10)
m.fit(x_tr, y_tr)

print(m.score(x_tr,y_tr))
print(m.score(x_ts,y_ts))
