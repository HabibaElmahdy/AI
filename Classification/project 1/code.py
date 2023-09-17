# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:31:11 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier #OR OVO

        #read
data = pd.read_csv('iris.csv')

        #split i/o
# if str in y dont change

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.90)

        #ova1
        
m = LogisticRegression(multi_class='ovr')
m.fit(x_tr,y_tr)

print(m.score(x_tr, y_tr))
print(m.score(x_ts, y_ts))

        #ova2

m = OneVsRestClassifier(LogisticRegression()) #OR OVO
m.fit(x_tr, y_tr)

print(m.score(x_tr, y_tr))
print(m.score(x_ts, y_ts))

