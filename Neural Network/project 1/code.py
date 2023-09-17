# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:43:54 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('iris.csv')


        #split to input & output        
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

        #split to train & test data        
x_train, x_test, y_train, y_test = train_test_split(x , y , train_size=0.90)

        #train
m = MLPClassifier(hidden_layer_sizes=(64, 128, 32),
                  learning_rate='constant',
                  learning_rate_init=0.002,
                  max_iter=500)

m.fit(x_train, y_train)
        #score
print(m.score(x_train, y_train))
print(m.score(x_test, y_test))
