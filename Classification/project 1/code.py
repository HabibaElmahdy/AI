# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:05:22 2023

@author: Habiba ELmahdy
"""
#import pandas as pd
import numpy as np
#import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

        #split data i/o > x/y
        
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

        #if there null
#print(data.isnull().sum())

        #split train & test data

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.95)

        #train
        
m = LogisticRegression()
m.fit(x_train, y_train)

print(m.score(x_train, y_train))

print(m.score(x_test, y_test))
