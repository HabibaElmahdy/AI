# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:26:28 2023

@author: Habiba ELmahdy
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('Iris.csv')

data = data.drop(['label'], axis=1)


cluster_num = []
j = []

#find best clusters num
#for i in range(1,50):
   # m = KMeans(n_clusters = i)
   # m.fit(data)
   #cluster_num.append(i)
    #j.append(m.inertia_)
    
m = KMeans(n_clusters = 4)
m.fit(data)
