import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("StudentsPerformance.csv")

            #cleaning
data = data.drop(["test preparation course","race/ethnicity"],axis=1)

            #lebeling
object_data = data.select_dtypes(include=["object"])
le = preprocessing.LabelEncoder()
for i in range(object_data.shape[1]):
    object_data.iloc[:,i] = le.fit_transform(object_data.iloc[:,i])

            #concat
num_data = data.select_dtypes(exclude=["object"])

data = pd.concat([object_data, num_data],axis=1)
print(data)

            
            #Ralationship
c = data.corr()

print(c)
sns.heatmap(c,annot=True)
plt.show()
