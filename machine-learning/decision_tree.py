# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
d = pd.read_csv("datasets/Social_Network_Ads.csv")
d.head()

# %%
x = d.iloc[:,[2,3]]
y = d.iloc[:,[4]]

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/4,random_state=0)

# %%
# Age: has a fixed range
# Salary: doesn't have a fixed range
# Hence, we need to standaradize the data, we can use StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%
# Can be entropy or gini
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)

# %%
y_pred = classifier.predict(x_test)

# %%
confusion_matrix(y_test,y_pred)
