# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
d = pd.read_csv("datasets/Social_Network_Ads.csv")
d.head()

# %%
x = d.iloc[:,[2,3]]
y = d.iloc[:,[4]]

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/4)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# %%
# Age: has a fixed range
# Salary: doesn't have a fixed range
# Hence, we need to standaradize the data, we can use StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%
from sklearn.tree import DecisionTreeClassifier

#%%
# Can be entropy or gini
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_train,y_train)

# %%
y_pred = classifier.predict(x_test)

# %%
print("DecisionTree Accuracy:", accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

#%%
from sklearn.ensemble import RandomForestClassifier

#%%
forest_classifier = RandomForestClassifier(n_estimators=100,criterion="entropy")
forest_classifier.fit(x_train,y_train)

y_pred2 = forest_classifier.predict(x_test)

print("RandomForest Accuracy:", accuracy_score(y_test,y_pred2))
confusion_matrix(y_test,y_pred2)

#%%
from sklearn.neighbors import KNeighborsClassifier

# %%
# Taking more than 5 neighbors is not worth the overfitting we may introduce.
k_neighbor_classifier = KNeighborsClassifier(n_neighbors=3,metric="minkowski",p=2)
k_neighbor_classifier.fit(x_train,y_train)

y_pred3 = k_neighbor_classifier.predict(x_test)

print("KNeighbor Accuracy:", accuracy_score(y_test,y_pred3))
confusion_matrix(y_test,y_pred3)
