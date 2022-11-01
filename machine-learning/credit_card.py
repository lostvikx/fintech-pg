#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
d = pd.read_csv("datasets/creditcard.csv")
d.head()

#%%
x = d.iloc[:,1:30]
y = d.iloc[:,[30]]
x.head()

# %%
# We might miss out on the fraud cases when random sampling, we use stratify to make both training and testing sets contain
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/4,random_state=0,stratify=y)

# %%
# Amount doesn't have a fixed range
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %%
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train["Class"])

# %%
y_pred = classifier.predict(x_test)

# %%
accuracy_score(y_test,y_pred)

# %%
confusion_matrix(y_test,y_pred)
