# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
d = pd.read_csv("datasets/Social_Network_Ads.csv")
d.head()

# %%
x = d.iloc[:,[2,3]]
y = d.iloc[:,[4]]

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=1/4,random_state=0)

# %%
# Age: has a fixed range
# Salary: doesn't have a fixed range
# Hence, we need to standaradize the data, we can use StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %%
# Fitting Logistic Regression to training set
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,np.ravel(y_train))

# %%
# Predict the test set
y_pred = classifier.predict(x_test)

# %%
# Test the accuracy of the model
accuracy_score(y_test,y_pred)

# %%
prob = classifier.predict_proba(x_train)

# %%
prob = pd.DataFrame(prob)
prob.head()

# %%
# Confusion Matrix
confusion_matrix(y_test,y_pred)
