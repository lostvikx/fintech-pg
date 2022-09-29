
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
# Start-up data
d = pd.read_csv("datasets/50_Startups.csv")
d.head()

# %%
# drops only the Profit col
x = d.drop("Profit",axis=1)
y = d.iloc[:,4].values

# %%
# Categorical data to 0,1 values
# We can drop one and still have data integrity
X = pd.get_dummies(x,drop_first=True)
X.head()

# %%
# 80-20: train-test
# random_state: any int to preserve similar data models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# %%
X_train.shape

# %%
reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_pred

# %%
score = r2_score(y_test,y_pred)
print("r2 score:", score)
mse = mean_squared_error(y_test,y_pred)
print("mean squared error:", mse)
print("root of mse:", np.sqrt(mse))