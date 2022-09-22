# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
dt = pd.read_csv("datasets/Salary_Data.csv")
x = dt.iloc[:,:-1].values  # yrs_exp
y = dt.iloc[:,1:].values  # salary

dt.head()

# %%
x.shape, y.shape

# %%
# Splitting data into train & test
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=1/3, random_state=0)

# %%
# Creating a regression model
reg = LinearRegression().fit(x_train,y_train)
# Coefficient of Determination (R^2)
reg.score(x_train,y_train)

# %%
reg.coef_

# %%
# Predict x = 10.2 in the model
reg.predict(np.array([[10.2]]))

# %%
y_pred = reg.predict(x_test)

# %%
# Mean Square Error
np.sum((y_test - y_pred)**2) / 20

# %%
mean_squared_error(y_test,y_pred)

# %%
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train, reg.predict(x_train))
plt.title("Salary vs Experience (Training)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# %%
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test, reg.predict(x_test))
plt.title("Salary vs Experience (Test)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
