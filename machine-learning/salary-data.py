# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

# %%
dt = pd.read_csv("datasets/Salary_Data.csv")
X = dt.iloc[:,:-1].values  # yrs_exp
y = dt.iloc[:,1:].values  # salary

#%%
ploy_reg = PolynomialFeatures(degree=4)
# Transform X to X_poly
X_poly = ploy_reg.fit_transform(X)
# ploy_reg.fit(X_poly,y)

reg = LinearRegression()
reg.fit(X,y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

# print(lin_reg.coef_)
print("y-intercept:", lin_reg.intercept_)

# %%
plt.scatter(X,y,color="red")
plt.plot(X,reg.predict(X))
plt.title("Linear Regression")
plt.show()

# %%
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(ploy_reg.fit_transform(X)),color="blue")
plt.title("Polynomial Regression")
plt.show()

# %%
def create_poly_plot(degree, X, y):
  ploy_reg = PolynomialFeatures(degree)
  # Transform X to X_poly
  X_poly = ploy_reg.fit_transform(X)
  # ploy_reg.fit(X_poly,y)

  reg = LinearRegression()
  reg.fit(X,y)
  lin_reg = LinearRegression()
  lin_reg.fit(X_poly,y)

  plt.scatter(X,y,color="red")
  plt.plot(X,lin_reg.predict(ploy_reg.fit_transform(X)),color="blue")
  plt.title(f"Polynomial Regression of degree {degree}")
  plt.show()

# %%
# Degree of 4 looks like a great fit
create_poly_plot(4,X,y)
