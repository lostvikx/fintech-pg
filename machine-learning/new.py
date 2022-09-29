# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])

# %%
reg = LinearRegression().fit(x,y)

# %%
print("slope:", reg.coef_)
print("y-intercept:", reg.intercept_)
print("R^2", reg.score(x,y))

# %%
rand_x = np.abs(np.random.standard_normal((6,1)) * 20)
y_pred = reg.predict(rand_x)
y_pred

# %%
reg.coef_ * rand_x + reg.intercept_

# %%
# Multi Linear Regression
x = np.array([[0,1],[5,1],[15,2],[25,5],[35,11],[45,15],[55,34],[60,35]])
y = np.array([4,5,20,14,32,22,38,43])

# %%
plt.scatter(x,y)
