#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# %%
d = pd.read_csv("datasets/Position_Salaries.csv")
d.head()

# %%
x = d.iloc[:,1:2].values
y = d.iloc[:,2].values

# %%
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

# %%
y_pred = regressor.predict([[2.5]])
y_pred

#%%
# x_grid = np.arange(min(x),max(x),0.01)
# x_grid = x_grid.reshape((len(x_grid),1))
# plt.scatter(x,y,color="red")
# plt.plot(x,regressor.predict(x_grid))

# %%
from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor(n_estimators=10)

#%%
reg2.fit(x,y)
y_pred2 = reg2.predict([[2.5]])
y_pred2
