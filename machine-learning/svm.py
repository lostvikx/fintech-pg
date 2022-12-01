#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("datasets/Position_Salaries.csv")
df.head()

#%%
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

# %%
regressor = SVR(kernel="rbf")
model = TransformedTargetRegressor(regressor=regressor,transformer=StandardScaler())

model = model.fit(X,y)

# %%
y_pred = model.predict([[6.5]])
y_pred

# %%
plt.scatter(X,y,color="red")
plt.plot(X,model.predict(X),color="green")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
