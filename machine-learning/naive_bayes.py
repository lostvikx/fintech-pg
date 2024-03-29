#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
df = pd.read_csv("datasets/Social_Network_Ads.csv")
df.head()

# %%
X = df.iloc[:,[2,3]].values
y = df.iloc[:,4].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# %%
sc = StandardScaler()
# Check what this is doing!
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# %%
y_pred = classifier.predict(X_test)

# %%
confusion_matrix(y_test,y_pred)

# %%
accuracy_score(y_test,y_pred)

#%%
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
  np.arange(start=X_set[:,0].min() - 1,stop=X_set[:,0].max() + 1,step=0.01),
  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.25,cmap=ListedColormap(("red","green")))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)

plt.title("Navie Bayes (Train set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
