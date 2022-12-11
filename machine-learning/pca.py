#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
df = pd.read_csv("datasets/wine_data.csv")
df.head(10)

# %%
df.iloc[:,1:].describe()

# %%
for col in df.columns[1:]:
  df.boxplot(col,by="Class",figsize=(7,4),fontsize=14)
  plt.title(f"{col}\n\n", fontsize=16)
  plt.xlabel("Wine Class", fontsize=14)

#%%
df.columns[1:]

# %%
plt.figure(figsize=(10,6))
plt.scatter(df["OD280/OD315 of diluted wines"],df["Flavanoids"],c=df["Class"],edgecolor="k",alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of two features showing correlation and class seperation",fontsize=15)
plt.xlabel("OD280/OD315 of diluted wines")
plt.ylabel("Flavanoids")
plt.show()

# %%
scaler = StandardScaler()

# %%
X = df.drop("Class",axis=1)
y = df["Class"]

# %%
X = scaler.fit_transform(X)
dfx = pd.DataFrame(data=X,columns=df.columns[1:])
dfx.head()

# %%
dfx.describe()

# %%
pca = PCA(n_components=None)
dfx_pca = pca.fit(dfx)

# %%
# Percentage of variance explained by each feature
explained_ratios = pca.explained_variance_ratio_
explained_ratios

# %%
plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(explained_ratios))],y=explained_ratios,s=100,alpha=0.75,c="orange",edgecolor="k")
# plt.grid(True)
plt.title("Explained variance ratio of \nthe fitted principal component vector",fontsize=15)
plt.xlabel("Principle components",fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()

# %%
# Sum of the total variance ratio should be 1
sum(explained_ratios)

# %%
dfx_trans = pca.transform(dfx)
dfx_trans

# %%
dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head()

# %%
# The first two features, namely Alcohol and Malic Acid are able the explain the dataset, they have a high variance ratios.
[explained_ratios[0],explained_ratios[1]]

# %%
plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df["Class"],edgecolors="k",alpha=0.75,s=150)
# Class separation using first two components
plt.title("Classification using two principal components\n", fontsize=16)
plt.xlabel("Principal Component 1: Alcohol",fontsize=14)
plt.ylabel("Principal Component 2: Malic Acid",fontsize=14)
plt.show()
