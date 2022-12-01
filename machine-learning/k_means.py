#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%%
df = pd.read_csv("datasets/Mall_Customers.csv")
df.head()

# %%
# np array
X = df.iloc[:,[3,4]].values

# %%
# .inertia_ gives us the error
wcss = []
test_n_clusters = range(1,11)

for i in test_n_clusters:
  kmeans = KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

# %%
plt.plot(test_n_clusters,wcss)
plt.title("The Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("Error (WCSS)")
plt.show()

#%% [markdown]
# We can see that at about 5 clusters, the error doesn't decrease by a lot.

# %%
kmeans = KMeans(n_clusters=5,init="k-means++")
y_kmeans = kmeans.fit_predict(X)

# %%
y_kmeans.shape

# %%
# Plotting the clusters
colors = ["red", "blue", "purple", "black", "green"]

# Plot clusters with different colors
for i in range(0,5):
  plt.scatter(X[y_kmeans==i,0],X[y_kmeans==i,1],s=25,c=colors[i],label=f"Cluster {i+1}")

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c="yellow",label="Centroids")

plt.title("Clusters of customers")
plt.xlabel("Annual Income (in $)")
plt.ylabel("Spending Score [1,100]")
plt.legend()
plt.show()
