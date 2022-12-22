#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %%
path = "../input/" # for kaggle
df = pd.read_csv("insurance.csv")
print(f"Shape: {df.shape}")

df.head()

# %%
df.describe()

#%%
# Make sure we don't have any null values
df[df.isnull().any(axis=1)]

# %%
corr = df.corr(method="pearson",numeric_only=True)
corr

# %%
sns.heatmap(corr,annot=True)
plt.title("Correlation heatmap")
plt.show()

#%% [markdown]
# There is no correlation between the variables.

# %%
plt.figure(figsize=(14,6))
sns.boxplot(x="children",y="charges",hue="sex",data=df,palette="Set2")
plt.title("Box plot of charges vs no. of children")
plt.show()

# %%
child_charges = df.loc[:,["children","charges"]]
child_charges.head()

#%%
child_charges.groupby("children").agg(["mean","min","max"])["charges"]

# %%
# Aggregate mean values for each n children
plt.figure(figsize=(10,7))
sns.barplot(data=child_charges,x="children",y="charges",color='#69b3a2')
plt.title("Mean charges for n children")
plt.show()

# %%
sns.scatterplot(data=df,x="age",y="charges",hue="smoker",palette="rocket")
plt.title("Age vs charges")
plt.show()

# %%
sns.scatterplot(data=df,x="bmi",y="charges",hue="smoker",palette="viridis")
plt.title("BMI vs charges")
plt.show()

# %%
df.head()

#%% [markdown]
# Let's create a new age_cat variable, make age a categorical var.
#
# * Young Adult: 18 - 35
# * Adult: 36 - 55
# * Elder: 56 or older

#%%
df["age_cat"] = np.nan
age_categories = ["Young Adult", "Adult", "Elder"]

for col in [df]:
  col.loc[(col["age"] >= 18) & (col["age"] <= 35), "age_cat"] = age_categories[0]
  col.loc[(col["age"] >= 36) & (col["age"] <= 55), "age_cat"] = age_categories[1]
  col.loc[col["age"] >= 56, "age_cat"] = age_categories[2]

df.head()

#%%
labels = df["age_cat"].unique().tolist()
amount = df["age_cat"].value_counts().tolist()

# ["#ff9999", "#b3d9ff", " #e6ffb3"]

plt.pie(x=amount,labels=labels,colors=sns.color_palette("Set2"),autopct='%.0f%%')
plt.title("Age categories")
plt.show()

#%%
sns.displot(data=df,x=df["bmi"],kde=True,color="#6a3d9a")
plt.title("Normal distribution: BMI")
plt.show()

#%%
sns.catplot(data=df,x="age_cat",y="charges",kind="bar",col="sex",palette=["#e78ac3"])
plt.title("Mean Charges by Age")
plt.show()

#%% [markdown]
# Mean could be affected easily by outliers, so we take median values as well.

#%%
mean_charges = []
median_charges = []

for cat in age_categories:
  mean_charges.append(df["charges"].loc[df["age_cat"] == cat].mean())
  median_charges.append(df["charges"].loc[df["age_cat"] == cat].median())

f = plt.figure(figsize=(12,4))

ax = f.add_subplot(121)
sns.barplot(x=age_categories,y=mean_charges,color="#fc8d62",ax=ax)
ax.set_ylabel("Mean Charges")
ax.set_title("Mean Charges by Age")

ax = f.add_subplot(122)
sns.barplot(x=age_categories,y=median_charges,color="#8da0cb")
ax.set_ylabel("Median Charges")
ax.set_title("Median Charges by Age")

plt.show()

#%% [markdown]
# [BMI Categories](https://www.cdc.gov/obesity/basics/adult-defining.html) defined by the Health Ministry of India:
#
# * Underweight: BMI < 18.5
# * Healthy: BMI >= 18.5 & BMI < 25
# * Overweight: BMI >= 25 & BMI < 30
# * Obesity: BMI >= 30

#%%
df["bmi_cat"] = np.nan
df.loc[(df["bmi"] < 18.5),"bmi_cat"] = "Underweight"
df.loc[(df["bmi"] >= 18.5) & (df["bmi"] < 25),"bmi_cat"] = "Healthy"
df.loc[(df["bmi"] >= 25) & (df["bmi"] < 30),"bmi_cat"] = "Overweight"
df.loc[(df["bmi"] >= 30),"bmi_cat"] = "Obesity"

df.head()

#%%
f, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

sns.stripplot(data=df,x="age_cat",y="charges",ax=ax1,color="#e5c494")
ax1.set_title("Relationship b/w Age & Charges")

sns.stripplot(data=df,x="age_cat",y="charges",hue="bmi_cat",ax=ax2,palette="Set2",legend="brief")
plt.legend(bbox_to_anchor=(1,1))
ax2.set_title("Relationshp b/w Age & Charges by BMI")

plt.show()

#%% [markdown]
# The obese group is charged more insurance premium than other BMI groups.

#%%
f, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

sns.stripplot(data=df,x="smoker",y="charges",ax=ax1,color="#e5c494")
ax1.set_title("Relationship b/w Smoker & Charges")

sns.stripplot(data=df,x="smoker",y="charges",hue="bmi_cat",ax=ax2,palette="Set2")
plt.legend(bbox_to_anchor=(1.38,1))
ax2.set_title("Relationship b/w Smoker & Charges by BMI")

plt.show()

# %% [markdown]
# Obese smokers are the group that are charged the highest insurance premium.

# %%
sns.scatterplot(data=df,x="bmi",y="charges",hue="bmi_cat",palette="Set1")
plt.title("Relationship b/w BMI & Charges by BMI Condition")
plt.show()

# %%
sns.scatterplot(data=df,x="bmi",y="charges",hue="smoker",palette="Set1")
plt.title("Relationship b/w BMI & Charges by Smoking Condition")
plt.show()

# %% [markdown]
# By looking at the scatter plot, we can conclude that on average smokers are charged higher insurance premium.
#
# Now let's try to do a cluster analysis using k-means cluster. First we need to implement something called as the elbow method to find the optimal number of clusters or k.

# %%
X = df[["bmi", "charges"]]

# WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
wcss = []
n_clusters_test = range(1,11)

for n in n_clusters_test:
  kmeans = KMeans(n_clusters=n,init="k-means++")
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

# %%
sns.lineplot(x=n_clusters_test,y=wcss,color="#66c2a5")
plt.title("The Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS (Error)")
plt.show()

# %% [markdown]
# From the above line plot, we can conclude that after 3 clusters the WCSS isn't decreasing significantly.

# %%
kmeans = KMeans(n_clusters=3,init="k-means++")
kmeans.fit(X)

# %%
kmeans.cluster_centers_

# %%
kmeans.labels_

# %%
ax = sns.scatterplot(x=X.values[:,0],y=X.values[:,1],hue=kmeans.labels_,palette="Set1",alpha=0.6)

sns.scatterplot(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],ax=ax,s=60,color="black",label="Centroids",alpha=0.9)

plt.title("Clusters of Policyholders")
plt.xlabel("BMI (kg/m2)")
plt.ylabel("Premium Charges ($)")
plt.show()

# %% [markdown]
# We can conclude that there are 3 clusters in the dataset.

# %%
df = pd.read_csv("insurance.csv")

categorical_data = ["sex","smoker","region"]
# OHE: One-Hot Encoding
df_encode = pd.get_dummies(data=df,prefix="OHE",prefix_sep="_",columns=categorical_data,drop_first=True)

df_encode.head()

# %%
print(f"Columns in original data frame:\n{df.columns.values}")
print(f"\nShape: {df.shape}")
print(f"\nColumns in encoded data frame:\n{df_encode.columns.values}")
print(f"\nNew Shape: {df_encode.shape}")

#%% [markdown]
# Let's check the distribution of our dependent variable: charges.
# We do this to know whether our dependent variable is normally distributed or not.
# 
# A normally distributed dependent variable will yield us more accurate results.


# %%
f= plt.figure(figsize=(12,4))

ax = f.add_subplot(121)
sns.kdeplot(data=df_encode,x="charges",ax=ax)
ax.set_title("Distribution of insurance premium charges")

ax = f.add_subplot(122)
sns.kdeplot(data=df_encode,x="charges",hue="children",ax=ax)
ax.set_title("Distribution of insurance premium charges for n children")

plt.show()

# %% [markdown]
# We can clearly see that the premium charges, which is also our dependent variable, does not follow a normal distribution.

# %%
sns.displot(x=np.log(df_encode["charges"]),kde=True,color='#7995c4')
plt.title("Distribution of ln charges")
plt.show()

#%% [markdown]
# By taking the log to the base 10 of the charges value, we get a normally distributed variable.

# %%
df_encode["log_charges"] = np.log(df_encode["charges"])

# %% [markdown]
# Let's try a linear regression model.

# %%
X = df_encode.drop(["charges","log_charges"],axis=1)
y = df_encode["log_charges"].values.reshape(-1,1)

#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# %%
lin_reg = LinearRegression().fit(X_train,y_train)

# %%
theta = list(lin_reg.intercept_) + list(lin_reg.coef_.flatten())
params_df = pd.DataFrame({"Columns": ["intercept"]+list(X.columns.values),"Theta": theta})

params_df

#%%
y_pred = lin_reg.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test,y_pred):.4f}")
print(f"R squared: {r2_score(y_test, y_pred):.4f}")

# %% [markdown]
# The model returns R^2 value of about 75%, so it fits our data test very well, but we can still imporve the performance of by using a different technique.

#%% [markdown]
# My personal data is used in the custom_test df.
# The data includes my age, bmi, children, smoker or not, etc.

# %%
custom_test = pd.DataFrame(columns=X_test.columns.values,data=[[22,21.5,0,1,0,0,0,1]])

prem_charges = lin_reg.predict(custom_test)
my_pred_charge = np.exp(prem_charges)[0][0]
print(f"My predicted premium charges: {my_pred_charge:.2f}")

# %%
sns.scatterplot(x=y_test.flatten(),y=y_pred.flatten())
plt.title("Check for linearity:\nActual Vs Predicted values")
plt.xlabel("ln(charges)")
plt.ylabel("ln(charges)")
plt.show()

# %%
residual = y_test - y_pred
sns.displot(x=residual.flatten(),kde=True,color="#e63b3f")
plt.axvline(residual.mean(),linestyle="--",color="black")
plt.title("Residual Error")
plt.xlabel("Difference from mean")
plt.show()

# %% [markdown]
# The mean for the above distribution plot is close to 0. Also, the residual mean is zero and residual error plot right skewed.
