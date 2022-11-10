# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#%%
d = pd.read_csv("datasets/PastHires.csv")
d.head()

# %%
df = d.copy()
convert = {"Y":1,"N":0}
df["Hired"] = df["Hired"].map(convert)
df["Employed?"] = df["Employed?"].map(convert)
df["Top-tier school"] = df["Top-tier school"].map(convert)
df["Interned"] = df["Interned"].map(convert)

convert = {"BS":0,"MS":1,"PhD":2}
df["Level of Education"] = df["Level of Education"].map(convert)

# %%
df.head()

# %%
features = list(df.columns[:6])
print(features)

y = df["Hired"]
x = df[features]

# %%
classifier = DecisionTreeClassifier()
classifier.fit(x,y)

#%%
test = pd.DataFrame(columns=x.columns,data=[[1,0,2,0,1,1],[3,1,3,0,0,1]])
test

#%%
# Can input an entire df to predict if they are hired or not:
classifier.predict(test)

#%%
# from IPython import Image
# from six import StringIO

# import pydotplus

# dot = StringIO()
# tree.export_graphviz(classifier, out_file=dot,feature_names=features)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
