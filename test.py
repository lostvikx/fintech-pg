import pandas as pd

# vik = pd.Series([50, 43, 48], index=["maths", "computers", "economics"])

# # print(vik)
# print(vik.loc["maths"])
# print(vik.iloc[0])

vik = [50, 48, 49]
john = [39, 40, 45]

df = pd.DataFrame({"Vikram": vik, "John": john}, index=["maths", "eco", "cs"])