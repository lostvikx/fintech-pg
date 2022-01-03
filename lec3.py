import pandas as pd
import os
import numpy as np

data_dir = os.getcwd() + "/data"

# df = pd.DataFrame()

df = pd.read_excel(data_dir+"/gapminder_full.xlsx")
