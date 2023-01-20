# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
kaggle_path = "../input/crypto/"
btc = pd.read_csv("data/btc_ohlc.csv",index_col="close_time",parse_dates=True)
eth = pd.read_csv("data/eth_ohlc.csv",index_col="close_time",parse_dates=True)

cryptos = [btc,eth]

print(btc.shape,eth.shape)
btc.tail()
# %%
btc[btc.isnull().any(axis=1)]
# %%
eth[eth.isnull().any(axis=1)]
# %%
sns.set_context("paper",rc={"font.size":10,"axes.titlesize":12}) # plt.rcParams
sns.set_style("dark")
sns.set_palette(palette="muted")
# %% [markdown]
# ## Change in price of Crypto asset
# Basic analysis of bitcoin and ethereum daily prices over a 9-month period.
#
# ### Close Price
# The closing price is the last price at which the cryptocurrency is traded during the regular trading day. Close price is the standard benchmark used by investors and traders to track the asset's performance over time.
# %%
f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

sns.lineplot(data=btc,x="close_time",y="close",ax=ax1)
ax1.tick_params(rotation=60)
ax1.set_title("Bitcoin Price")

sns.lineplot(data=eth,x="close_time",y="close",ax=ax2)
ax2.tick_params(rotation=60)
ax2.set_title("Ethereum Price")

plt.show()
# %%
df = pd.concat([btc[["close","volume"]],eth[["close","volume"]]],axis=1)
df.columns = ["btc_close","btc_vol","eth_close","eth_vol"]

df.head()
# %% [markdown]
# ### Volume
# Volume shows the number of buyers and sellers trading a particular currency on the exchange.
# %%
avg_vol = [df["btc_vol"].mean(),df["eth_vol"].mean()]
sns.barplot(x=["BTC","ETH"],y=avg_vol)
plt.title("Avg. volume traded")
plt.ylabel("Volume")
plt.xlabel("Currency")
plt.show()
# %% [markdown]
# ## Moving Averages
# A moving average is an indicator that shows the average price of a security over a certain period of time. Time period can be of [10,20,50] days.
# %%
time_periods = [10,20,50] # days
for period in time_periods:
  for crypto in cryptos:
    crypto[f"{period}_day_ma"] = crypto["close"].rolling(window=period).mean()
# %%
btc.head()
# %%
f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

btc.iloc[:,[3,6,7,8]].plot(ax=ax1)
ax1.set_ylabel("price")
ax1.set_title("BTC")

eth.iloc[:,[3,6,7,8]].plot(ax=ax2)
ax2.set_ylabel("price")
ax2.set_title("ETH")

plt.show()
# %% [markdown]
# Both the cryptocurrencies show similar trends and moving averages. We can observe that there was a massive fall of prices during the month of Jul. Also, we see a recent uptrend in prices.
#
# We can see that in Jan 2023 the higher period moving average, in this case a 50-day, is being cut-off by a lower moving average. Technical analysts call this is called as a golden cross, it suggest an uptrend market conditions.
# %% [markdown]
# ## Daily Percentage Returns
# Now, we must analyze the daily returns of the cryptocurrencies, and not just the absolute values.
# %%
df["btc_returns"] = df["btc_close"].pct_change()
df["eth_returns"] = df["eth_close"].pct_change()

df.head()
# %%
f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

sns.histplot(data=df,x="btc_returns",kde=True,ax=ax1)
ax1.set_title("BTC Returns")

sns.histplot(data=df,x="eth_returns",kde=True,ax=ax2)
ax2.set_title("ETH Returns")

plt.show()
# %%
f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

df["btc_returns"].plot(ax=ax1,linestyle="--",marker="o")
ax1.set_title("BTC Returns")
ax1.set_ylabel("returns")

df["eth_returns"].plot(ax=ax2,linestyle="--",marker="o")
ax2.set_title("ETH Returns")
ax2.set_ylabel("returns")

plt.show()
# %% [markdown]
# ### Returns Correlation Analysis
# Correlation is any statistical relationship, whether causal or not, between two random variables or bivariate data.
# %%
rets = df[["btc_returns","eth_returns"]]
return_corr = rets.corr(method="pearson")
return_corr
# %%
sns.heatmap(return_corr,annot=True)
plt.title("Returns Correlation Heatmap")
plt.show()
# %% [markdown]
# The close prices of BTC and ETH are highly correlated by a coefficient of 0.89. Meaning that both cryptocurrencies tend to follow the same trend direction. Same is the case with the volume of trades that take place.
#
# The following is a `jointplot` that combines both a scatter and a histogram for bivariate data.
# %%
sns.jointplot(data=df,x="btc_returns",y="eth_returns",marginal_ticks=True,ratio=3,dropna=True,kind="scatter")
plt.show()
# %% [markdown]
# ## Risk Analysis
# Standard deviation helps determine market volatility or the spread of asset prices from their average price.
# %%
sns.scatterplot(x=rets.mean(),y=rets.std(),s=60)
plt.xlabel("% Return")
plt.ylabel("Risk (sd)")

for label, x, y in zip(["BTC","ETH"],rets.mean(),rets.std()):
  plt.annotate(label,xy=(x,y),xytext=(30,30),textcoords="offset points",ha="right",va="bottom",arrowprops=dict(arrowstyle="-",color="black",connectionstyle="arc3,rad=-0.2"))

plt.show()
# %%
btc_df = pd.read_csv("data/btc_ohlc_5years.csv",index_col="close_time",parse_dates=True)
btc_df.head()
# %%
btc_df[btc_df.isnull().any(axis=1)]
# %%
plt.figure(figsize=(17,6))
sns.lineplot(data=btc_df,x="close_time",y="close")
plt.title("Close Price 5-Year History")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.show()
# %%
btc_close = btc_df[["close"]]
data = btc_close.values
print(f"Data shape: {data.shape}")
# 80% train & 20% test dataset
train_data_len = int(len(data) * 0.80)
print(f"Train data len: {train_data_len}")
# %%
# Transform features by scaling each feature to a given range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
scaled_data
# %%
train_data = scaled_data[:train_data_len]
x_train = []
y_train = []

# x: [*,*,*,...] | len(x): 50 | shape: (1426,50)
# y: [$]
# 50 x values and at the end (time) 1 y value.
for i in range(50, train_data_len):
  x_train.append(train_data[i-50:i,0])
  y_train.append(train_data[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)
# %%
# Convert to a 3D-array
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
# %%
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Building the LSTM model
model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam",loss="mean_squared_error")
# %%
# TODO: Change epochs to 10
model.fit(x_train,y_train,epochs=2,batch_size=1)
# %%
test_data = scaled_data[train_data_len-50:]
x_test = []
y_test = data[train_data_len:,:]

for i in range(50,len(test_data)):
  x_test.append(test_data[i-50:i,0])
  # y_test.append(test_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
# %%
y_pred = model.predict(x_test)
# Back from scaler data
y_pred = scaler.inverse_transform(y_pred)
# %%
# Root Mean Squared Error (RMSE)
np.sqrt(np.mean((y_pred - y_test) ** 2))
# %%
train = btc_close.iloc[:train_data_len]
valid = btc_close.iloc[train_data_len:]
valid = valid.reset_index()
valid["predicted"] = y_pred
valid = valid.set_index("close_time")
print(valid.shape)
valid.tail()
# %%
plt.figure(figsize=(17,6))
plt.plot(train["close"])
plt.plot(valid[["close","predicted"]])

plt.title("LSTM Model Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")

plt.legend(["Train","Validate","Predicted"])
plt.show()