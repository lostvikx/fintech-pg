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
# %% [markdown]
# This shows that ETH is a high risk or more volatile cryptocurrency than BTC, but also provides higher returns. Higher the risk, higher the expected return.
#
# ## Deep Learning for Crypto price prediction
# There are several machine learning models that can be used to predict stock prices. One such model is Long Short-Term Memory (LSTM). 
# 
# The LSTM (Long Short-Term Memory) layers are a type of recurrent neural network (RNN) that are commonly used for time series data, like stock prices.
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
# %% [markdown]
# Transform features by scaling each feature to a given range. Commonly the feature range is from 0 to 1. This helps optimize the data, for faster algorithm runtime.

# %%
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
# %% [markdown]
# From the set of 50-days close price of the currency, we want to predict the 51th day close price.
# 
# * Predictor variable: 50-day close price
# * Target variable: 51th close price
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
# %% [markdown]
# The first LSTM layer has 128 units and the input shape is taken from `x_train.shape[1]` and it has return_sequences=True which is used to connect with next LSTM layer.
#
# The second LSTM layer has 64 units and return_sequences is set to False which means it is the last layer of the LSTM network.
#
# The first dense layer has 25 units and the last dense layer has 1 unit. This architecture is used for regression problem.
#
# The model is then compiled with the Adam optimizer and mean squared error (MSE) loss function. This means that the model will be trained to minimize the MSE between the predicted values and the true values in the training data.
# %%
model.fit(x_train,y_train,epochs=1,batch_size=1)
# %% [markdown]
# Few general guidelines for LSTM parameters:
# 
# LSTM:
# * The number of units in the LSTM layer can affect the model's ability to capture long-term dependencies in the data. A larger number of units generally means that the model has more capacity to learn complex patterns, but it also increases the risk of overfitting. A good starting point is typically in the range of 64 to 256 units.
#
# Dense layers:
# * The number of units in the last dense layer should be equal to the number of output classes.
#
# batch_size: 
# * The batch size used during training. This is the number of samples per gradient update. 
# * The larger the batch size, the more memory space you'll need. But smaller batch size can lead to more updates and that can result in a better model.
#
# epochs:
# * The number of times the model will cycle through the data. One epoch is when an entire dataset is passed forward and backward through the neural network only once.
# * Typically, a larger number of epochs is used to train the model, like 10 or 20, to allow the model to learn from more examples.
# * In our case, we use `epoch = 1` for as this can be computationally expensive.
# %%
test_data = scaled_data[train_data_len-50:]
x_test = []
y_test = data[train_data_len:,:]

for i in range(50,len(test_data)):
  x_test.append(test_data[i-50:i,0])

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
# %% [markdown]
# ### Why LSTM?
# Recurrent Neural Networks (RNNs) are a type of neural network that allows for information to be passed from one step of the network to the next. LSTM (Long Short-Term Memory) is a specific type of RNN that is able to effectively learn and retain long-term dependencies by using gates to control the flow of information. These gates can be thought of as switches that determine whether to let new information in, or to keep previous information stored. This allows LSTMs to better handle tasks that require remembering previous events, such as language translation, speech recognition, stock price prediction.