import requests
import pandas as pd

def get_crypto_historic_prices(symbol, exchange="bitstamp", after="2022-08-01"):
  url = f"https://api.cryptowat.ch/markets/{exchange}/{symbol}usd/ohlc"

  res = requests.get(url, params={
    "period": "3600",
    "after": str(int(pd.Timestamp(after).timestamp()))
  })

  res.raise_for_status()
  data = res.json()
  
  # 3600s (fetch hourly data)
  df = pd.DataFrame(data["result"]["3600"], columns=[
    "close_time", "open", "high", "low", "close", "volume", "quote_volume"
  ])

  # Parse date
  df['close_time'] = pd.to_datetime(df['close_time'], unit='s')
  df.set_index('close_time', inplace=True)

  return df

def main():
  # btc and eth are the two most popular/traded crypto-currencies
  btc = get_crypto_historic_prices("btc")
  print(btc.tail())
  print(btc.shape)

if __name__ == "__main__":
  main()
