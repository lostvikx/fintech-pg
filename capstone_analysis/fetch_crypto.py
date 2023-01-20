import requests
import pandas as pd
import os

def get_crypto_historic_prices(symbol, exchange="bitstamp", after="2018-01-01", period="86400"):
  url = f"https://api.cryptowat.ch/markets/{exchange}/{symbol}usd/ohlc"

  res = requests.get(url, params={
    "period": period,
    "after": str(int(pd.Timestamp(after).timestamp()))
  })

  res.raise_for_status()
  data = res.json()
  
  # tokens
  cost = data.get("allowance").get("cost")
  remaining = data.get("allowance").get("remaining")
  print(f"Cost: {cost}\nRemaining: {remaining}")  # Cost: 0.015

  # 3600s (fetch hourly data)
  df = pd.DataFrame(data["result"][period], columns=[
    "close_time", "open", "high", "low", "close", "volume", "quote_volume"
  ])

  # Parse date
  df['close_time'] = pd.to_datetime(df['close_time'], unit='s')
  df.set_index('close_time', inplace=True)

  return df

def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))

  # btc and eth are the two most popular/traded crypto-currencies
  currencies = ["btc","eth"]
  for currency in currencies:
    df = get_crypto_historic_prices(currency)
    print(df.shape)
    df.to_csv(f"{dir_path}/data/{currency}_ohlc_5years.csv")

if __name__ == "__main__":
  main()
