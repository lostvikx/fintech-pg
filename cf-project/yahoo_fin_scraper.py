from curses import meta
from bs4 import BeautifulSoup
import urllib.request as req
import pandas as pd
from time import sleep
import sys
import argparse

# Change the directory

# URL for Yahoo Finance website
def generate_url(ticker):
  """
  Args:
    ticker: Stock ticker symbol

  Returns:
    A list of urls strings of yahoo finance
  """
  url_is = f"https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}"
  url_bs = f"https://finance.yahoo.com/quote/{ticker}/balance-sheet?p={ticker}"

  return (url_is, url_bs)


# fetch the entire html file of the url
def fetch_data(url):
  """
  Args:
    url: Yahoo Finance site specific url

  Returns:
    A string of HTML document
  """
  # User-Agent is important
  try:
    read_data = req.urlopen(req.Request(url, headers={'User-Agent': 'Mozilla/5.0'})).read()
  except:
    print("error")
    exit()
    
  return read_data


def _convert_to_num(col):
  """
  Converts all the financial values to integers (without any hyphens and commas) in a dataframe

  Args:
    col: column name

  Retuns:
    np.darray (values updated column!)
  """
  clean_col = []

  # clean'em
  for val in col:
    clean_val = val.replace(",", "").replace("-", "")
    clean_col.append(clean_val)

  final_col = pd.to_numeric(clean_col)

  return final_col


# creates a dataframe
def create_df(data):
  """
  Creates a DataFrame object

  Args:
    data: Entire HTML string

  Returns:
    DataFrame of either Income Statement or Balance Sheet
  """

  soup = BeautifulSoup(data, "lxml")

  # list of all table rows
  table = soup.find_all("div", class_="D(tbr)")

  headers = []
  temp_list = []
  final = []

  # create headers
  for item in table[0].find_all('div', class_='D(ib)'):
    headers.append(item.text)  # statement contents

  for row in table:
    temp = row.find_all("div", class_="D(tbc)")

    for line in temp:
      temp_list.append(line.text)

    final.append(temp_list)
    temp_list = []


  df = pd.DataFrame(final[1:])
  df.columns = headers

  for col in headers[1:]:
    df[col] = _convert_to_num(df[col])

  # no nan
  final_df = df.fillna(0)
  # print(final_df)

  return final_df


def create_csv_file(df, file_name):
  """
  Creates a .csv file with financial data

  Args:
    df: dataframe
    file_name: save .csv file with this name

  Returns:
    None
  """

  try:
    df.to_csv(f"./test/{file_name}.csv", encoding="utf-8", index=False)
  except RuntimeError:
    print(f"RuntimeError: Try changing your directory mate!")
  else:
    print(f"Pass: {file_name} was saved!")


def main():

  parser = argparse.ArgumentParser(description="Fetch the Balance Sheet & Income Statement for any listed company using its ticker, powered by Yahoo Finance.")

  # parser.add_argument("-t", "--ticker", type=str, help="a ticker symbol of a company")

  parser.add_argument("ticker_list", nargs="+", help="a list of tickers", type=str)

  args = parser.parse_args()
  tickers = args.ticker_list

  # Change these to custom tickers!
  # tickers = ["INFY.NS", "MINDTREE.NS", "TCS.NS"]
  # tickers = ["AMD", "NFLX", "TSLA"]

  for tick in tickers:
    [tick_is, tick_bs] = generate_url(tick)

    # Income Statement
    is_data = fetch_data(url=tick_is)
    is_df = create_df(data=is_data)
    create_csv_file(df=is_df, file_name=f"{tick}_IS")

    # Balance Sheet
    sleep(2) # Prevent from getting IP blocked (rate-limiting)!
    bs_data = fetch_data(url=tick_bs)
    bs_df = create_df(data=bs_data)
    create_csv_file(df=bs_df, file_name=f"{tick}_BS")


if __name__ == "__main__":
  main()