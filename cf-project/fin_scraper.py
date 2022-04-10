from bs4 import BeautifulSoup
import urllib.request as ur
import pandas as pd
import time

# URL for Yahoo Finance website
def generate_url(ticker):
  """Returns List of urls: [IS, BS]"""
  url_is = f"https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}"
  url_bs = f"https://finance.yahoo.com/quote/{ticker}/balance-sheet?p={ticker}"

  return [url_is, url_bs]

# fetch the entire html file of the url
def fetch_data(url):
  # User-Agent is important
  read_data = ur.urlopen(ur.Request(url, headers={'User-Agent': 'Mozilla/5.0'})).read()

  return read_data

def convert_to_num(col):
  """Cleans & converts df[col]"""
  first_col = [i.replace(",", "") for i in col]
  second_col = [i.replace("-", "") for i in first_col]
  final_col = pd.to_numeric(second_col)

  return final_col

# creates a .csv file
def create_df(data, file_name):
  """Creates a .csv file with financial data"""

  soup = BeautifulSoup(data, "lxml")

  # selects all table rows on the website
  features = soup.find_all("div", class_="D(tbr)")

  headers = []
  temp_list = []
  final = []
  index = 0

  # create headers
  for item in features[0].find_all('div', class_='D(ib)'):
    headers.append(item.text)  # statement contents

  while index <= len(features) - 1:
    #filter for each line of the statement
    temp = features[index].find_all('div', class_='D(tbc)')
    for line in temp:
      #each item adding to a temporary list
      temp_list.append(line.text)
    #temp_list added to final list
    final.append(temp_list)
    #clear temp_list
    temp_list = []
    index += 1

  df = pd.DataFrame(final[1:])
  df.columns = headers

  for col in headers[1:]:
    df[col] = convert_to_num(df[col])

  final_df = df.fillna(0)

  final_df.to_csv(f"./fin_data/{file_name}.csv", encoding="utf-8", index=False)

  print(f"{file_name} Saved 🎉")


# Change these to custom tickers!
tickers = ["INFY.NS", "MINDTREE.NS", "TCS.NS"]

for tick in tickers:
  for i in range(2):
    infy_url = generate_url(tick)[i]
    data = fetch_data(infy_url)

    if i:
      file_name = f"{tick}_BS"
    else:
      file_name = f"{tick}_IS"

    print("Waiting for 10s")
    time.sleep(10)  # Fetch a GET request after 5s, to prevent 403 restriction.
    create_df(data, file_name)
