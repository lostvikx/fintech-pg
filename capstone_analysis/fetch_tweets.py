import snscrape.modules.twitter as twitter
import pandas as pd

def main():
  # Base Parameters
  # ["decentralized+finance", "crypto", "defi"]
  keyword = "crypto"
  n_tweets = 3000
  fav_thresh = 200

  # Last 6 months
  search_params = f" min_faves:{fav_thresh} since:2021-11-01 until:2022-11-01"

  tweets = []

  for i, tweet in enumerate(twitter.TwitterSearchScraper(keyword + search_params).get_items()):

    if i >= n_tweets:
      break

    tweets.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.retweetCount, tweet.rawContent, tweet.hashtags, tweet.url, tweet.sourceUrl])

  tweets_df = pd.DataFrame(tweets, columns=["user_name", "date", "likes","retweets", "content", "hashtags", "url", "source_url"])

  name = keyword.split(" ")
  name = "_".join(name)
  # print(name)
  file_name = f"data/{name}_tweets_{n_tweets}.csv"

  tweets_df.to_csv(file_name)
  # print(tweets_df)

if __name__ == "__main__":
  main()
