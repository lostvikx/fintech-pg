#%%
import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud

#%%
def remove_emoji(text):
  emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

  clean = emoji_pattern.sub(r'', text) # no emoji
  return clean

def cleansing(tweet:str):
  # Remove hashtag sign but keep the text
  content = tweet.replace("\n", " ").replace("#", "").replace("_", " ").replace("@", "").replace('&amp;', 'and')
  # Remove emojis
  content = remove_emoji(content)
  # Remove any links
  content = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", content)
  content = re.sub('&lt;/?[a-z]+&gt;', '', content)

  return content.strip()

#%%
df = pd.read_csv("data/crypto_tweets.csv",index_col=0)
df.head()

#%%
df["clean"] = df["content"].apply(cleansing)
tweets = pd.DataFrame(df["clean"])
tweets.columns = ["text"]
tweets.head()

#%%
tweets["polarity"] = tweets["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
tweets["subjectivity"] = tweets["text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

tweets.head()

#%%
tweets["polarity"].describe()

#%%
def get_sentiment(score):
  if score > 0.15:
    return "positive"
  elif score < 0.10:
    return "negative"
  else:
    return "neutral"

# %% [markdown]
# ### Sentimental Analysis: Subjectivity & Polarity
#
# * Subjectivity detection aims to remove 'factual' or 'neutral' content, objective text that does not contain any opinion. 
# * Polarity detection aims to differentiate the opinion into 'positive' and 'negative'.
#
# Sentiment analysis focuses on the polarity of a text (positive, negative, neutral) but it also goes beyond polarity to detect specific feelings and emotions (angry, happy, sad, etc), urgency (urgent, not urgent) and even intentions (interested v. not interested).

# %%
tweets["sentiment"] = tweets["polarity"].apply(get_sentiment)
tweets.head()

#%%
tweets.isnull().values.any()

# %%
plt.scatter(tweets["polarity"],tweets["subjectivity"],color="black",s=10)
plt.title("Polarity vs. Subjectivity Scatter Plot")
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# %%
tweets["sentiment"].value_counts().plot(kind="bar",color="purple")
plt.title("Sentiment Bar Plot")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

#%%
tweets["sentiment"].value_counts(normalize=True)

#%%
tweets["sentiment"].value_counts().plot(kind="pie",autopct='%1.1f%%')
plt.title("Sentiment Pie Plot")
plt.show()

#%%
text = " ".join(list(tweets["text"]))

wordcloud = WordCloud(width=800, height=400).generate(text)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#%% [markdown]
# ### Larger sample data
#
# Now I take a dataset with 3000 tweets.

#%%
df = pd.read_csv("data/crypto_tweets_3000.csv", index_col=0)
df.head()

# %%
df["clean"] = df["content"].apply(cleansing)
df.head()

# %%
tweets = pd.DataFrame(df[["user_name","clean"]])
tweets.columns = ["user_name","text"]
tweets.head()

#%%
tweets["polarity"] = tweets["text"].apply(
    lambda t: TextBlob(t).sentiment.polarity)
tweets["subjectivity"] = tweets["text"].apply(
    lambda t: TextBlob(t).sentiment.subjectivity)

tweets.head()

#%%
tweets.describe()

#%%
def get_sentiment(score):
  if score > 0.15:
    return "positive"
  elif score < 0.10:
    return "negative"
  else:
    return "neutral"

# %%
tweets["sentiment"] = tweets["polarity"].apply(get_sentiment)
tweets.head()

#%%
tweets.isnull().values.any()

#%%
s_tweets = tweets[tweets["subjectivity"] > 0.7].sort_values(by="subjectivity")

s_tweets.head()

#%%
s_tweets.describe()

# %%
plt.scatter(s_tweets["polarity"],s_tweets["subjectivity"],color="black")
plt.title("Polarity vs. Subjectivity Scatter Plot")
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# %%
s_tweets["sentiment"].value_counts().plot(kind="bar",color="orange")
plt.title("Sentiment Bar Plot")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# %%
p_tweets = s_tweets.sort_values(by="polarity")

# for i in range(0,20):
#   print(f"{i}.", p_tweets["text"].iloc[i], f"~ @{p_tweets['user_name'].iloc[i]}")

p_tweets.head()

# %%
negative_tweets_examples = [0, 1, 4, 8]

for i in negative_tweets_examples:
  print("-", p_tweets["text"].iloc[i], f"~ @{p_tweets['user_name'].iloc[i]}\n")

#%%
p_tweets = s_tweets.sort_values(by="polarity",ascending=False)

# for i in range(0,20):
#   print(f"{i}.", p_tweets["text"].iloc[i], f"~ @{p_tweets['user_name'].iloc[i]}")

p_tweets.head()

# %%
positive_tweets_examples = [1, 8, 9, 12]

for i in positive_tweets_examples:
  print("-", p_tweets["text"].iloc[i], f"~ @{p_tweets['user_name'].iloc[i]}\n")
