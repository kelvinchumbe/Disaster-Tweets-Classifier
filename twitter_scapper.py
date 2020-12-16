# import files and give access to tokens and keys
import tweepy

access_token = "3427869346-MEJppXw8HCFAUixqBhFllq3tA9xHoqUwIg9ecqA"
access_token_secret = "tr0zuDgPz273wrWmWc4m3rEExZPcYHilVUJtU1a1j0iuv"
consumer_key = "KbaSScGM99zYGhcRGuAuScTtk"
consumer_secret = "YcTefmpkmzcBYJOQSE31Nn30nUgvBDFIQEN2mzBKO2OgLgz2aR"


def twitter_authenticate(access_token, access_token_secret, consumer_key, consumer_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def scrap_tweets(api, search_words, date_since, num_tweets=2500):
    tweets = tweepy.Cursor(api.search, q=search_words, lang="en",
                           since=date_since, tweet_mode='extended').items(num_tweets)
    tweets_list = [[tweet.id, tweet.created_at, tweet.user.screen_name, tweet.user.location,
                    tweet.full_text]for tweet in tweets]

    return tweets_list
