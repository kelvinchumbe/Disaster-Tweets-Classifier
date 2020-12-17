import streamlit as st
import pandas as pd
import numpy as np
import tweepy
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# from preprocess_utils import *
# from twitter_scapper import *

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# st.write("HELLO WORLD")
# s = st.text_input("Enter Text: ")
# st.write(s)

# Define some common contractions in English
contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


# Define some preprocessing helper functions

def removeHyperLinks(words):
    """ Remove hyperlinks from the texts """

    new_words = []
    for word in words:
        new_word = re.sub(r"http\S+", "", word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def removeUsernames(words):
    """ Remove @username from the tweet"""

    new_words = []
    for word in words:
        new_word = re.sub('@[\w]+', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def removeHashtags(words):
    """ Remove #hashtags symbol from the tweet"""

    new_words = []
    for word in words:
        new_word = re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def removePunctuation(words):
    """Remove punctuation from the tweet"""

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def expandContractions(words):
    """ Expand contractions from the tweet"""

    new_words = []
    for word in words:
        expanded = contractions.get(word, word)
        new_word = expanded.split(" ")
        new_words = new_words + new_word

    return new_words


def removeStopWords(words):
    """Remove stop words from the tweet"""

    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatizeWords(words):
    """Lemmatize the words"""

    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def removeRT(words):
    """Remove RT which indicates the tweet is a retweet"""

    new_words = []
    for word in words:
        new_word = re.sub(r'RT', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def preprocessTweets(tweet):
    """ Combines all preprocessing steps for the tweets"""
    tweet = tweet.split(" ")

    tweet = removeHyperLinks(tweet)
    tweet = removeUsernames(tweet)
    tweet = removeHashtags(tweet)
    tweet = expandContractions(tweet)
    tweet = removeStopWords(tweet)
    tweet = removePunctuation(tweet)
    tweet = removeRT(tweet)
    tweet = lemmatizeWords(tweet)

    return tweet

# Define a function to reverse one hot encoded labels


def reverseEncoded(encoded_labels):
    return np.argmax(encoded_labels, axis=1)


# TWITTER SCRAPPER

access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


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


st.title("Tweet Classifier Web App")

api = twitter_authenticate(
    access_token, access_token_secret, consumer_key, consumer_secret)

st.subheader("Include the # symbol before the hashtag")
search_words = st.text_input("Enter a hashtag to collect tweets: ")
date_since = st.text_input("Enter date from which you want tweets from: ")
num_tweets = st.text_input("Enter number of tweets to pull: ")


if st.button("Get Tweets"):
    bar = st.progress(0)
    # Scrap tweets using the twitter api
    try:
        tweets_list = scrap_tweets(
            api, search_words, date_since, num_tweets=int(num_tweets))
    except TweepError as e:
        if 'Failed to send request:' in e.reason:
            st.write("Time out error caught.")

    bar.progress(100)
    st.write("Done")

    # Put tweets in a dataframe and display sample 30 tweets
    tweets_df = pd.DataFrame(data=tweets_list, columns=[
        "tweet_id", "date", "username", "location", "text"])

    st.subheader("Sample of 20 tweets that have been retrieved from Twitter")
    st.write(tweets_df.sample(20))

    # Get the texts from the tweets for preprocessing
    tweets_text = tweets_df.text

    # Preprocess tweets using helper functions in preprocess_utils module
    tweets_text = tweets_text.apply(lambda x: preprocessTweets(x))

    # Load tokenizer object
    tokenizer_file = "tokenizer.pkl"
    tokenizer = pickle.load(open(tokenizer_file, "rb"))

    # Generate sequences using the tokenizer object used in training with a vocabulary already determined
    sequences = tokenizer.texts_to_sequences(tweets_text)

    # Use the same sequences length as during training
    max_sequence_len = 50

    # Pad sequences to max sequence length
    text = pad_sequences(sequences, maxlen=max_sequence_len)

    # Load keras model
    bilstm_model_file = "bilstm_model.h5"
    bilstm_model = load_model(bilstm_model_file, compile=False)

    # Make predictions
    y_pred = bilstm_model.predict(text)
    y_pred = reverseEncoded(y_pred)

    # result_df = pd.concat([tweets_df["text"].values, y_pred],
    #                       axis=1, keys=["text", "prediction"])

    result_df = pd.DataFrame(
        data={"text": tweets_df["text"].values, "prediction": y_pred})

    # Display to user model predictions
    st.write(result_df)
