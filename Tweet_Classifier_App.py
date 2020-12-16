import streamlit as st
import pandas as pd
import numpy as np
import tweepy
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from preprocess_utils import *
from twitter_scapper import *


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
    tweets_list = scrap_tweets(
        api, search_words, date_since, num_tweets=int(num_tweets))

    bar.progress(100)
    st.write("Done")

    # Put tweets in a dataframe and display sample 30 tweets
    tweets_df = pd.DataFrame(data=tweets_list, columns=[
        "tweet_id", "date", "username", "location", "text"])

    # x = st.slider("Select sample of tweets to display")

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

    result_df = pd.concat([tweets_df["text"].values, y_pred],
                          axis=1, keys=["text", "prediction"])

    # Display to user model predictions
    st.write(result_df)
