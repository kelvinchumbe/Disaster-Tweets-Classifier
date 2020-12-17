import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# # Download stopwords from nltk module
# nltk.download('stopwords')

# # Download wordnet from nltk module
# nltk.download('wordnet')

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
