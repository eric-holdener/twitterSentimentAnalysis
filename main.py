# This is a sample Python script.
from fileinput import filename
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import twint
import matplotlib.pyplot as plt
import os

def main():
    # key values to search in the scraper, returned from the gui
    keyword = 'Fresno'
    tweetsToScrape = 100000
    fileName = 'scrapedTweets.csv'

    # scrape the keywords
    if os.path.exists(fileName):
        os.remove(fileName)

    scrape(keyword, tweetsToScrape, fileName)
    classificationDf = pd.read_csv(fileName)
    tweets = classificationDf.tweet

    # # download tweet samples and stopwords from nltk
    # nltk.download('twitter_samples')
    # nltk.download('stopwords')
    #
    # load text fields of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    training(all_positive_tweets, all_negative_tweets)

    logprior, loglikelihood, test_x, test_y = training(all_positive_tweets, all_negative_tweets)


    tweetArray, sentArray = sentimentAnalysis(tweets, logprior, loglikelihood)

    maxSent, minSent = minMaxSent(tweetArray, sentArray)

    visualize(tweetArray, maxSent, minSent)

def scrape(keyword, scope, fileName):
    c = twint.Config()

    # topic to scrape
    c.Search = [keyword]

    # number of tweets to scrape
    c.Limit = scope

    c.Store_csv = True
    c.Output = fileName

    twint.run.Search(c)

def preprocessing(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text RT
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove the # symbol
    tweet = re.sub(r'#', '', tweet)

    # tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []

    # removing stopwords and punctuation
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            # stemming
            stem_word = stemmer.stem(word)
            tweet_clean.append(stem_word)

    return tweet_clean

# build word count dictionary that stores word count of each word in the corpus
# in dictionary, each key is a 2 element tuple containing a (word, y) pair
# word is an element in a processed tweet
# y is an integer representing the corpus - 1 for positive, 0 for negative
def count_tweets(tweets, ys):
    ys_list = np.squeeze(ys.tolist())
    freqs = {}

    for y, tweet in zip(ys_list, tweets):
        for word in preprocessing(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

# lookup function returns positive and negative frequencies for specific words
def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n

# prepare data for training and testing - 80-20 split for training and testing respectively
def training(all_positive_tweets, all_negative_tweets):
    # segment training - select first 4,000 (80%) for training, and everything after 4,000 (20%) for testing
    train_pos = all_positive_tweets[:4000]
    test_pos = all_positive_tweets[4000:]

    train_neg = all_negative_tweets[:4000]
    test_neg = all_negative_tweets[4000:]

    # make our training and testing x variables by combining the positive and negative sentiment datasets
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # numpy array for the labels in the training set
    train_y = np.append(np.ones((len(train_pos))), np.zeros((len(train_neg))))
    test_y = np.append(np.ones((len(test_neg))), np.zeros((len(test_neg))))

    # build a frequency dictionary
    # pass in training variables x and y to find frequencies
    freqs = count_tweets(train_x, train_y)
    # freqs returns a pair of (keword, sentiment[1, 0 - 1 is positive, 0 is negative])
    # followed by : n (number of occurences)

    logliklihood = {}
    logprior = 0

    # calculate V, number of unique words in the vocabulary
    # vocab is a list of all the unique words in freqs, made by iterating over each pair in keys and extracting
    # index 0, which is the word
    # V is just the length of the list vocab, i.e. how many unique words there are
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # Calculate N_pos, N_neg, V_pos, V_neg
    # N_pos : total number of positive words
    # N_neg : total number of negative words
    # V_pos : total number of unique positive words
    # V_neg : total number of unique negative words

    # set the total number of positive and negative words and positive and negative unique words to 0
    N_pos = N_neg = V_pos = V_neg = 0
    # iterates over each pair of (word, sentiment) in freqs
    for pair in freqs.keys():
        # index 1 of the pair is sentiment
        # if index 1 is greater than 0, it's a positive sentiment
        if pair[1]>0:
            # v_pos is incremented by one
            # n_pos is incremented by number of occurences, the other value stored in freqs[pair]
            V_pos += 1
            N_pos += freqs[pair]
        # else if paid[1] is 0 or less, sentiment is negative
        else:
            # increment v_neg by one
            # n_neg is incremented by number of occurences, the other value stored in freqs[pair]
            V_neg += 1
            N_neg += freqs[pair]

    # number of Documents (tweets)
    D = len(train_y)

    # D_pos, number of positive documents
    # returns length of the list train_y after filtering for x>0 (positive comments)
    D_pos = len(list(filter(lambda x: x > 0, train_y)))

    # D_neg, number of negative documents
    # returns length of the list train_y after filtering for x<=0 (positive comments)
    D_neg = len(list(filter(lambda x: x <= 0, train_y)))

    # calculate the logprior
    # log of number of positive documents - log of number of negative documents
    logprior = np.log(D_pos) - np.log(D_neg)

    # for each word in the unique word list
    for word in vocab:
        freqs_pos = lookup(freqs, word, 1)
        freqs_neg = lookup(freqs, word, 0)

        # calculate the probability of each word being positive and negative
        p_w_pos = (freqs_pos+1)/(N_pos+V)
        p_w_neg = (freqs_neg+1)/(N_neg+V)

        logliklihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, logliklihood, test_x, test_y

def naive_bayes_predict(tweet, logprior, logliklihood):
    word_1 = preprocessing(tweet)
    p = 0
    p+=logprior

    for word in word_1:
        if word in logliklihood:
            p+=logliklihood[word]

    return p

def test_naive_bayes(test_x, test_y, logprior, logliklihood):
    accuracy = 0
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, logliklihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = np.mean(np.absolute(test_y - y_hats))
    accuracy = 1-error

    return accuracy

def sentimentAnalysis(tweets, logprior, loglikelihood):
    tweetArray = []
    sentArray = []
    for tweet in tweets:
        p = naive_bayes_predict(tweet, logprior, loglikelihood)
        tweetArray.append([tweet, p])
        sentArray.append(p)
        # if p>1:
        #     print('\033[92m')
        #     print(f'{tweet} :: Positive sentiment ({p: .2f})')
        # else:
        #     print('\033[91m')
        #     print(f'{tweet} :: Negative sentiment ({p: .2f})')
    return tweetArray, sentArray

def visualize(tweetArray, maxSent, minSent):
    negativeTweets = 0
    positiveTweets = 0
    neutralTweets = 0

    for tweet in tweetArray:
        if minSent <= tweet[1] < -1:
            negativeTweets += 1
        elif -1 <= tweet[1] <= 1:
            neutralTweets += 1
        elif 1 < tweet[1] <= maxSent:
            positiveTweets += 1

    x = np.array([positiveTweets, negativeTweets, neutralTweets])

    myLabels = ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets']
    myColors = ['green', 'red', 'gray']

    plt.pie(x, labels=myLabels, colors = myColors, autopct='%1.1f%%')
    plt.show()


def minMaxSent(tweetArray, sentArray):
    i = 0
    runningTotal = 0
    maxSent = 0
    minSent = 0
    for valuePair in tweetArray:
        if valuePair[1] > maxSent:
            maxSent = valuePair[1]
        elif valuePair[1] < minSent:
            minSent = valuePair[1]
        runningTotal = runningTotal + valuePair[1]
        i += 1
    averageSentiment = runningTotal / i
    return maxSent, minSent


if __name__ == '__main__':
    main()


