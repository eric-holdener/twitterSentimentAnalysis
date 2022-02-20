# This is a sample Python script.
import twitterScraper
import sentimentAnalysis
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

def main():
    # key values to search in the scraper, returned from the gui
    keyword = 'Kanye'
    tweetsToScrape = 1000
    fileName = 'scrapedTweets.csv'

    # scrape the keywords
    # twitterScraper.scrape(keyword, tweetsToScrape, fileName)
    # classificationDf = pd.read_csv(fileName)
    # tweets = classificationDf.tweet

    # # download tweet samples and stopwords from nltk
    # nltk.download('twitter_samples')
    # nltk.download('stopwords')
    #
    # load text fields of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    sentimentAnalysis.training(all_positive_tweets, all_negative_tweets)

    # logprior, loglikelihood, test_x, test_y = sentimentAnalysis.training(all_positive_tweets, all_negative_tweets)

    # sentimentAnalysis.sentimentAnalysis(tweets, logprior, loglikelihood)

if __name__ == '__main__':
    main()


