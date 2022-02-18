# This is a sample Python script.
import twitterScraper
import sentimentAnalysis

import pandas as pd

def main():
    keyword = 'Kanye'
    tweetsToScrape = 1000
    fileName = 'sent_analysis_tweets.csv'
    twitterScraper.scrape(keyword, tweetsToScrape, fileName)
    classificationDf = pd.read_csv(fileName)
    trainingDf = pd.read_csv('C:\\Users\\ehold\\Desktop\\Folders\\Datasets\\training.1600000.processed.noemoticon.csv')
    sentimentAnalysis.naiveBayes(trainingDf, classificationDf)

if __name__ == '__main__':
    main()


