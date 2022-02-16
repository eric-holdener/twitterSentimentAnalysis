# This is a sample Python script.
import twitterScraper
import pandas as pd

def main():
    keyword = 'Kanye'
    tweetsToScrape = 1000
    fileName = 'sent_analysis_tweets.csv'
    twitterScraper.scrape(keyword, tweetsToScrape, fileName)
    df = pd.read_csv(fileName)
    print(df)

if __name__ == '__main__':
    main()


