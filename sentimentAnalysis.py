import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


# either naive bayesian algorithm or a neural net for the sentiment analysis - research both
# (potentially code both and run tests on which is better / more accurate?)

# sentiment dataset acquired here: https://www.kaggle.com/kazanova/sentiment140

# change file read to something more universal
df = pd.read_csv('C:\\Users\\ehold\\Desktop\\Folders\\Datasets\\training.1600000.processed.noemoticon.csv')

def bagOfWords(dataSet):
    # set data to local variable
    trainingDataset = dataSet

    # add columns the training data
    trainingDataset.columns = ['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Text']

    # generate Document Term Matrix (DTM) using CountVectorizer from Sklearn
    # nGram model is how many words are taken at a time
    # i.e - bigram model would return the following from this sentence - 'bigram model', 'model would', would return',
    # etc

    # generate token for the DTM
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    # tokenizer - overrides string tokenization step, generate tokenizer from NLTK's regex tokenizer (default is none)
    # lowercase - True, left out of countvectorizer as true is default
    # stop_words - 'english' - default is none, we provide english as stop words to imporve results
    # ngram_range - (1, 1) is only monograms, (2,2) is only bigrams, (1,2) is both, etc
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(dataSet['Text'])
    return(text_counts, trainingDataset)

def naiveBayes(trainingDF, classificationDF):
    # pass in text counts from bag of words and the training dataset with column names
    text_counts, trainingDataSet = bagOfWords(trainingDF)

    # split data into training and testing data (predictor data x = text_counts, predicted data y = sentiment)
    x_train, x_test, y_train, y_test = train_test_split(text_counts, trainingDataSet['Sentiment'],
                                                        test_size=0.25, random_state=5)

    # define the multinomial naive bayes model
    MNB = MultinomialNB()

    # fit the multinomial model onto our data with training data
    MNB.fit(x_train, y_train)

    # find accuracy of the model
    predicted = MNB.predict(x_test)
    accuracy_score = metrics.accuracy_score(predicted, y_test)
    print(str('{:04.2f}'.format(accuracy_score*100))+'%')

    # to do - implement classification on scraped tweets





if __name__ == '__main__':
    naiveBayes(df)