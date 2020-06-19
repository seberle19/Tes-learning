import pandas as pd
from random import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


'''
    I wrote this program after I discovered that all of the hard work that I
    had previously done to build a neural network from scratch that could work
    as a classifier was unnecessary because libraries like sklearn have built
    in classifiers that do the job just as well if not better ¯\_(ツ)_/¯

'''


def get_Daily_Tweets():
    '''
    Function that will get a dataframe of the date and the number of tweets on
    that particular day.
    '''

    tweets = pd.read_csv('new_elonmusk_tweets.csv', usecols=['created_at', 'text'])
    #show all columns:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    #isolate just days
    tweets['created_at'] = tweets.created_at.str.split().str[0]

    #group by date with count number of tweets per days
    date_counts = tweets.groupby(['created_at']).aggregate(['count'])

    return date_counts



def get_TSLA_Data():
    '''
    Function that will access Tiingo's data for TSLA and return a dataframe
    of the net change of the TSLA stock for a particular date.
    '''

    TSLA_prices = pd.read_csv('stock_prices.csv', usecols=['date',
                                    'close', 'open'])

    #return just one data frame that has the change in price of stock
    TSLA_prices['change'] = TSLA_prices['close'] - TSLA_prices['open']

    TSLA_prices['date'] = TSLA_prices.date.str.split().str[0]

    #extract just date and change in price
    TSLA_price_changes = TSLA_prices[['date', 'change']]

    return TSLA_price_changes


def merge_frames(date_count, TSLA_price):
    '''
    Function that will merge the two data frames in places where the dates are
    the same. The function will return one data frame with the the number of
    tweets and the change in stock price for that particular date

    @date_count: dataframe with the counts of tweets per date
    @TSLA_price: dataframe with the change in price of TSLA stock for date
    '''

    result = pd.merge(date_count, TSLA_price, left_on='created_at',
                                    right_on='date')


    return result



def separate_Data(combined_data):
    '''
    Function that will separate the data into X, the training set, and y the
    output

    @combined_data: dataframe of all of the data together
    '''

    combined_data.columns = ['number of tweets', 'date', 'change in price']

    num_tweets = combined_data['number of tweets']
    change_in_price = combined_data['change in price']

    X = num_tweets.to_numpy()
    y = change_in_price.to_numpy()

    return X,y



def scale_Input_Data(X):
    '''
    Function that will scale X and y so that the classifier is able to be more
    accurate.

    @X: The unscaled input dataset
    @y: the unscaled output dataset
    '''
    #create standard scaler
    scaler = StandardScaler()

    scaler.fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled


def classify_Y(y):
    '''
    Function that will convert positive change in stock price to 1 and
    negative change in stock price to 0.

    @y: original data with numeric stock price changes
    '''

    new_y = np.full_like(y,0)
    #if positive change change output to 1
    for i in range(len(y)):
        if y[i] > 0:
            new_y[i] = 1

    return new_y



def get_User_Data():
    '''
    Function that will let the user input the number of tweets that have happend
    so far and then return this number to make a prediction using the neural
    network.
    '''

    num_tweets = int(input("Please input the number of Tweets so far today:"))

    num_tweets = np.array([num_tweets])

    return num_tweets.reshape(1,-1)



def main():

    date_counts = get_Daily_Tweets()

    TSLA_price_changes = get_TSLA_Data()

    combined = merge_frames(date_counts, TSLA_price_changes)

    #seperate input and output data
    X, y = separate_Data(combined)

    #reshape arrays for analysis
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)

    #convert y to classification:
    y = classify_Y(y)

    #scale so that classifier has better performance
    X_scaled = scale_Input_Data(X)

    #split data for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                test_size=0.3)

    #create neural network MLPClassifier
    clf = MLPClassifier(max_iter=200).fit(X_train, y_train.ravel())

    #test accuracy of classifier
    accuracy = clf.score(X_test, y_test)

    print("MLP Classifier is accurate %f percent of the time on unseen data."
                                            % (accuracy * 100))

    #allow for user input
    number_of_tweets = get_User_Data()

    change = clf.predict(number_of_tweets)

    print("Classifier predicted %d" % change[0])
    if change == 1:
        print("Stock price expected to rise")
    else:
        print("Stock price expected to drop")

if __name__ == "__main__":
    main()
