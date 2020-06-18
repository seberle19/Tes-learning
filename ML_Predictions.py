import tiingo
import pandas as pd
import tweepy
import csv
from random import random
from random import seed
from math import exp
import numpy as np
from sklearn.model_selection import KFold


'''
Program that tries to connect the change in TSLA stock price to the Tweets of
Elon Musk. The program determines how many times Elon Musk has tweeted on a
certain day and the corresponding change in price of the TSLA stock for that
same day. This program then trains a neural network to build a classifier that
attempts to learn based on the number of tweets from Elon Musk, whether the
TSLA stock price will increase or decrease. After the network has been built,
the user is prompted to enter a number of tweets, and then the expected
performance of the stock is computed using the neural network and displayed.

*** THIS IS NOT MEANT TO BE USED TO ACTUALLY GIVE RECOMMENDATIONS ON THE
    PERFORMANCE OF THE STOCK MARKET!!!! It is just a project built for fun.  ***
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




def get_Data_List(combined_dataframe):
    '''
    Function that will return the data from the data frame in a list format
    that is easy to use with the neural network.

    @combined_dataframe: the dataframe with the date, the number of tweets that
                         day and the change in stock price of that day
    '''


    combined_dataframe.columns = ['number of tweets', 'date', 'change in price']
    #Create new column 'class' which tells if stock went up or not
    combined_dataframe['class'] = np.where(combined_dataframe['change in price'] > 0
                                                                , 1, 0)

    tweets_and_prices = combined_dataframe[['number of tweets','class']]

    return tweets_and_prices.values.tolist()




def get_Min_And_Max(dataset):
    '''
    Function that will find the minimum and maximum of the data set and return
    both. Helper function for normalization

    @dataset: the dataset to find the min and max of
    '''

    min_max = [[min(column), max(column)] for column in zip(*dataset)]
    return min_max #min first index, max second index



def normalize_Data(dataset):
    '''
    Funciton that will normalize the data so that it is all on a similar scale.

    @dataset: dataset to be normalized
    '''

    min_max = get_Min_And_Max(dataset)

    #last column is classifier
    for row in dataset:
        for i in range(len(min_max) - 1):
            #minus minimum over range
            row[i] =  (row[i] - min_max[i][0])/ (min_max[i][1] - min_max[i][0])

    return dataset




def build_Neural_Network(n_inputs, n_hidden, n_outputs):
    '''
    Function that will create the data and data structures for the neural
    network to work. It will return the weights in the neural network

    @n_inputs: number of input values (raw data)
    @n_hidden: number of neurons in the hidden layer
    @n_outputs: number of outputs in the neural network (number of classes)
    '''


    # Initialize a network with random weights. The network is fully connected
    network = list()

    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)

    return network




def neuron_Activation(weights, inputs):
    '''
    This is a function that will calculate the activation for each neuron in
    a layer of the neural network given an input and a weight for that neuron.
    It also accounts for bias by assuming that the bias input is the last value
    in the weights array. This is a helper function for forward propagate.

    @weights: the weights computed for each neuron in the network
    @inputs: the inputs to that neuron in the network
    '''
    #bias
    activation = weights[-1]
    #activation is sum of product of weight and input plus bias
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]

    #return activation of each neuron in this particular layer
    return activation


def neuron_Transfer(activation):
    '''
    Use the sigmoid function to calculate the actual output of the neuron in
    the network. Sigmoid is used to map from 0 to 1 and to compute the

    @activation: the weight times the input for a particular neuron
    '''
    return 1.0 / (1.0 + exp(-activation))

def forward_Propagate(network, row):
    '''
    Function that will propagate forward through the neural network. This means
    that the output from one layer of the network becomes the input for the
    next layer of the network.

    @network: the neural network
    @row: a single row from the dataset
    '''

    inputs = row
    #go through network
    for layer in network:
        new_inputs = []
        #go through nerons one-by-one
        for neuron in layer:
            #calculate activation for neuron using its own weights and the inputs
            activation = neuron_Activation(neuron['weights'], inputs)
            #calculate output of this neuron
            neuron['output'] = neuron_Transfer(activation)
            new_inputs.append(neuron['output'])
        #set the current outputs to the next layer's inputs
        inputs = new_inputs

    return inputs


def transfer_derivative(output):
    '''
    Function that will take the output of a particular neuron and then calculate
    the transfer derivative of the sigmoid function for that neuron

    @output: the output from a neuron in the network
    '''

    return output * (1.0 - output)




def back_Propagate(network, expected):
    '''
    Function that will run back propagation of the neural network. It will
    calculate the error between the expected values and actual output values
    '''
    #start from the output later
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        #if we are in the last layer
        if i == len(network) - 1:
            for j in range(len(layer)):
                #calculate difference between output and expected
                #set neuron
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        else:
            #if we are in any other layer
            for j in range(len(layer)):
                error = 0.0
                #for all neurons in NEXT layer
                for neuron in network[i+1]:
                    #error is weight of current times error of current
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        #save errors for next level for each neuron
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])




def recompute_Weights(network, row, learning_rate):
    '''
    Function that will compute the update to the existing weight using the
    error of the current weight.

    @network: the neural network
    @row: the row in the dataset
    @learning: (alpha) value that will determine how much weight can change
               each time
    '''

    for i in range(len(network)):
    	inputs = row[:-1]
    	if i != 0: #if not in first row inputs are prev layers outputs
    		inputs = [neuron['output'] for neuron in network[i - 1]]
    	for neuron in network[i]:
    		for j in range(len(inputs)):
                #update each neuron's weight based on error and learning rate
    			neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            #bias term has no input
    		neuron['weights'][-1] += learning_rate * neuron['delta']


def train_Neural_Network(network, training_set, learning_rate, num_iterations,
                                                                num_outputs):
    '''
    Function that will actually train the neural network to get theta values/
    the weights of each node in the network. This will utilize stoachstic
    gradient descent.

    @network: the neural network
    @training_set: training set data
    @learning_rate: alpha that will determine how much weights change
    @num_iterations: number of times to run stoch. grad. descent
    @num_outputs: number of output classes in the problem
    '''

    for iteration in range(num_iterations):
        sum_error = 0 #used to check error between actual and expected
        for row in training_set:
            outputs = forward_Propagate(network, row) #get initial outputs
            #get expetect values
            expected = [0 for i in range(num_outputs)]
            #bias term
            expected[-1] = 1
            #compute error for every output
            sum_error += sum([(expected[i] - outputs[i])**2 for i in
                                            range(len(expected))])

            #back propagate error
            back_Propagate(network, expected)

            recompute_Weights(network, row, learning_rate)


        #print error to know improvements are being made
        print('iteration=%d, lrate=%.3f, error=%.3f' % (iteration,
                                                learning_rate, sum_error))


    return


def predict_TSLA_Price_Change(network, row):
    '''
    This function will use the neural network that was created to predict the
    change in price of the TSLA stock based on the number of tweets from Elon
    Musks's twitter account. It will return the index of the class which
    has the highest output.

    @network: the neural network
    @row: a row of input data
    '''
    output = forward_Propagate(network, row)
    print(output)

    return output.index(max(output))





def get_User_Data():
    '''
    Function that will let the user input the number of tweets that have happend
    so far and then return this number to make a prediction using the neural
    network.
    '''

    num_tweets = input("Please input the number of Tweets so far today:")

    return float(num_tweets)



def main():

    date_counts = get_Daily_Tweets()

    TSLA_price_changes = get_TSLA_Data()

    combined = merge_frames(date_counts, TSLA_price_changes)

    data_list = get_Data_List(combined)

    normed_data = normalize_Data(data_list)

    num_hidden = 10 #arbitrary number of hidden nodes
    #number of classes is number of diff class values
    num_classes = len(set(row[-1] for row in data_list))
    num_inputs = len(data_list[0]) - 1 #number of inputs in data


    network = build_Neural_Network(num_inputs, num_hidden, num_classes)

    l_rate = 0.25
    num_iterations = 100
    train_Neural_Network(network, data_list, l_rate, num_iterations, num_classes)



    number_of_tweets = get_User_Data()


    # Make a prediction with a network
    prediction = predict_TSLA_Price_Change(network, [number_of_tweets, None])

    #determine output
    print("Neural Network predicts a value of %d" % prediction)
    if prediction == 1:
        print("Stock price expected to rise")
    else:
        print("Stock price expected to fall")




if __name__ == "__main__":
    main()
