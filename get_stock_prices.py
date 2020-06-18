import tiingo
import pandas as pd
import tweepy
import csv


'''
Program to get data on a stock from the Tiingo service
'''

def get_stock_From_Tiingo(stock):
    '''
    Function that will return the collected data on TSLA from Tiingo. It will
    return a dataframe with the change in stock price over the course of each
    day.
    '''
    config = {}

    # To reuse the same HTTP Session across API calls (and have better performance), include a session key.
    config['session'] = True

    # API token ***REMOVE***
    config['api_key'] = "6c0e64a433dac88e9272fd9caaba27d1dee851a5"

    # Initialize
    client = tiingo.TiingoClient(config)

    #get the data on TSLA
    TSLA_info = client.get_dataframe(stock, startDate='2010-07-2',
                                    endDate='2020-06-10')


    return TSLA_info


def main():

    prices = get_stock_From_Tiingo('TSLA')

    #convert to csv so that it can be used later
    prices.to_csv('stock_prices.csv')




if  __name__ == '__main__':
    main()
