import tiingo
import pandas as pd
import tweepy
import csv


#TODO: implement this for any CEO, maybe by company name or just username 

def get_Elons_Tweets():
    '''
    Function that will get information on Elon Musk's tweets, such as
    number of tweets per day.
    '''

    #Twitter API credentials
    #CHANGE BEFORE POSTING
    consumer_key = "UZaefZdzkPNYnZi8fN9HIPzim"
    consumer_secret = "tXGE5YUfqUxz4sLUOJohKJKApyWfTyrVi7Mg9ivGFPd0mn135t"
    access_key = "1665830713-gWWMNp5q26j3XxgYQB4v3kgudijpjzBlipFjQQb"
    access_secret = "RZVr5x1GteJHjhAfrPAJ47WEegUpSvKythIMK9fPA8KXr"

    #set up authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)


    screen_name = 'elonmusk'

    #copied code from: https://gist.github.com/yanofsky/5436496
    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print(f"...{len(alltweets)} tweets downloaded so far")

    #transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    #write the csv
    with open(f'new_{screen_name}_tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)

    pass


    return




if __name__ == "__main__":
    main()
