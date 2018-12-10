
# !/usr/bin/python
# -*- coding: utf-8 -*-

import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Sentiments:
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0

    def __init__(self, previous, sentiment_method):
        self.processed_data = previous.processed_data
        self.sentiment_method = sentiment_method


    def sentiment_analysis_by_text(self):
        print("entered sentiment analysis method")
        tweets = self.processed_data
        tweet_sentiments = []
        sentiment_score = []
        positive_tweets = 0
        neutral_tweets = 0
        negative_tweets = 0

        try:
            print("sentiment method: ", self.sentiment_method)
            if "vader" in self.sentiment_method:
                sid = SentimentIntensityAnalyzer()
                print("calculating sentiment using ", self.sentiment_method)
                for index, row in tweets.iterrows():
                    sentiment_polarity = sid.polarity_scores(row['text'].encode().decode('ascii', errors="replace"))
                    # print("sentiment_polarity: ", sentiment_polarity["compound"])
                    if sentiment_polarity['compound'] < -0.2:
                        negative_tweets += 1
                        sentiment = Sentiments.NEGATIVE
                    elif sentiment_polarity['compound'] <= 0.2:
                        sentiment = Sentiments.NEUTRAL
                        neutral_tweets += 1
                    else:
                        sentiment = Sentiments.POSITIVE
                        positive_tweets += 1
                    tweet_sentiments.append(sentiment)
                    sentiment_score.append(sentiment_polarity["compound"])
            else:
                for index,row in tweets.iterrows():
                    blob = TextBlob(row['text'].encode().decode('ascii', errors="replace"))
                    sentiment_polarity = blob.sentiment.polarity
                    if sentiment_polarity < 0:
                        negative_tweets += 1
                        sentiment = Sentiments.NEGATIVE
                    elif sentiment_polarity <= 0.2:
                        sentiment = Sentiments.NEUTRAL
                        neutral_tweets += 1
                    else:
                        sentiment = Sentiments.POSITIVE
                        positive_tweets += 1
                    tweet_sentiments.append(sentiment)
                    sentiment_score.append(sentiment_polarity["compound"])
        except Exception as e:
            print(e)
        print("tweets sentiments list length: ", len(tweet_sentiments))
        print("tweets sentiments column length: ", tweets.shape)
        tweets['sentiment'] = tweet_sentiments
        tweets['sentiment_score'] = sentiment_score

        total_tweets_analysed = positive_tweets + neutral_tweets + negative_tweets
        positive_tweets_percentage = positive_tweets / total_tweets_analysed * 100
        neutral_tweets_percentage = neutral_tweets / total_tweets_analysed * 100

        print("\nNo. of positive tweets = {} Percentage = {}".format(positive_tweets, positive_tweets_percentage))
        print("\nNo. of neutral tweets  = {} Percentage = {}".format(neutral_tweets, neutral_tweets_percentage))
        print("\nNo. of negative tweets = {} Percentage = {}".format(negative_tweets, 100 - (positive_tweets_percentage + neutral_tweets_percentage)))

        self.processed_data = tweets
        return self.processed_data