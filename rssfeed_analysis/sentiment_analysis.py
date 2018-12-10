
# !/usr/bin/python
# -*- coding: utf-8 -*-

import re
from textblob import TextBlob


class Sentiments:
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0

    def __init__(self, previous):
        self.processed_data = previous.processed_data


    def sentiment_analysis_by_text(self):
        print("entered sentiment analysis method")
        rssfeeds = self.processed_data
        rssfeed_sentiments = []
        sentiment_score = []
        positive_rssfeeds = 0
        neutral_rssfeeds = 0
        negative_rssfeeds = 0
        print("size:", self.processed_data.shape)

        try:
            for index,row in rssfeeds.iterrows():
                blob = TextBlob(row['summary'].encode().decode('ascii', errors="replace"))
                sentiment_polarity = blob.sentiment.polarity
                # print("index: ", index, "sentiment polarity: ", sentiment_polarity)
                if sentiment_polarity < 0:
                    negative_rssfeeds += 1
                    sentiment = Sentiments.NEGATIVE
                elif sentiment_polarity <= 0.2:
                    sentiment = Sentiments.NEUTRAL
                    neutral_rssfeeds += 1
                else:
                    sentiment = Sentiments.POSITIVE
                    positive_rssfeeds += 1
                rssfeed_sentiments.append(sentiment)
                sentiment_score.append(sentiment_polarity)
                # print("calculating sentiment for row ",index," sentiment= ",sentiment)
        except Exception as e:
            print(row["summary"])
        print("rssfeed sentiments list length: ", len(rssfeed_sentiments))
        print("rssfeed sentiments column length: ", rssfeeds.shape)
        # print(rssfeed_sentiments)
        rssfeeds['sentiment'] = rssfeed_sentiments
        rssfeeds['sentiment_score'] = sentiment_score

        total_rssfeeds_analysed = positive_rssfeeds + neutral_rssfeeds + negative_rssfeeds
        positive_rssfeeds_percentage = positive_rssfeeds / total_rssfeeds_analysed * 100
        neutral_rssfeeds_percentage = neutral_rssfeeds / total_rssfeeds_analysed * 100

        print("\nNo. of positive tweets = {} Percentage = {}".format(positive_rssfeeds, positive_rssfeeds_percentage))
        print("\nNo. of neutral tweets  = {} Percentage = {}".format(neutral_rssfeeds, neutral_rssfeeds_percentage))
        print("\nNo. of negative tweets = {} Percentage = {}".format(negative_rssfeeds, 100 - (positive_rssfeeds_percentage + neutral_rssfeeds_percentage)))

        self.processed_data = rssfeeds
        return self.processed_data