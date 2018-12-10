from plotly.graph_objs import *
from plotly import graph_objs
import plotly
import pandas as pd
import datetime
from dateutil.parser import parse

class Plotting:

    def __init__(self, previous):
        self.processed_data = previous.processed_data
        self.data_model = previous.data_model
        self.wordlist = previous.wordlist


    def plot(self):
        scores_df = pd.DataFrame()

        """
        Plot to visualize tweets based on users posting frequency
        """
        print("&&&&&&&&&&&&&&&&&&&&&&&\nprocessed data tail \n\n\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(self.processed_data.tail())
        tweets_per_user = self.processed_data.groupby(["id"]).count()
        print("shape of data: ", tweets_per_user.shape)
        print("tweets_per_user: \n***********************************\n", tweets_per_user.head(),
              "\n********************************\n")
        tweets_barchart_data = tweets_per_user.groupby(['timestamp']).count()
        print("tweets_barchart_data: \n***********************************\n", tweets_barchart_data.head(),
              "\n********************************\n")
        print("distribution of tweets: ", tweets_barchart_data.index)
        no_of_posts = list(tweets_barchart_data.index)
        frequency = list(tweets_barchart_data["text"])

        dist = [
            graph_objs.Bar(
                x=no_of_posts,
                y=frequency
            )]
        plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="Tweets per User")},
                            filename='plots/tweets.html')


        """
        High frequency plot
        """

        tweets_barchart_high_frequency_data = tweets_barchart_data[tweets_barchart_data.index<20]
        print("tweets_high_frequency_barchart_data: \n***********************************\n",
              tweets_barchart_high_frequency_data,
              "\n********************************\n")
        no_of_posts = list(tweets_barchart_data.index)
        frequency = list(tweets_barchart_data["text"])

        dist = [
            graph_objs.Bar(
                x=no_of_posts,
                y=frequency
            )]
        plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="High Frequency Tweets Plot")},
                            filename='plots/tweets_high_frequency.html')

        """
        Plot distribution of tweet lengths
        """
        print("self.processed_data column types: ", self.processed_data.dtypes)
        words_per_tweet = []
        # print(self.processed_data.columns)
        # print(self.processed_data.head())
        self.processed_data.text = self.processed_data.text.astype(str)
        for index, row in self.processed_data.iterrows():
            words_per_tweet.append(len(row["text"].split()))
        
        dist = [graph_objs.Histogram(x=words_per_tweet)]

        plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="Words distribution in Tweets")},
                            filename='plots/Words_distribution_in_Tweets.html')



        """
        Plot to visualize tweets per day
        """
        df_ = self.processed_data
        dates = []

        # def try_parsing_date(text):
        #     for fmt in (
        #     '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M', '%m/%d/%Y %I:%M %p', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M:%S'):
        #         try:
        #             return datetime.strptime(text, fmt)
        #         except ValueError:
        #             pass
        #         print("error: ", text)
        #     raise ValueError('no valid date format found')
        print(df_.head())
        print(df_.shape)
        for index, row in df_.iterrows():
            try:
                dates.append(parse(row["timestamp"], fuzzy=True))
            except Exception as e:
                df_.drop(df_.index[index], inplace=True)

        df_["modified_date"] = dates
        print("First 5 rows: \n", df_.head())

        tweets_day = pd.DataFrame()
        # Create a column from the datetime variable
        tweets_day['datetime'] = df_["modified_date"]
        tweets_day['sentiment'] = df_["sentiment"]
        # Convert that column into a datetime datatype
        tweets_day['datetime'] = pd.to_datetime(tweets_day['datetime'])
        # Set the datetime column as the index
        tweets_day.index = tweets_day['datetime']

        print("tweets per day data: \n***********************************\n", tweets_day.head(),
              "\n********************************\n")
        # print("tweets per day resampled data: \n***********************************\n",
        #       tweets_day.resample('1D').count()["datetime"],
        #       "\n********************************\n")
        print("shape of data before removing unwanted dates: ", tweets_day.shape)
        tweets_day = tweets_day[tweets_day["datetime"]>datetime.date(2018,3,13)]
        print("shape of data after removing unwanted dates: ", tweets_day.shape)
        tweet_posts = [
            graph_objs.Bar(
                x=tweets_day.resample('1D').mean().index,
                y=tweets_day.resample('1D').count()["datetime"],
            )]

        plotly.offline.plot({"data": tweet_posts, "layout": graph_objs.Layout(title="tweets per Day")},
                            filename='plots/tweets_perday.html')


        """
        Plot top words in the word list
        """

        words = pd.read_csv("data/wordlist.csv")
        x_words = list(words.loc[0:10, "word"])
        x_words.reverse()
        y_occ = list(words.loc[0:10, "occurrences"])
        y_occ.reverse()

        dist = [
            graph_objs.Bar(
                x=y_occ,
                y=x_words,
                orientation="h"
            )]
        plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="Top words in built wordlist")},
                            filename='plots/top_words.html')

        """
        Plot to visualize the top words in each sentiment
        """

        print("shape of data: ", self.processed_data.shape)

        grouped = self.data_model.groupby(["label"]).sum()
        words_to_visualize = []
        sentiments = [1, -1, 0]
        # get the most 7 common words for every sentiment
        for sentiment in sentiments:
            words = grouped.loc[sentiment, :]
            words.sort_values(inplace=True, ascending=False)
            for w in words.index[:7]:
                if w not in words_to_visualize:
                    words_to_visualize.append(w)

        # visualize it
        plot_data = []
        for sentiment in sentiments:
            plot_data.append(graph_objs.Bar(
                x=[w.split("_")[0] for w in words_to_visualize],
                y=[grouped.loc[sentiment, w] for w in words_to_visualize],
                name=sentiment
            ))

        plotly.offline.plot({"data": plot_data, "layout": graph_objs.Layout(title="Most common words across sentiments")},
                            filename="plots/emotions_wordcount.html")

        """
        Plot to visualize sentiment over time
        """

        # df_ = pd.read_csv("data/clean_outfile_with_sentiments.csv")
        # df_ = self.processed_data
        # dates = []
        plot_data = []

        for sentiment in sentiments:
            scores_df = pd.DataFrame()
            # Create a column from the datetime variable
            scores_df['datetime'] = df_.loc[df_["sentiment"] == sentiment, "modified_date"]
            # Convert that column into a datetime datatype
            scores_df['datetime'] = pd.to_datetime(scores_df['datetime'])
            # Create a column from the numeric score variable
            scores_df['score'] = df_.loc[df_["sentiment"] == sentiment, "sentiment_score"]
            # Set the datetime column as the index
            scores_df.index = scores_df['datetime']

            print("shape of data before removing unwanted dates: ", tweets_day.shape)
            scores_df = scores_df[scores_df["datetime"] > datetime.date(2018,3,13)]
            print("shape of data after removing unwanted dates: ", tweets_day.shape)

            # print("Check the data before plotting sentiment over time: ", scores_df.head())
            # print("range of data: ",min(scores_df["datetime"]), max(scores_df["datetime"]))
            plot_data.append(Scatter(
                x=scores_df.resample('5Min').mean().index,
                y=scores_df.resample('5Min').mean()["score"],
                name=sentiment
            ))

        plotly.offline.plot({"data": plot_data,
                             "layout": graph_objs.Layout(
                                 title="sentiment over time",
                                 xaxis=dict(range=[min(scores_df["datetime"]), max(scores_df["datetime"])])
                             )
                             }, filename="plots/sentiment_by_5_mins.html")

        """
        Plot historical prices data
        """

        prices_data = pd.read_csv("data/BTC.csv")
        prices_data['datetime'] = pd.to_datetime(prices_data['date'], unit='s')
        data = [Scatter(
            x=prices_data["datetime"],
            y=prices_data["close"],
            name=sentiment
        )]

        plotly.offline.plot({"data": data,
                             "layout": graph_objs.Layout(
                                 title="prices over time",
                                 xaxis=dict(range=[min(scores_df["datetime"]), max(prices_data["datetime"])])
                             )
                             }, filename="plots/prices_data.html")

        """
        Plot to visualize number of tweets vs price fluctuation
        """
        # print("data to visualize number of tweets vs price fluctuation: ", scores_df.head())
        # print("data throwing error: ", scores_df.resample('60Min').mean().index)
        # print("data throwing error: ", scores_df.resample('60Min').count()["datetime"])
        tweets = Scatter(
            x=scores_df.resample('60Min').mean().index,
            y=scores_df.resample('60Min').count()["datetime"],
            mode='lines',
            name="No of tweets"
        )

        prices = Scatter(
            x=prices_data["datetime"],
            y=prices_data["close"],
            name="Bitcoin Price",
            mode="lines"
        )
        data = [tweets, prices]

        plotly.offline.plot({"data": data,
                             "layout": graph_objs.Layout(dict(
                            title='volume of tweets vs price fluctuation',
                            xaxis=dict(
                                     rangeselector=dict(
                                         buttons=list([
                                             dict(count=1,
                                                  label='1w',
                                                  step='week',
                                                  stepmode='backward'),
                                             dict(count=1,
                                                  label='1m',
                                                  step='month',
                                                  stepmode='backward'),
                                             dict(count=3,
                                                  label='3m',
                                                  step='month',
                                                  stepmode='backward'),
                                             dict(count=1,
                                                  label='YTD',
                                                  step='year',
                                                  stepmode='todate'),
                                             dict(count=1,
                                                  label='1y',
                                                  step='year',
                                                  stepmode='backward'),
                                             dict(step='all')
                                                        ])
                                                        ),
                                     rangeslider=dict(),
                                     type='date'
                                    )
                            )
                        )}, filename="plots/sentiment_by_hour.html")


        """
        Plot to visualize number of tweets every hour
        """

        plotly.offline.plot({"data": [tweets],
                             "layout": graph_objs.Layout(dict(
                                 title='volume of tweets',
                                 xaxis=dict(
                                     rangeselector=dict(
                                         buttons=list([
                                             dict(count=1,
                                                  label='1w',
                                                  step='week',
                                                  stepmode='backward'),
                                             dict(count=1,
                                                  label='1m',
                                                  step='month',
                                                  stepmode='backward'),
                                             dict(count=3,
                                                  label='3m',
                                                  step='month',
                                                  stepmode='backward'),
                                             dict(count=1,
                                                  label='YTD',
                                                  step='year',
                                                  stepmode='todate'),
                                             dict(count=1,
                                                  label='1y',
                                                  step='year',
                                                  stepmode='backward'),
                                             dict(step='all')
                                         ])
                                     ),
                                     rangeslider=dict(),
                                     type='date'
                                 )
                             )
                             )}, filename="plots/number_of_tweets_everyhour.html")

        """
        Plot to visualize prices for train and test data
        """

        # test_data = pd.read_csv("data/one_month_clean_test_data_with_prices.csv")
        # min_date = min(test_data["timestamp"])
        # data.wordlist = pd.read_csv("data/wordlist.csv")
        #
        # prices_data = pd.read_csv("data/BTC.csv")
        #
        #
        #
        # prices_data['datetime'] = pd.to_datetime(prices_data['date'], unit='s')
        # data1 = [Scatter(
        #     x=prices_data["datetime"],
        #     y=prices_data["close"],
        #     name=sentiment
        # )]
        #
        # plot_data.append(Scatter(
        #     x=scores_df.resample('5Min').mean().index,
        #     y=scores_df.resample('5Min').mean()["score"],
        #     name=sentiment
        # ))
        #
        # prices = Scatter(
        #     x=prices_data["datetime"],
        #     y=prices_data["close"],
        #     name="Bitcoin Price",
        #     mode="lines"
        # )
        # data = [tweets, prices]
        #
        # plotly.offline.plot({"data": data,
        #                      "layout": graph_objs.Layout(dict(
        #                          title='volume of tweets vs price fluctuation',
        #                          xaxis=dict(
        #                              rangeselector=dict(
        #                                  buttons=list([
        #                                      dict(count=1,
        #                                           label='1w',
        #                                           step='week',
        #                                           stepmode='backward'),
        #                                      dict(count=1,
        #                                           label='1m',
        #                                           step='month',
        #                                           stepmode='backward'),
        #                                      dict(count=3,
        #                                           label='3m',
        #                                           step='month',
        #                                           stepmode='backward'),
        #                                      dict(count=1,
        #                                           label='YTD',
        #                                           step='year',
        #                                           stepmode='todate'),
        #                                      dict(count=1,
        #                                           label='1y',
        #                                           step='year',
        #                                           stepmode='backward'),
        #                                      dict(step='all')
        #                                  ])
        #                              ),
        #                              rangeslider=dict(),
        #                              type='date'
        #                          )
        #                      )
        #                      )}, filename="plots/sentiment_by_hour.html")
