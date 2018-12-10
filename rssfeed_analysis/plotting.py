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
        Plot to visualize reddit posts based users posting frequncy
        """

        # print("wordlist: \n", self.wordlist)
        # print("columns in self.processed_data: ", self.processed_data.columns)
        redditposts_per_user = self.processed_data.groupby(["author"]).count()
        print("shape of data: ", redditposts_per_user.shape)
        print("redditposts_per_user: "
              "\n***********************************\n",
              redditposts_per_user.head(),
              "\n********************************\n")

        redditposts_barchart_data=redditposts_per_user.groupby(['title']).count()

        print("redditposts_barchart_data: "
              "\n***********************************\n",
              redditposts_barchart_data.head(),
              "\n********************************\n")

        print("distribution of reddit posts: ", redditposts_barchart_data.index)
        no_of_posts = list(redditposts_barchart_data.index)
        frequency = list(redditposts_barchart_data["summary"])

        dist = [
            graph_objs.Bar(
                x=no_of_posts,
                y=frequency
            )]
        plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="Tweets per User")},
                            filename='plots/redditposts.html')

        """
        Plot to visualize reddit posts per day
        """
        df_ = self.processed_data
        dates = []

        def try_parsing_date(text):
            for fmt in ('%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M', '%m/%d/%Y %I:%M %p', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M:%S'):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    pass
                print("error: ", text)
            raise ValueError('no valid date format found')


        for index, row in df_.iterrows():
            try:
                dates.append(parse(row["timestamp_ms"], fuzzy=True))
            except Exception as e:
                df_.drop(df_.index[index], inplace=True)

        df_["modified_date"] = dates
        print("First 5 rows: \n", df_.head())

        redditposts_day = pd.DataFrame()
        # Create a column from the datetime variable
        redditposts_day['datetime'] = df_["modified_date"]
        redditposts_day['sentiment'] = df_["sentiment"]
        # Convert that column into a datetime datatype
        redditposts_day['datetime'] = pd.to_datetime(redditposts_day['datetime'])
        # Set the datetime column as the index
        redditposts_day.index = redditposts_day['datetime']

        print("redditposts_per day data: \n***********************************\n", redditposts_day.head(),
              "\n********************************\n")
        print("redditposts_per day data resampled data: \n***********************************\n",
              redditposts_day.resample('1D').count()["datetime"],
              "\n********************************\n")

        reddit_posts = [
            graph_objs.Bar(
                x=redditposts_day.resample('1D').mean().index,
                y=redditposts_day.resample('1D').count()["datetime"],
            )]

        plotly.offline.plot({"data": reddit_posts, "layout": graph_objs.Layout(title="Reddit posts per Day")},
                            filename='plots/redditposts_perday.html')


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

        # print("shape of data: ", self.processed_data.shape)
        # print("wordlist: \n", self.wordlist)

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

        plot_data = []

        total_sentiment_df = pd.DataFrame()
        total_sentiment_df['datetime'] = df_["modified_date"]
        total_sentiment_df['datetime'] = pd.to_datetime(total_sentiment_df['datetime'])
        total_sentiment_df['score'] = df_["sentiment_score"]
        total_sentiment_df.index = total_sentiment_df['datetime']

        plot_data.append(Scatter(
            x=total_sentiment_df.resample('60Min').mean().index,
            y=total_sentiment_df.resample('60Min').mean()["score"],
            name="Overall Sentiment",
            mode="lines"
        ))

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
            print(scores_df.head())
            plot_data.append(Scatter(
                x=scores_df.resample('15Min').mean().index,
                y=scores_df.resample('15Min').mean()["score"],
                name=sentiment,
                mode="lines"
            ))


        plotly.offline.plot({"data": plot_data,
                             "layout": graph_objs.Layout(
                                 title="sentiment over time",
                                 xaxis=dict(type="date", range=['2018-03-13', max(scores_df["datetime"])]))
                                 # yaxis=dict(type='log', autorange=True))
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
                                 xaxis=dict(range=['2018-03-13', max(prices_data["datetime"])])
                             )
                             }, filename="plots/prices_data.html")

        """
        Plot to visualize number of reddit_posts vs price fluctuation
        """

        reddit_posts = Scatter(
            x=scores_df.resample('60Min').mean().index,
            y=scores_df.resample('60Min').count()["datetime"],
            mode='lines',
            name="No of reddit_posts"
        )

        prices = Scatter(
            x=prices_data["datetime"],
            y=prices_data["close"],
            name="Bitcoin Price",
            mode="lines"
        )
        data = [reddit_posts, prices]
        # plotly.offline.plot({"data": data,
        #                      "layout": graph_objs.Layout(
        #                          title="volume of reddit_posts vs price fluctuation",
        #                          xaxis=dict(range=['2018-03-10', '2018-03-30'])
        #                      )
        #                      }, filename="plots/sentiment_by_5_mins.html")

        plotly.offline.plot({"data": data,
                             "layout": graph_objs.Layout(dict(
                            title='volume of reddit_posts vs price fluctuation',
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
        Plot to visualize number of rssfeeds ever hour
        """

        reddit_posts = [Scatter(
            x=scores_df.resample('60Min').mean().index,
            y=scores_df.resample('60Min').count()["datetime"],
            mode='lines+markers',
            name="No of reddit_posts"
        )]

        plotly.offline.plot({"data": reddit_posts,
                             "layout": graph_objs.Layout(dict(
                                 title='volume of reddit_posts',
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
                             )}, filename="plots/number_of_reddit_posts.html")





