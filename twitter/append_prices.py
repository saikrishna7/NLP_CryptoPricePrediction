import pandas as pd
from dateutil.parser import parse
import datetime
from itertools import islice
import os


class AppendPrices:

    def append_prices_to_tweets(self, file, is_testing_set):

        if os.path.isfile("data/one_month_clean_train_data.csv") and os.path.exists(
                "data/one_month_clean_test_data.csv"):
            if is_testing_set:
                tweets = pd.read_csv("data/one_month_clean_test_data.csv")
            else:
                tweets = pd.read_csv("data/one_month_clean_data.csv")
        else:
            print("Appending prices to tweets: ", file)
            df = pd.read_csv(file)
            print(df.head(1))
            if "level_0" in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
            if "index" in df.columns:
                df.drop(['index'], axis=1, inplace=True)
            df.reset_index(inplace=True, drop=True)
            print("dataframe shape: ", df.shape)

            dates = []
            for index, row in df.iterrows():
                try:
                    dates.append(parse(row["timestamp"], fuzzy=True))
                except Exception as e:
                    df.drop(df.index[index], inplace=True)

            df["modified_date"] = dates
            print("Data head: ", df.head())

            df["datetime"] = pd.to_datetime(df["modified_date"])
            df.sort_values(by=["datetime"], inplace=True)

            # print("Data after chaging date to pandas datetime: ", df.head())
            # tweets = df[df.datetime>datetime.date(2018,3,14)]
            # tweets = df[df.datetime<datetime.date(2018,4,13)]
            tweets = df
            # print("tweets dataframe head: ", tweets.head())
            tweets.reset_index(inplace=True, drop=True)
            if not is_testing_set:
                tweets = tweets[6397:]
                tweets = tweets[tweets.datetime < datetime.date(2018, 4, 17)]
            else:
                tweets = tweets[tweets.datetime >= datetime.date(2018, 4, 17)]
            print("dataframe shape after removing unwanted dates: ", tweets.shape)
            print("min date: ", min(tweets.datetime))
            print("max date: ", max(tweets.datetime))
            if is_testing_set:
                tweets.to_csv("data/one_month_clean_test_data.csv", index=False)
            else:
                tweets.to_csv("data/one_month_clean_data.csv", index=False)

            # 2018-04-05 23:55:00 pm - 1522972799
            # 2018-03-13 23:59:59 pm - 1520985599
            test_data_prices = pd.read_csv("data/BTC.csv")
            if is_testing_set:
                test_data_prices = pd.read_csv("data/BTC.csv")
                test_data_prices.drop(columns=['Unnamed: 0'], inplace=True)
                test_data_prices["datetime"] = pd.to_datetime(test_data_prices['date'], unit='s')
                test_data_prices = test_data_prices[test_data_prices.datetime >= datetime.date(2018, 4, 17)]
                test_data_prices.to_csv("data/test_data_prices.csv", index=False)
                test_data_prices.reset_index(inplace=True, drop=True)

            # april 11, 12 am: 1523404800
            # March 25, 2018 12:00:00 AM : 1521936000
            # March 14, 2018 12:00:00 AM : 1520985600

            prices = pd.read_csv("data/BTC.csv")
            # prices = prices[prices.date < 1523404800]
            prices = prices[prices.date >= 1520985600]
            prices.drop(columns=['Unnamed: 0'], inplace=True)
            prices["datetime"] = pd.to_datetime(prices['date'], unit='s')
            prices.to_csv("data/train_data_prices.csv", index=False)

        print("Max date of train/test data for adding prices: ", max(tweets['datetime']))
        prices = prices[prices.datetime <= max(tweets['datetime'])]
        # print("tweets head \n #######################", tweets.head())
        tweets.datetime = pd.to_datetime(tweets.datetime)
        if "level_0" in tweets.columns:
            tweets.drop(['level_0'], axis=1, inplace=True)
        if "index" in tweets.columns:
            tweets.drop(['index'], axis=1, inplace=True)
        tweets.reset_index(inplace=True, drop=True)
        print("tweets head after resetting indexes: \n", tweets.head())
        # print(prices.columns)
        # print(tweets.columns)
        new_index = 0
        if is_testing_set:
            prices = test_data_prices
        go_to_next_price_point = False
        for i, price_row in prices.iterrows():
            # print(price_row)
            go_to_next_price_point = False
            for j, tweet in islice(tweets.iterrows(), new_index, None):
                # print(tweet["datetime"])
                # print(price_row["datetime"])
                if i == 0:
                    if (tweet["datetime"] <= price_row["datetime"]):
                        print("j: ", j)
                        print("First price point row")
                        print(tweet["datetime"])
                        print(price_row["datetime"])
                        tweets.loc[j, 'close'] = price_row["close"]
                        tweets.loc[j, 'high'] = price_row["high"]
                        tweets.loc[j, 'low'] = price_row["low"]
                        tweets.loc[j, 'open'] = price_row["open"]
                        tweets.loc[j, 'quoteVolume'] = price_row["quoteVolume"]
                        tweets.loc[j, 'volume'] = price_row["volume"]
                        tweets.loc[j, 'weightedAverage'] = price_row["weightedAverage"]
                else:
                    print("tweet['datetime']: ", tweet["datetime"])
                    print("price_row['datetime']: ", price_row["datetime"])
                    # print("tweets.head(): \n\n", tweets.head())
                    # print("\n\n prices.head(): \n\n", prices.head())
                    if (tweet["datetime"] <= price_row["datetime"] and tweet["datetime"] >= prices.loc[i-1, 'datetime']):
                        tweets.loc[j, 'close'] = price_row["close"]
                        tweets.loc[j, 'high'] = price_row["high"]
                        tweets.loc[j, 'low'] = price_row["low"]
                        tweets.loc[j, 'open'] = price_row["open"]
                        tweets.loc[j, 'quoteVolume'] = price_row["quoteVolume"]
                        tweets.loc[j, 'volume'] = price_row["volume"]
                        tweets.loc[j, 'weightedAverage'] = price_row["weightedAverage"]
                        print("$$$$$$$$$$$$$$$$$$$  adding price to row  $$$$$$$$$$$$$$$$$")
                        print("j: ", j)
                        print(tweet["datetime"])
                        print(price_row["datetime"])

                    else:
                        print("######################### Breaking #########################")
                        print("j: ", j)
                        print(tweet["datetime"])
                        print(price_row["datetime"])
                        print("prices previous row: ", prices.loc[i - 1, 'datetime'])
                        print("The new index for inner loop iteration will be: ", j)
                        new_index = j
                        go_to_next_price_point = True
                        break
                if go_to_next_price_point: break
        if "index" in tweets.columns:
            tweets.drop(['index'], axis=1, inplace=True)
        if not is_testing_set:
            tweets.to_csv("data/one_month_clean_data_with_prices.csv", index=False)
        else:
            tweets.to_csv("data/one_month_clean_test_data_with_prices.csv", index=False)
