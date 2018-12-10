from collections import Counter
import nltk
import pandas as pd
import os
from get_all_s3_files import GetTrainData
from get_last_hour_s3_files import GetTestData
from remove_noise_tweets import RemoveNoiseTweets
from append_prices import AppendPrices
"""
Data preprocessing

Loading the data

The code snippet below will load the data form the given file for further processing or just read the already preprocessed file from the cache. 
There's also a distinction between processing testing and training data. As the test.csv file was full of empty entries, they were removed. 
Additional class properties such as data_model, wordlist etc. will be used later.

"""


class DataInitializer():
    data = []
    processed_data = []
    wordlist = []

    data_model = None
    data_labels = None
    is_testing = False

    def initialize(self, csv_file, is_testing_set=False, col_names=None, cache_bow_output=None,
                   cache_word2vec_output=None, duration=None, plotting=None):
        print("from_cached_bow: ", cache_bow_output)
        print("from_cached_word2vec: ", cache_word2vec_output)
        print("Data is for plotting: ", plotting)
        if plotting:
            self.processed_data = pd.read_csv(csv_file, header=0, names=col_names)
            return

        if os.path.isfile(cache_bow_output) and os.path.isfile(cache_word2vec_output):
            self.data_model = pd.read_csv(cache_bow_output, encoding='utf-8')
            self.word2vec_data = pd.read_csv(cache_word2vec_output, encoding='utf-8')
            return
        self.is_testing = is_testing_set

        if not is_testing_set:
            if col_names is not None:
                print("train set")
                print("colnames: ", col_names)
                self.data = pd.read_csv(csv_file, header=0, names=col_names)
            else:
                if not os.path.exists("data/one_month_clean_data_with_prices.csv"):
                    if not os.path.exists("data/one_month_clean_train_data.csv"):
                        if not os.path.exists("data/clean_train.csv"):
                            print("Collecting tweets from S3")
                            traindata = GetTrainData()
                            traindata.main()
                            print("Cleaning tweets, removing duplicates and noise")
                            data = RemoveNoiseTweets("data/clean_train.csv", is_testing_set)
                            data.removenoise()
                            data = AppendPrices()
                            data.append_prices_to_tweets("data/clean_train.csv", is_testing_set)
                            print("train set")
                            self.data = pd.read_csv(csv_file, header=0)
                        else:
                            print("clean_train data available, removing duplicates and noise")
                            data = RemoveNoiseTweets("data/clean_train.csv", is_testing_set)
                            data.removenoise()
                            data = AppendPrices()
                            data.append_prices_to_tweets("data/clean_train.csv", is_testing_set)
                            print("train set")
                            self.data = pd.read_csv(csv_file, header=0)
                            # names=["index", "timestamp", "followers_count",
                            # "favourites_count", "id", "screen_name", "text",
                            # "modified_date", "datetime", "price"]
                    else:
                        print("Entering appending prices class")
                        data = AppendPrices()
                        data.append_prices_to_tweets("data/clean_train.csv", is_testing_set)
                        print("train set")
                        self.data = pd.read_csv(csv_file, header=0)
                        # names=["index", "timestamp", "followers_count",
                        # "favourites_count", "id", "screen_name", "text",
                        # "modified_date", "datetime", "price"]
                else:
                    print("train set")
                    self.data = pd.read_csv(csv_file, header=0)
                    # names=["index", "timestamp", "followers_count",
                    # "favourites_count", "id", "screen_name", "text",
                    # "modified_date", "datetime", "price"]


        else:
            print("reading test data")
            if duration:
                print("test dataset: ", csv_file)
                testdata = GetTestData()
                testdata.main(duration)
            if not os.path.exists(csv_file):
                if not os.path.exists("data/one_month_clean_test_data.csv"):
                    if not os.path.exists("data/clean_test.csv"):
                        print("Collecting tweets from S3")
                        testdata = GetTestData()
                        testdata.main()
                        data = RemoveNoiseTweets("data/clean_test.csv", is_testing_set)
                        data.removenoise()
                        data = AppendPrices()
                        data.append_prices_to_tweets("data/clean_test.csv", is_testing_set)
                        print("test set")
                        self.data = pd.read_csv(csv_file, header=0)
                    else:
                        print("clean data available, removing duplicates and noise")
                        data = RemoveNoiseTweets("data/clean_test.csv", is_testing_set)
                        data.removenoise()
                        data = AppendPrices()
                        data.append_prices_to_tweets("data/clean_test.csv", is_testing_set)
                        print("test set")
                        self.data = pd.read_csv(csv_file, header=0)
                        # names=["index", "timestamp", "followers_count",
                        # "favourites_count", "id", "screen_name", "text",
                        # "modified_date", "datetime", "price"]
                else:
                    print("Entering appending prices class")
                    data = AppendPrices()
                    data.append_prices_to_tweets("data/clean_test.csv", is_testing_set)
                    print("test set")
                    self.data = pd.read_csv(csv_file, header=0)
                    # names=["index", "timestamp", "followers_count",
                    # "favourites_count", "id", "screen_name", "text",
                    # "modified_date", "datetime", "price"]
            else:
                print("test set")
                self.data = pd.read_csv(csv_file, header=0)
                # names=["index", "timestamp", "followers_count",
                # "favourites_count", "id", "screen_name", "text",
                # "modified_date", "datetime", "price"]
        print(csv_file)
        print("Error: ", self.data[0:5])
        # print("Error columns: ", self.data.columns)
        not_null_text = 1 ^ pd.isnull(self.data["text"])
        not_null_id = 1 ^ pd.isnull(self.data["id"])
        self.data = self.data.loc[not_null_id & not_null_text, :]

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
        # print(self.data.head())
        print("Data columns are: \n", self.data.columns, "\n")
        if "index" in self.data.columns:
            self.data.drop(['index'], axis=1, inplace=True)
        if "level_0" in self.data.columns:
            self.data.drop(['level_0'],axis=1,inplace=True)
        self.processed_data = self.data
        print("shape before dropping NaN rows: ", self.processed_data.shape)
        self.processed_data.dropna(axis=0, how='any', inplace=True)
        print("shape after dropping NaN rows: ", self.processed_data.shape)
        self.processed_data['text'] = self.processed_data['text'].astype(str)
        self.wordlist = []
        self.data_model = None
        self.data_labels = None
