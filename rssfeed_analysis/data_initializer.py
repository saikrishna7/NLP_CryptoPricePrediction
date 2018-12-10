from collections import Counter
import nltk
import pandas as pd
import os
from get_all_s3_files import GetTrainData
from get_last_hour_s3_files import GetTestData

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

    def initialize(self, csv_file, is_testing_set=False, col_names=None, from_cached_bow=None, from_cached_word2vec=None, duration=None):
        if from_cached_bow is not None:
            self.data_model = pd.read_csv(from_cached_bow, encoding='utf-8')
            self.word2vec_data = pd.read_csv(from_cached_word2vec, encoding='utf-8')
            return
        self.is_testing = is_testing_set

        # if is_testing_set:
        #     if not os.path.exists(csv_file):
        #         traindata = GetTrainData()
        #         traindata.main()
        # else:
        #     if not os.path.isfile(csv_file):
        #         testdata = GetTestData()
        #         testdata.main()


        if not is_testing_set:
            if col_names is not None:
                print("train set")
                print("colnames: ", col_names)
                self.data = pd.read_csv(csv_file, header=0, names=col_names)
            else:
                if not os.path.exists(csv_file):
                    traindata = GetTrainData()
                    traindata.main()
                print("train set")
                self.data = pd.read_csv(csv_file, header=0, names=["author", "title", "timestamp_ms", "summary"])
        else:
            print("reading test data")
            if duration:
                testdata = GetTestData()
                testdata.main(duration)
            elif not os.path.exists(csv_file):
                testdata = GetTestData()
                testdata.main(duration=24)
            self.data = pd.read_csv(csv_file, header=0, names=["author", "title", "timestamp_ms", "summary"])
            not_null_text = 1 ^ pd.isnull(self.data["summary"])
            not_null_id = 1 ^ pd.isnull(self.data["author"])
            self.data = self.data.loc[not_null_id & not_null_text, :]

        self.processed_data = self.data
        print("shape before dropping NaN rows: ", self.processed_data.shape)
        self.processed_data.dropna(axis=0, how='any', inplace=True)
        print("shape after dropping NaN rows: ", self.processed_data.shape)
        self.processed_data['summary'] = self.processed_data['summary'].astype(str)
        print("shape: ", self.processed_data.shape)
        # print(self.processed_data.head())
        self.wordlist = []
        self.data_model = None
        self.data_labels = None

