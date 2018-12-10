from data_initializer import DataInitializer
from data_cleaning import DataCleaning
from data_tokenize import DataTokenize
from data_cleaner import DataCleaner
from collections import Counter
from word_list import WordList
from bagofwords import BagOfWords
from sentiment_analysis import Sentiments
from classification import Classification
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from plotting import Plotting
from twitterdata import TwitterData
from word2vecprovider import Word2VecProvider
from get_prices_data import GetPricesData
from remove_noise_tweets import RemoveNoiseTweets
from append_prices import AppendPrices
import os
from collections import Counter
from datetime import datetime
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, SimpleRNN

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
import keras.backend as K
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from plotly.graph_objs import *
from plotly import graph_objs
import plotly
import pickle
from collections import Counter
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


seed = 1000
use_test_data = True

min_occurrences=5
test_data_word2vec_file_name = "data/processed_test_word2vec_" + str(min_occurrences) + ".csv"
train_data_word2vec_file_name = "data/processed_train_word2vec_" + str(min_occurrences) + ".csv"

test_data_bow_file_name = "data/processed_test_bow_" + str(min_occurrences) + ".csv"
train_data_bow_file_name = "data/processed_train_bow_" + str(min_occurrences) + ".csv"


def preprocess(data_path, is_testing, min_occurrences=5, cache_bow_output=None, cache_word2vec_output=None, duration=None, sentiment_method=None):
    if duration and cache_bow_output and cache_word2vec_output:
        data = DataInitializer()
        data.initialize(data_path, is_testing, duration=duration)
    elif cache_bow_output and cache_word2vec_output:
        data = DataInitializer()
        data.initialize(data_path, is_testing, cache_bow_output=cache_bow_output, cache_word2vec_output=cache_word2vec_output)
    else:
        data = DataInitializer()
        data.initialize(data_path, is_testing)

    if not os.path.isfile("data/Train_BTC.csv"):
        prices_data = GetPricesData()
        prices_data.main()

    if not os.path.isfile("data/Test_BTC.csv"):
        prices_data = GetPricesData()
        prices_data.main()

    data = DataCleaning(data, is_testing)
    data.cleanup(DataCleaner(is_testing))

    if is_testing:
        print("Testing data shape:", data.processed_data.shape)
    else:
        print("Training data shape:", data.processed_data.shape)

    data = Sentiments(data, sentiment_method=sentiment_method)
    data.sentiment_analysis_by_text()


    print("First five rows with sentiment: ", data.processed_data.head())
    if is_testing:
        data.processed_data.to_csv("data/one_month_clean_test_data_with_prices.csv", sep=',', encoding='utf-8', index=False)
        # os.remove(data_path)
    else:
        data.processed_data.to_csv("data/one_month_clean_data_with_prices.csv", sep=',', encoding='utf-8', index=False)
        # os.remove(data_path)



    if os.path.isfile(cache_word2vec_output):
        print("cache_word2vec_output file name: ", cache_word2vec_output)
        word2vec_data_model = pd.read_csv(cache_word2vec_output)
        data.data_model = pd.read_csv(cache_bow_output)
        print("data model head: ", data.data_model.head(5))
    else:
        data = DataTokenize(data)
        data.tokenize()
        data.stem()

        data = WordList(data)
        data.build_wordlist(min_occurrences=min_occurrences)

        word2vec_data = data
        data = BagOfWords(data.processed_data, data.wordlist, is_testing)
        data.build_data_model()
        print("data model head: ", data.data_model.head(5))

        """
        Word 2 vec
        """

        word2vec = Word2VecProvider()

        # REPLACE PATH TO THE FILE
        word2vec.load("data/glove.twitter.27B.200d-with2num.txt")
        word2vec_data = TwitterData(word2vec_data)
        word2vec_data.build_final_model(word2vec)
        word2vec_data_model = word2vec_data.data_model

        if "original_id" in word2vec_data_model.columns:
            word2vec_data_model.drop("original_id", axis=1, inplace=True)
        word2vec_data_model.dropna(axis=0, inplace=True)
        word2vec_data_model.reset_index(inplace=True, drop=True)
        word2vec_data_model.index = word2vec_data_model['timestamp']

    print("final word2vec data model: \n", word2vec_data_model.head(), "\n")

    # if not is_testing:
    #     data = Plotting(data)
    #     data.plot()

    if not is_testing:
        if not os.path.isfile("train_sequences"):
            print("\n##########################\n"
                  "Tokenizing the tweets\n"
                  "############################\n")
            texts = []
            sentiments = []
            tokenized_data = pd.DataFrame()

            for text in data.processed_data["text"]:
                texts.append(text)

            for sentiment in data.processed_data['sentiment']:
                sentiments.append(sentiment)

            print("texts: ", texts[0:5])
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')
            padded_sequences = pd.DataFrame(data=padded_sequences)

            merged_train_data = pd.concat([padded_sequences, data.processed_data[["high", "low", "open",
                                                                                  "quoteVolume", "volume",
                                                                                  "weightedAverage"]]], axis=1)
            train_targets = data.processed_data[["close"]]
            print("shape of merged train data: ", merged_train_data.shape)

            with open('data/train_sequences', 'wb') as fp:
                pickle.dump(merged_train_data, fp)
            with open('data/train_prices', 'wb') as fp:
                pickle.dump(train_targets, fp)


            # load the whole embedding into memory
            embeddings_index = dict()
            with open("data/glove.twitter.27B.200d-with2num.txt", "r", encoding="utf-8") as my_file:
                for line in my_file:
                    values = line.split()
                    word = values[0]
                    coefs = numpy.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            # f.close()
            print("*" * 80, "\n" * 10)
            print('Loaded %s train word vectors.' % len(embeddings_index))
            print('Total %s of word indexes.' % len(tokenizer.word_index))

            with open('data/embeddings_index', 'wb') as fp:
                pickle.dump(embeddings_index, fp)
            with open('data/train_word_indexes', 'wb') as fp:
                pickle.dump(tokenizer.word_index, fp)

            # encode class values as integers
            # encoder = LabelEncoder()
            # encoder.fit(sentiments)
            # encoded_sentiments = encoder.transform(sentiments)

            # convert integers to dummy variables (i.e. one hot encoded)
            # dummy_sentiments = np_utils.to_categorical(encoded_sentiments)

            # for text in data.processed_data.loc[data.processed_data['sentiment'] != 0, "text"]:
            #     texts.append(text)
            #
            # for sentiment in data.processed_data.loc[data.processed_data['sentiment'] != 0, "sentiment"]:
            #     sentiments.append(sentiment)

    else:
        if not os.path.isfile("test_sequences"):
            print("\n##########################\n"
                  "Tokenizing the tweets\n"
                  "############################\n")
            texts = []
            sentiments = []
            tokenized_data = pd.DataFrame()

            for text in data.processed_data["text"]:
                texts.append(text)

            for sentiment in data.processed_data['sentiment']:
                sentiments.append(sentiment)

            print("texts: ", texts[0:5])
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')
            padded_sequences = pd.DataFrame(data=padded_sequences)

            merged_test_data = pd.concat([padded_sequences, data.processed_data[["high", "low", "open",
                                                                                  "quoteVolume", "volume",
                                                                                  "weightedAverage"]]], axis=1)
            test_targets = data.processed_data[["close"]]
            print("shape of merged test data: ", merged_test_data.shape)

            with open('data/test_sequences', 'wb') as fp:
                pickle.dump(merged_test_data, fp)
            with open('data/test_prices', 'wb') as fp:
                pickle.dump(test_targets, fp)
            with open('data/test_word_indexes', 'wb') as fp:
                pickle.dump(tokenizer.word_index, fp)

            # padded_sequences = pd.DataFrame(data=padded_sequences)


    print("\n\n##################################################\npadded sequence head: \n", padded_sequences[0:5])
    print("\n####################################################\n padded sequence length \n",
          len(padded_sequences))


    if not os.path.isfile(train_data_word2vec_file_name) or not os.path.isfile(test_data_word2vec_file_name):
        if cache_bow_output is not None:
            data.data_model.to_csv(cache_bow_output, index=False, float_format="%.6f")
            word2vec_data_model.to_csv(cache_word2vec_output, index=False, float_format="%.6f")
    return data.data_model, word2vec_data_model


def preprare_data_for_processing(min_occurrences, use_cache_for_train, use_cache_for_test, duration, sentiment_method):
    training_data = None
    testing_data = None
    print("Loading data...")

    if duration is not None:
        if os.path.isfile(test_data_word2vec_file_name) and os.path.isfile(test_data_bow_file_name):
            os.remove(test_data_word2vec_file_name)
            os.remove(test_data_bow_file_name)
        testing_data, word2vec_testing_data = preprocess("data/one_month_clean_test_data_with_prices.csv", True, min_occurrences,
                                                         test_data_bow_file_name, test_data_word2vec_file_name, duration)
    if not os.path.isfile("data/BTC.csv"):
        prices_data = GetPricesData()
        prices_data.main()
    if use_cache_for_train:
        print("Reading the processed files")
        train_data_initializer_obj = DataInitializer()
        train_data_initializer_obj.initialize(None, cache_bow_output=train_data_bow_file_name,
                                              cache_word2vec_output=train_data_word2vec_file_name)
        training_data = train_data_initializer_obj.data_model
        word2vec_training_data = train_data_initializer_obj.word2vec_data
    else:
        print("Preprocessing data...")
        training_data, word2vec_training_data = preprocess("data/one_month_clean_data_with_prices.csv", False, min_occurrences, train_data_bow_file_name, train_data_word2vec_file_name, sentiment_method=sentiment_method)

    if use_cache_for_test:
        test_data_initializer_obj = DataInitializer()
        test_data_initializer_obj.initialize(None, cache_bow_output=test_data_bow_file_name,
                                             cache_word2vec_output=test_data_word2vec_file_name)
        word2vec_testing_data = test_data_initializer_obj.word2vec_data
        testing_data = test_data_initializer_obj.data_model
        print("Loaded from cached files...")
    else:
        testing_data, word2vec_testing_data = preprocess("data/one_month_clean_test_data_with_prices.csv", True, min_occurrences, test_data_bow_file_name, test_data_word2vec_file_name, sentiment_method=sentiment_method)

    print("Data preprocessed & cached...")
    return training_data, word2vec_training_data, testing_data, word2vec_testing_data


def log(text):
    print(text)
    with open("log.txt", "a") as log_file:
        log_file.write(str(text) + "\n")

lstm_size = 256

def lstm_cell(keep_prob=None):
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


def get_batches(x, y, batch_size=100):
    # print("shape of x", x.shape)
    # print("head of x", x.head())
    # print("length of x", len(x))
    n_batches = len(x) # batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    # print("return values: ", x, y)
    for index in range(0, len(x), batch_size):
        yield x[index:index + batch_size], y[index:index + batch_size]

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(128, batch_input_shape=(None, None, 1), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def train_and_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """

    scaler = StandardScaler().fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(df)
    # x = df.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)
    return df, scaler

    # return df / df.iloc[0] - 1

def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=10, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.

        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    # print("window length: ",window_len)
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        # print("idx: ", idx, "idx + window length: ", idx+window_len)
        # print("tmp: ", tmp)
        if zero_base:
            # print("Normalizing the data")
            tmp,scaler = normalise_zero_base(tmp)
            # print("tmp: ", tmp)
        window_data.append(tmp.values)
    return np.array(window_data), scaler


def prepare_data(train_data, test_data, target_col, window_len=10, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    # train_data, test_data = train_and_test_split(df, test_size=test_size)

    # extract window data
    X_train, temp1 = extract_window_data(train_data.iloc[:, 2:7], window_len, zero_base)
    X_test, temp2 = extract_window_data(test_data.iloc[:, 2:7], window_len, zero_base)
    # print("X_train: ", X_train)
    # print("shape of X_test: ", X_test.shape)
    # print("X_test data: ", X_test)
    # extract targets
    y_train = train_data["close"][window_len:].values
    y_test = test_data["close"][window_len:].values
    mean = np.mean(y_test)
    std = np.std(y_test)
    print("train_data['close'][:-window_len].values: ", train_data["close"][:-window_len].values)
    # if zero_base:
    #     y_train = y_train / train_data["close"][:-window_len].values - 1
    #     y_test = y_test / test_data["close"][:-window_len].values - 1
    y_train = y_train.reshape(-1, 1)
    # y_train, temp3 = normalise_zero_base(y_train)
    y_test = y_test.reshape(-1, 1)
    # y_test, scaler = normalise_zero_base(y_test)

    return X_train, X_test, y_train, y_test, mean, std

def line_plot(train_data, test_data):
    """
    Train and test sets
    """

    plot_data = []
    plot_data.append(Scatter(
        x=train_data.datetime,
        y=train_data.price,
        name="train"
    ))

    plot_data.append(Scatter(
        x=test_data.datetime,
        y=test_data.price,
        name="test"
    ))

    plotly.offline.plot({"data": plot_data,
                         "layout": graph_objs.Layout(
                             title="Bitcoin price in training and test data",
                             xaxis=dict(range=[min(train_data["datetime"]), max(test_data["datetime"])])
                         )
                         }, filename="plots/train_and_test_bitcoin_prices.html")


if __name__ == "__main__":
    def main():
        m = 5
        use_cache_for_train = os.path.isfile(train_data_bow_file_name) and os.path.isfile(train_data_word2vec_file_name)
        use_cache_for_test = os.path.isfile(test_data_bow_file_name) and os.path.isfile(test_data_word2vec_file_name)
        print("Preparing data with min_occurrences=" + str(m))
        training_data, word2vec_training_data, testing_data, word2vec_testing_data = preprare_data_for_processing(m, use_cache_for_train, use_cache_for_test, duration=None, sentiment_method="vader")
        log("********************************************************")
        log("Validating for {0} min_occurrences:".format(m))

        # if use_cache_for_train:
        #     print("creating plots")
        #     col_names = ["timestamp", "followers_count", "favourites_count", "id", "screen_name", "text",
        #                  "modified_date", "datetime", "price", "sentiment","sentiment_score"]
        #     data = DataInitializer()
        #     data.initialize("data/one_month_clean_data_with_prices.csv", col_names=col_names, plotting=True)
        #     print("printing head:\n*******************************\n")
        #     # data.processed_data = data.processed_data.reset_index(drop=True)
        #     print(data.processed_data.head())
        #
        #     data.data_model = pd.read_csv(train_data_bow_file_name)
        #     data.wordlist = pd.read_csv("data/wordlist.csv")
        #     data = Plotting(data)
        #     data.plot()


        """
         word2vec + Random Forest
        """
        print("***************************************************\n"
              "FOR WORD2VEC WITH RANDOM FORESTS:\n"
              "***************************************************\n")
        # print("word2vec_training_data head: ", word2vec_training_data.head())
        # print("word2vec_testing_data head: ", word2vec_testing_data.head())
        X_train, X_test, y_train, y_test = train_test_split(word2vec_training_data.iloc[:, 4:],
                                                            word2vec_training_data.iloc[:, 3],
                                                            train_size=0.7, random_state=seed)
                                                            #, stratify = word2vec_training_data.iloc[:, 1]
        if "index" in word2vec_training_data.columns:
            word2vec_training_data.drop(columns=['index'], inplace=True)
        if "index" in word2vec_testing_data.columns:
            word2vec_testing_data.drop(columns=['index'], inplace=True)

        if use_test_data:
            X_train = word2vec_training_data.iloc[:, 4:]
            y_train = word2vec_training_data.iloc[:, 3]

            X_test = word2vec_testing_data.iloc[:, 4:]
            y_test = word2vec_testing_data.iloc[:, 3]

        # regr = RandomForestRegressor(max_depth=2, random_state=0)
        # regr.fit(X_train, y_train)
        # # print(regr.feature_importances_)
        #
        # # Make predictions using the testing set
        # y_pred = regr.predict(X_test)
        #
        # # The mean squared error
        # print("Mean squared error: %.2f"
        #       % mean_squared_error(y_test, y_pred))
        # # Explained variance score: 1 is perfect prediction
        # print('Variance score: %.2f' % r2_score(y_test, y_pred))
        #
        #
        # pred_prices = pd.DataFrame()
        # # Create a column from the datetime variable
        # pred_prices['datetime'] = word2vec_testing_data["timestamp"]
        # pred_prices['price'] = y_pred
        # # Convert that column into a datetime datatype
        # pred_prices['datetime'] = pd.to_datetime(pred_prices['datetime'])
        # # Set the datetime column as the index
        # pred_prices.index = pred_prices['datetime']
        #
        # data_preds = Scatter(
        #                 x=pred_prices.resample('5Min').mean().index,
        #                 y=pred_prices.resample('5Min').mean()["price"],
        #                 mode="lines",
        #                 name="Predicted price"
        #             )
        #
        # original_prices = pd.DataFrame()
        # # Create a column from the datetime variable
        # original_prices['datetime'] = word2vec_testing_data["timestamp"]
        # original_prices['price'] = y_test
        # # Convert that column into a datetime datatype
        # original_prices['datetime'] = pd.to_datetime(original_prices['datetime'])
        # # Set the datetime column as the index
        # original_prices.index = original_prices['datetime']
        #
        # data_originals = Scatter(
        #     x=original_prices.resample('5Min').mean().index,
        #     y=original_prices.resample('5Min').mean()["price"],
        #     mode="lines",
        #     name="Original price"
        # )
        # print(min(pred_prices["datetime"]), max(pred_prices["datetime"]))
        #
        # # print("\n\n\n\n\n","*"*40, [y_test, y_pred], "*"*40,"\n\n\n\n\n")
        # data=[data_originals, data_preds]
        # plotly.offline.plot({"data": data,
        #                      "layout": graph_objs.Layout(
        #                          title="predicted bitcoin price",
        #                          xaxis=dict(range=[min(pred_prices["datetime"]), max(pred_prices["datetime"])])
        #                      )
        #                      }, filename="plots/predicted_prices.html")


        print("####################################################\n"
              "WORD2VEC WITH LSTM:\n"
              "####################################################\n")

        # print("word2vec_training_data head: ", word2vec_training_data.head())

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers.embeddings import Embedding

        if use_test_data:
            X_train = word2vec_training_data.iloc[:, 4:]
            y_train = word2vec_training_data.iloc[:, 3]

            X_test = word2vec_testing_data.iloc[:, 4:]
            y_test = word2vec_testing_data.iloc[:, 3]

        # model_location = './data/model/'
        #
        # # Keras convolutional model
        # batch_size = 32
        # nb_epochs = 5
        # vector_size = 15
        #
        # model = Sequential()
        # model.add(Dense(32, activation='elu', input_dim=203))
        # model.add(Dense(1, activation='linear'))
        # model.compile(optimizer='adam',
        #               loss='mse',
        #               metrics=['accuracy'])
        #
        # # Fit the model
        # model.fit(X_train, y_train,
        #           batch_size=batch_size,
        #           shuffle=True,
        #           epochs=nb_epochs,
        #           validation_data=(X_test, y_test),
        #           callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])
        #
        # score = model.evaluate(X_test, y_test, verbose=0)
        #
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        #
        # # Save the model
        # # serialize model to JSON
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")


        print("####################################################\n"
              "WORD2VEC LSTM WITH CNN:\n"
              "####################################################\n")
        # LSTM for sequence classification for tweets
        # import numpy
        # from keras.models import Sequential
        # from keras.layers import Dense
        # from keras.layers import LSTM
        # from keras.layers.embeddings import Embedding
        #
        # X_train = np.array(X_train)
        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        # X_test = np.array(X_test)
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        #
        # model = Sequential()
        # vocab_size = 10001  # len(top_words) + 1
        # embedding_vector_length = 32
        # model.add(Conv1D(32, kernel_size=3, input_shape=(X_train.shape[1], 1), activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
        # model.add(Dropout(0.25))
        #
        # model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
        # model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
        # model.add(Dropout(0.25))
        #
        # model.add(Flatten())
        #
        # model.add(Dense(128, activation='elu'))
        # model.add(Dense(128, activation='elu'))
        # model.add(Dense(1, activation='linear'))
        #
        # # Compile the model
        # model.compile(loss='mse',
        #               optimizer="adam",
        #               metrics=['accuracy'])
        # model.summary()
        # # Fit the model
        # model.fit(X_train, y_train,
        #           batch_size=10000,
        #           shuffle=True,
        #           epochs=10,
        #           validation_data=(X_test, y_test),
        #           callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])
        #
        # score = model.evaluate(X_test, y_test, verbose=0)
        #
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])


        with open('data/train_sequences', 'rb') as fp:
            train_sequences = pickle.load(fp)
        print("train sequences type: ", type(train_sequences))

        with open('data/train_prices', 'rb') as fp:
            train_prices = pickle.load(fp)

        with open('data/test_sequences', 'rb') as fp:
            test_sequences = pickle.load(fp)

        with open('data/test_prices', 'rb') as fp:
            test_prices = pickle.load(fp)
        word_indexes={}
        with open('data/train_word_indexes', 'rb') as fp:
            word_indexes.update(pickle.load(fp))

        with open('data/test_word_indexes', 'rb') as fp:
            word_indexes.update(pickle.load(fp))

        with open('data/embeddings_index', 'rb') as fp:
            embeddings_indexes = pickle.load(fp)


        # fix random seed for reproducibility
        numpy.random.seed(7)
        # load the dataset but only keep the top n words, zero the rest

        vocab_size = len(word_indexes) + 1
        print("vocab_size: ", vocab_size)
        embedding_matrix = numpy.zeros((vocab_size, 200))
        for word, i in word_indexes.items():
            embedding_vector = embeddings_indexes.get(word)
            # print("len(embedding_vector): ", len(embedding_vector))
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        max_tweet_length = 20
        print("embedding_matrix.shape: ", embedding_matrix.shape)
        print("train_sequences.shape: ", train_sequences.shape)
        # print("embedding matrix head: ", embedding_matrix[0:5])

        print("\n ###############################################\n")
        print("Simple LSTM model to predict bitcoin price from other fields")
        print("\n#################################################\n")
        from sklearn.metrics import mean_absolute_error
        np.random.seed(42)

        # data params
        window_len = 7
        test_size = 0.2
        zero_base = True

        # model params
        lstm_neurons = 100
        epochs = 5
        batch_size = 128
        loss = 'mse'
        dropout = 0.25
        optimizer = 'adam'


        train_data = pd.read_csv("data/one_month_clean_data_with_prices.csv")
        test_data = pd.read_csv("data/one_month_clean_test_data_with_prices.csv")
        prices_data = pd.read_csv("data/train_data_prices.csv")
        prices_data = prices_data[["close", "high", "low", "open", "quoteVolume", "volume", "weightedAverage"]]
        test_data_prices = pd.read_csv("data/test_data_prices.csv")
        test_data_prices = test_data_prices[["close", "high", "low", "open", "quoteVolume", "volume", "weightedAverage"]]
        # train, test, X_train, X_test, y_train, y_test = prepare_data(
        #    prices_data , "close", window_len=window_len, zero_base=zero_base, test_size=test_size)

        # data = prices_data.iloc[:, 1:]
        # targets = prices_data.iloc[:, 0]
        print("shape of prices data: ", prices_data.shape[0], prices_data.shape[1])
        # print("prices data head: ", data[0:5])


        X_train = prices_data.iloc[:, 1:]
        y_train = prices_data.iloc[:, 0]
        X_test = test_data_prices.iloc[:, 1:]
        y_test = test_data_prices.iloc[:, 0]

        # X_train, temp1 = normalise_zero_base(X_train)
        # X_test, temp2 = normalise_zero_base(X_test)

        # X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # X_test = np.array(X_test)
        # y_test = np.array(y_test)

        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # print("targets.shape: ", targets.shape[0])
        # print("targets.shape: ", targets.shape[1])
        # targets = targets.reshape(targets.shape[0], targets.shape[1], 1)

        # X_train, X_test, y_train, y_test = train_test_split(data, targets, train_size=0.7,
        #                                                     test_size=0.3, random_state=42)

        # model = build_lstm_model(
        #     X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        #     optimizer=optimizer)
        # print("model summary: ", model.summary())
        # history = model.fit(
        #     X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        #
        # preds = model.predict(X_test)
        # # print("preds: ", preds)
        # print("mean_absolute_error(preds, y_test): ", mean_absolute_error(preds, y_test))
        # # print("length of predictions: ", len(preds))
        # # print("length of test data: ", len(y_test))
        # plt.figure(1)
        # plt.scatter(range(len(preds)), preds, c='r')
        # plt.scatter(range(len(y_test)), y_test, c='g')
        # plt.show()

        print("\n ###############################################\n")
        print("Simple Linear Regression model to predict bitcoin price using other fields in prices data")
        print("\n#################################################\n")

        # # Create linear regression object
        # regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
        #
        # # Train the model using the training sets
        # regr.fit(X_train, y_train)
        #
        # # Make predictions using the testing set
        # y_pred = regr.predict(X_test)
        #
        # # The coefficients
        # print('Coefficients: \n', regr.coef_)
        # # The mean squared error
        # print("Mean squared error: %.2f"
        #       % mean_squared_error(y_test, y_pred))
        # # Explained variance score: 1 is perfect prediction
        # print('Variance score: %.2f' % r2_score(y_test, y_pred))
        #
        # # Plot outputs
        # # print("X_test head: ", X_test.head())
        # plt.scatter(X_test["open"], y_test, color='black')
        # plt.plot(X_test["open"], y_pred, color='blue', linewidth=3)
        #
        # plt.xticks(())
        # plt.yticks(())
        #
        # plt.show()


        print("\n ###############################################\n")
        print("Simple Linear Regression model to predict bitcoin price using other fields in prices data")
        print("\n#################################################\n")

        # print("shape of trained sequences: ", train_sequences.shape)
        #
        # # Create linear regression object
        # regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
        #
        # # Train the model using the training sets
        # regr.fit(train_sequences, train_prices)
        #
        # # Make predictions using the testing set
        # print(test_sequences[test_sequences.isnull().any(axis=1)])
        # y_pred = regr.predict(test_sequences)
        #
        # # The coefficients
        # print('Coefficients: \n', regr.coef_)
        # # The mean squared error
        # print("Mean squared error: %.2f"
        #       % mean_squared_error(test_prices, y_pred))
        # # Explained variance score: 1 is perfect prediction
        # print('Variance score: %.2f' % r2_score(test_prices, y_pred))
        #
        # # Plot outputs
        # # print("X_test head: ", X_test.head())
        # plt.scatter(test_sequences["open"], test_prices, color='black')
        # plt.plot(test_sequences["open"], test_prices, color='blue', linewidth=3)
        #
        # plt.xticks(())
        # plt.yticks(())
        #
        # plt.show()


        print("\n ###############################################\n")
        print("LSTM model to predict bitcoin price with text sequences and other price related data")
        print("\n#################################################\n")
        # print("train sequences: ", train_sequences.head())
        # print("*"*40, "\n"*3)
        # print("test sequences: ", test_sequences.head())
        # print("*" * 40, "\n" * 3)
        # print("train prices: ", train_prices.head())
        # print("*" * 40, "\n" * 3)
        # print("test prices: ", test_prices.head())
        # print("*" * 40, "\n" * 3)
        # train_sequences = np.array(train_sequences)
        # train_sequences = train_sequences.reshape(train_sequences.shape[0], train_sequences.shape[1], 1)
        #
        # test_sequences = np.array(test_sequences)
        # test_sequences = test_sequences.reshape(test_sequences.shape[0], test_sequences.shape[1], 1)
        #
        # # define model
        # model = Sequential()
        # model.add(LSTM(lstm_neurons, input_shape=(train_sequences.shape[1], train_sequences.shape[2]), return_sequences=True))
        # model.add(Dropout(dropout))
        # # e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=26, trainable=False)
        # # model.add(e)
        # # model.add(Flatten())
        #
        # model.add(LSTM(128))
        # model.add(Dense(1, activation='linear'))
        # model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        # print("model.summary(): \n", model.summary())
        # print("np.isnan(X_train).any(): ", np.argwhere(np.isnan(test_sequences).any()))
        # print("test sequences: ", test_sequences)
        #
        # model.fit(train_sequences, train_prices, nb_epoch=5, batch_size=100)
        # loss, accuracy = model.evaluate(test_sequences, test_prices, verbose=0)
        # print('Test loss:', loss)
        # print("Accuracy: %.2f%%" % (accuracy * 100))

        print("\n ###############################################\n")
        print("Simple LSTM model to predict bitcoin price with windows")
        print("\n#################################################\n")

        # X_train, X_test, y_train, y_test, mean, std = prepare_data(
        #    prices_data, test_data_prices, "close", window_len=window_len, zero_base=zero_base, test_size=test_size)
        #
        # # print("shape of prices data: ", data.shape[0], data.shape[1])
        # # print("prices data head: ", data[0:5])
        #
        # # print("X train shape: ", X_train[1:].shape)
        # print("np.isnan(X_train).any(): ", np.argwhere(np.isnan(X_train[1:]).any()))
        # model = Sequential()
        # model.add(LSTM(26, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        # model.add(Dropout(dropout))
        # model.add(LSTM(26))
        # model.add(Dense(units=1))
        # model.add(Activation("relu"))
        # # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)
        # model.compile(loss="mean_squared_error", optimizer=optimizer)
        #
        # print("model summary: ", model.summary())
        # print("y_test shape: ", y_test.shape)
        # print("X_test head: ", y_test[0:5])
        # print("y_train shape: ", y_train.shape)
        # print("y_train head: ", y_train[0:5])
        # history = model.fit(
        #     X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_test, y_test))
        #
        # preds = model.predict(X_test)
        # # preds = scaler.inverse_transform(preds)
        #
        # print("preds and originals: ", [preds, test_data_prices["close"][:-7]])
        # print("mean_absolute_error(preds, y_test): ", mean_absolute_error(preds, test_data_prices["close"][:-7]))
        # # print("length of predictions: ", len(preds))
        # # print("length of test data: ", len(y_test))
        # plt.figure(2)
        # plt.subplot(211)
        # # plt.scatter(range(20), preds[0:20], c='r')
        # # plt.scatter(range(20), test_data_prices["close"][0:20], c='g')
        #
        # plt.scatter(range(len(preds)), preds, c='r')
        # plt.scatter(range(len(test_data_prices["close"])), test_data_prices["close"], c='g')
        #
        # plt.subplot(212)
        # plt.plot(history.history['loss'])
        # plt.show()



        print("####################################################\n")
        print("Simple LSTM for Sequence Classification with Dropout\n")
        print("####################################################\n")

        # fix random seed for reproducibility
        # numpy.random.seed(7)
        #
        #
        # # define model
        # model = Sequential()
        # e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_tweet_length, trainable=False)
        # model.add(e)
        # model.add(Dropout(0.2))
        # model.add(LSTM(128, return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(3, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        # model.fit(train_sequences, train_sentiments, nb_epoch=3, batch_size=100)
        # # Final evaluation of the model
        # loss, accuracy = model.evaluate(test_sequences, test_sentiments, verbose=0)
        # print('Test loss:', loss)
        # print("Accuracy: %.2f%%" % (accuracy * 100))

        print("####################################################\n")
        print("LSTM for Sequence Classification with CNN\n")
        print("####################################################\n")

        # fix random seed for reproducibility
        # numpy.random.seed(7)
        #
        # create the model
        print("train sequences shape: ", train_sequences.shape)
        print("train prices shape: ", train_prices.shape)

        print("test sequences shape: ", test_sequences.shape)
        print("test prices shape: ", test_prices.shape)

        print("test sequences: \n\n", test_sequences,"\n"*10)
        print("test prices: \n\n", test_prices,"\n"*10)

        with open("test.txt",'wb') as outfile:
            outfile.write("\n".join(test_sequences))
        with open("test_prices.txt",'wb') as outfile:
            outfile.write("\n".join(test_prices))

        model = Sequential()
        print("embedding_matrix shape: ", embedding_matrix.shape)
        e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=26, trainable=False)
        model.add(e)
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='elu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        try:
            model.fit(train_sequences, train_prices, nb_epoch=3, batch_size=100)
        except Exception as e:
            print(e)

        # Final evaluation of the model
        loss, accuracy = model.evaluate(test_sequences, test_prices, verbose=0)
        print('Test loss:', loss)
        print("Accuracy: %.2f%%" % (accuracy * 100))


        print("####################################################\n")
        print("Simple RNN\n")
        print("####################################################\n")

        # fix random seed for reproducibility
        # numpy.random.seed(7)
        #
        # # create the model
        # model = Sequential()
        # e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_tweet_length, trainable=False)
        # model.add(e)
        # model.add(Dropout(0.2))
        # model.add(SimpleRNN(max_tweet_length, return_sequences=False))
        # model.add(Dense(3, activation='softmax'))
        #
        # adam = optimizers.Adam(lr=0.001)
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # print(model.summary())
        # model.fit(train_sequences, train_sentiments, nb_epoch=3, batch_size=100)
        #
        # # Final evaluation of the model
        # scores = model.evaluate(test_sequences, test_sentiments, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1] * 100))


        print("####################################################\n")
        print("Deep RNN\n")
        print("####################################################\n")

        # model = Sequential()
        # model.add(LSTM(15, input_length=max_tweet_length, return_sequences=True))
        # model.add(LSTM(15, return_sequences=True))
        # model.add(LSTM(15, return_sequences=True))
        # model.add(LSTM(15, return_sequences=False))
        # model.add(Dense(3, activation='softmax'))
        #
        # adam = optimizers.Adam(lr=0.001)
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # model.fit(train_sequences, train_sentiments, nb_epoch=3, batch_size=100)
        #
        # # Final evaluation of the model
        # scores = model.evaluate(test_sequences, test_sentiments, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1] * 100))


main()
