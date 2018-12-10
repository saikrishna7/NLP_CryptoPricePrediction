from data_initializer import DataInitializer
from data_cleaning import DataCleaning
from data_tokenize import DataTokenize
from data_cleaner import DataCleaner
from word_list import WordList
from bagofwords import BagOfWords
from sentiment_analysis import Sentiments
from classification import Classification
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from plotting import Plotting
from redditdata import RedditData
import os
from word2vecprovider import Word2VecProvider
from get_prices_data import GetPricesData
from sklearn.externals import joblib

from collections import Counter
from datetime import datetime
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
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


seed = 1000
use_test_data = True

min_occurrences=5
test_data_word2vec_file_name = "data/processed_test_word2vec_bow_" + str(min_occurrences) + ".csv"
train_data_word2vec_file_name = "data/processed_train_word2vec_bow_" + str(min_occurrences) + ".csv"

test_data_bow_file_name = "data/processed_test_bow_" + str(min_occurrences) + ".csv"
train_data_bow_file_name = "data/processed_train_bow_" + str(min_occurrences) + ".csv"


def preprocess(data_path, is_testing, min_occurrences=5, cache_bow_output=None, cache_word2vec_output=None, duration=None):
    if duration:
        data = DataInitializer()
        data.initialize(data_path, is_testing, duration=duration)
    else:
        data = DataInitializer()
        data.initialize(data_path, is_testing)


    if os.path.isfile("data/BTC.csv"):
        prices_data = GetPricesData()
        prices_data.main()

    data = DataCleaning(data, is_testing)
    data.cleanup(DataCleaner(is_testing))

    if is_testing:
        print("Testing data shape:", data.processed_data.shape)
    else:
        print("Training data shape:", data.processed_data.shape)

    data = Sentiments(data)
    data.sentiment_analysis_by_text()
    print("First five rows with sentiment: ", data.processed_data.head())
    if is_testing:
        data.processed_data.to_csv("data/clean_test_with_sentiments.csv", sep=',', encoding='utf-8', index=False)
        # os.remove(data_path)
    else:
        data.processed_data.to_csv("data/clean_train_with_sentiments.csv", sep=',', encoding='utf-8', index=False)
        # os.remove(data_path)

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
    word2vec.load("../twitter/data/glove.twitter.27B.200d.txt")

    word2vec_data = RedditData(word2vec_data)
    word2vec_data.build_final_model(word2vec)

    word2vec_data_model = word2vec_data.data_model
    if "index" in word2vec_data_model.columns:
        word2vec_data_model.drop("index", axis=1, inplace=True)
    word2vec_data_model.dropna(axis=0, inplace=True)
    word2vec_data_model.reset_index(inplace=True)
    word2vec_data_model.index = word2vec_data_model['timestamp_ms']
    print("final word2vec data model: \n", word2vec_data_model.head(), "\n")

    """
    Tokenizing the data
    """
    texts = []
    sentiments = []
    tokenized_data = pd.DataFrame()
    for text in data.processed_data["summary"]:
        texts.append(text)
    for sentiment in data.processed_data["sentiment"]:
        sentiments.append(sentiment)
    print("texts: ", texts[0:5])
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=200)

    print("\n\n##################################################\npadded sequence head: \n", padded_sequences[0:5])
    print("\n####################################################\n padded sequence length \n", len(padded_sequences))


    if not is_testing:
        data = Plotting(data)
        data.plot()

    if cache_bow_output is not None:
        data.data_model.to_csv(cache_bow_output, index=False, float_format="%.6f")
        word2vec_data_model.to_csv(cache_word2vec_output, index=False, float_format="%.6f")
        with open('sequences', 'wb') as fp:
            pickle.dump(padded_sequences, fp)
        with open('sentiments', 'wb') as fp:
            pickle.dump(sentiments, fp)

    return data.data_model, word2vec_data_model


def preprare_data(min_occurrences, use_cache, duration):
    training_data = None
    testing_data = None
    print("Loading data...")
    if duration is not None:
        if os.path.isfile(test_data_word2vec_file_name) and os.path.isfile(test_data_bow_file_name):
            os.remove(test_data_word2vec_file_name)
            os.remove(test_data_bow_file_name)
        testing_data, word2vec_testing_data = preprocess("data/clean_test.csv", True,
        min_occurrences,test_data_bow_file_name, test_data_word2vec_file_name, duration)
    if not os.path.isfile("data/BTC.csv"):
        prices_data=GetPricesData()
        prices_data.main()
    if use_cache:
        train_data_initializer_obj = DataInitializer()
        train_data_initializer_obj.initialize(None, from_cached_bow=train_data_bow_file_name, from_cached_word2vec = train_data_word2vec_file_name)
        training_data = train_data_initializer_obj.data_model
        word2vec_training_data = train_data_initializer_obj.word2vec_data

        test_data_initializer_obj = DataInitializer()
        test_data_initializer_obj.initialize(None, from_cached_bow=test_data_bow_file_name, from_cached_word2vec = test_data_word2vec_file_name)
        word2vec_testing_data = test_data_initializer_obj.word2vec_data
        testing_data = test_data_initializer_obj.data_model
        print("Loaded from cached files...")
    else:
        print("Preprocessing data...")
        training_data, word2vec_training_data = preprocess("data/clean_train.csv", False, min_occurrences, train_data_bow_file_name, train_data_word2vec_file_name)
        testing_data, word2vec_testing_data = preprocess("data/clean_test.csv", True, min_occurrences, test_data_bow_file_name, test_data_word2vec_file_name)
        print("Data preprocessed & cached...")

    return training_data, word2vec_training_data, testing_data, word2vec_testing_data


def log(text):
    print(text)
    with open("log.txt", "a") as log_file:
        log_file.write(str(text) + "\n")


if __name__ == "__main__":
    def main():
        m=5
        use_cache = os.path.isfile(train_data_bow_file_name) and os.path.isfile(
            test_data_bow_file_name) and os.path.isfile(train_data_word2vec_file_name) and os.path.isfile(
            test_data_word2vec_file_name)
        print("Preparing data with min_occurrences=" + str(m))

        training_data, word2vec_training_data, testing_data, word2vec_testing_data = preprare_data(m, use_cache, duration=None)

        log("********************************************************")
        log("Validating for {0} min_occurrences:".format(m))
        if use_cache:
            col_names=["author", "title", "timestamp_ms", "summary", "sentiment", "sentiment_score"]
            data = DataInitializer()
            data.initialize("data/clean_train_with_sentiments.csv", col_names=col_names)
            print("printing head:\n*******************************\n")
            data.processed_data = data.processed_data.reset_index(drop=True)
            # data.processed_data.rename(columns={"author": "timestamp_ms", "timestamp_ms", "summary"})
            print(data.processed_data.head())
            original_data=data.processed_data
            data.data_model = pd.read_csv(train_data_bow_file_name)
            data.wordlist = pd.read_csv("data/wordlist.csv")
            data = Plotting(data)
            data.plot()

        """
        Naive Bayes
        """
        print("***************************************************\n"
              "FOR NAIVE BAYES:\n"
              "***************************************************\n")
        print("testing_data shape: ", testing_data.shape)
        print("testing_data head: ", testing_data.head())
        X_train, X_test, y_train, y_test = train_test_split(training_data.iloc[:, 1:], training_data.iloc[:, 0],
                                                            train_size=0.7, stratify=training_data.iloc[:, 0],
                                                            random_state=seed)


        if use_test_data:
            X_train = training_data.iloc[:, 1:]
            y_train = training_data.iloc[:, 0]

            X_test = testing_data.iloc[:, 1:]
            y_test = testing_data.iloc[:, 0]
        precision, recall, accuracy, f1 = Classification.test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())

        # nb_acc = Classification.cv(BernoulliNB(), training_data.iloc[:, 1:], training_data.iloc[:, 0])


        """
        Random Forest
        """
        print("***************************************************\n"
              "FOR RANDOM FORESTS:\n"
              "***************************************************\n")
        X_train, X_test, y_train, y_test = train_test_split(training_data.iloc[:, 1:], training_data.iloc[:, 0],
                                                            train_size=0.7, stratify=training_data.iloc[:, 0],
                                                            random_state=seed)
        if use_test_data:
            X_train = training_data.iloc[:, 1:]
            y_train = training_data.iloc[:, 0]

            X_test = testing_data.iloc[:, 1:]
            y_test = testing_data.iloc[:, 0]

        precision, recall, accuracy, f1 = Classification.test_classifier(X_train, y_train, X_test, y_test,
                                                          RandomForestClassifier(random_state=seed, n_estimators=403,
                                                                                 n_jobs=-1))
        # rf_acc = Classification.cv(RandomForestClassifier(n_estimators=403, n_jobs=-1, random_state=seed), training_data.iloc[:, 1:],
        #                            training_data.iloc[:, 0])


        """
         Word2Vec + Random Forest
        """
        print("***************************************************\n"
              "FOR WORD2VEC WITH RANDOM FORESTS:\n"
              "***************************************************\n")

        X_train, X_test, y_train, y_test = train_test_split(word2vec_training_data.iloc[:, 2:],
                                                            word2vec_training_data.iloc[:, 1],
                                                            train_size=0.7, stratify=word2vec_training_data.iloc[:, 1],
                                                            random_state=seed)
        # word2vec_training_data.drop(columns=['index'], inplace=True)
        # word2vec_testing_data.drop(columns=['index'], inplace=True)
        print("word2vec_training_data.columns: ", word2vec_training_data.columns)

        if use_test_data:
            X_train = word2vec_training_data.iloc[:, 3:]
            y_train = word2vec_training_data.iloc[:, 1]

            X_test = word2vec_testing_data.iloc[:, 3:]
            y_test = word2vec_testing_data.iloc[:, 1]

        precision, recall, accuracy, f1 = Classification.test_classifier(X_train, y_train, X_test, y_test,
                                                                         RandomForestClassifier(n_estimators=403,
                                                                                                n_jobs=-1,
                                                                                                random_state=seed))


        print("***************************\n")
        print("For Regression\n")
        print("***************************\n")

        print("first five rows: ", word2vec_training_data.head())
        X_train = word2vec_training_data.iloc[:, 4:]
        y_train = word2vec_training_data.iloc[:, 3]

        X_test = word2vec_testing_data.iloc[:, 4:]
        y_test = word2vec_testing_data.iloc[:, 3]

        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X_train, y_train)
        # print(regr.feature_importances_)
        # print(regr.predict([[0, 0, 0, 0]]))
        predictions = regr.predict(X_test)
        print("predictions:\n*****************************", predictions, "\n****************************\n")
        print("Real values:\n*****************************", y_test, "\n****************************\n")
        print("score: ", regr.score(X_test, y_test))

        redditposts_sentiment = pd.DataFrame()
        # Create a column from the datetime variable
        redditposts_sentiment['datetime'] = word2vec_testing_data["timestamp_ms"]
        redditposts_sentiment['sentiment_score'] = predictions
        # Convert that column into a datetime datatype
        redditposts_sentiment['datetime'] = pd.to_datetime(redditposts_sentiment['datetime'])
        # Set the datetime column as the index
        redditposts_sentiment.index = redditposts_sentiment['datetime']

        reddit_posts = [Scatter(
                        x=redditposts_sentiment.resample('5Min').mean().index,
                        y=redditposts_sentiment.resample('5Min').mean()["sentiment_score"],
                        mode="lines"
                    )]

        plotly.offline.plot({"data": reddit_posts, "layout": graph_objs.Layout(title="Reddit posts sentiment")},
                            filename='plots/redditposts_predicted_sentiment.html')

        print("***************************************************\n"
              "FOR KERAS:\n"
              "***************************************************\n")
        X_train, X_test, y_train, y_test = train_test_split(word2vec_training_data.iloc[:, 2:],
                                                            word2vec_training_data.iloc[:, 1],
                                                            train_size=0.7, stratify=word2vec_training_data.iloc[:, 1],
                                                            random_state=seed)
        # word2vec_training_data.drop(columns=['index'], inplace=True)
        # word2vec_testing_data.drop(columns=['index'], inplace=True)
        print("word2vec_training_data.columns: ", word2vec_training_data.columns)
        if use_test_data:
            X_train = word2vec_training_data.iloc[:, 3:]
            y_train = word2vec_training_data.iloc[:, 1]

            X_test = word2vec_testing_data.iloc[:, 3:]
            y_test = word2vec_testing_data.iloc[:, 1]


        # params
        use_gpu = True

        config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                                inter_op_parallelism_threads=multiprocessing.cpu_count(),
                                allow_soft_placement=True,
                                device_count={'CPU': 1,
                                              'GPU': 1 if use_gpu else 0})

        session = tf.Session(config=config)
        K.set_session(session)

        model_location = './data/model/'

        # Keras convolutional model
        batch_size = 32
        nb_epochs = 10
        vector_size = 200
        # Tweet max length (number of tokens)
        max_tweet_length = 15
        print("X_train shape:", X_train.shape)
        print("Y_train shape:", y_train.shape)
        print("x_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        model = Sequential()

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=204))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  shuffle=True,
                  epochs=nb_epochs,
                  validation_data=(X_test, y_test),
                  callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])

        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Save the model
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        print("****************************\n")
        print("Building a Neural Network\n")
        print("****************************\n")

        with open('sequences', 'rb') as fp:
            sequences = pickle.load(fp)

        with open('sentiments', 'rb') as fp:
            sentiments = pickle.load(fp)


        EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model = Sequential()
        model.add(Embedding(20000, 128, input_length=200))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(128))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(sequences, np.array(sentiments), validation_split=0.5, epochs=10)


main()