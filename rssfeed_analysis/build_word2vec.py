import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
from collections import Counter

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
# LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentece = gensim.models.deprecated.doc2vec.LabeledSentence

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense, Activation


from data_cleaner import DataCleaner
from sentiment_analysis import Sentiments
from data_initializer import DataInitializer
from data_cleaning import DataCleaning
import nltk


n=100
n_dim = 200
codes = {'positive':1, 'negative':-1, 'neutral':0}
whitelist = ["n't","not"]

# Below function loads the dataset and extracts the two columns
#   tweet text
#   sentiment of the tweet

def ingest():
    seed = 1000

    data = DataInitializer()
    data.initialize("data/train.csv")

    data=DataCleaning(data)
    data.cleanup(DataCleaner())

    data = Sentiments(data)
    data.sentiment_analysis_by_text()


    data = data.processed_data[['sentiment', 'text']]
    print('dataset loaded with shape', data.shape)
    print("Distribution of sentiments: ", pd.Series(data["sentiment"]).value_counts())

    # data["sentiment"] = data["sentiment"].map(codes)

    return data

data = ingest()


def tokenize(tweets, tokenizer=nltk.word_tokenize):
    def tokenize_row(row):
        row["text"] = tokenizer(row["text"])
        row["tokens"] = [] + row["text"]
        return row

    tweets = tweets.apply(tokenize_row, axis=1)
    return tweets


words = Counter()
for idx in data.index:
    words.update(data.loc[idx, "text"])

words.most_common(5)


# The results of the tokenization should now be cleaned to remove lines with 'NC' which are resulted from tokenization error
def postprocess(data, min_occurrences=3, max_occurences=500, stopwords=nltk.corpus.stopwords.words("english"),
                       whitelist=None):
    # data = data.head(n)
    print("first 5 rows of data before tokenization: ", data.head(5))
    # data['tokens'] = data['text'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = tokenize(data)
    for index, tweet in data.iterrows():
        tweet["tokens"] = [word for word in tweet.tokens if word not in stopwords]

    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


data = postprocess(data)
print("first 5 rows of data post processing: ", data.head(5))

# Build the word2vec model

x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).sentiment), test_size=0.2)


# Before feeding lists of tokens into the word2vec model, turn them into LabeledSentence objects beforehand
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')



# check the first element from x_train
print("The first element from x_train", x_train[0])


# so each element is basically some object with two attributes: a list (of tokens) and a label
# Now we are ready to build the word2vec model from x_train i.e. the corpus
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=n, epochs=10)


# Once the model is built and trained on the corpus of tweets use it to convert words to vectors.
# Each word will be a 200-dimension vector. You get vectors of the words of the corpus.
print("For example - tweet_w2v['bitcoin']: ", tweet_w2v['bitcoin'])


# Use most_similar() in Word2Vec gensim implementation to get the top n similar ones.
print("For example - words similar in meaning and context to bitcoin: ", tweet_w2v.most_similar('bitcoin'))



# Building a sentiment classifier
print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))



# define a function that, given a list of tweet tokens, creates an averaged tweet vector
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec




# Use above function to convert x_train and and x_test into list of vectors. The function also scales each column to have zero mean and unit standard deviation.
from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)


print("final data for training: \n"+"*"*100+"\n", train_vecs_w2v, "\n")
# feed the vectors created above into a neural network classifier. Its very easy to define layers and activation functions in Keras.
# Below defined is a basic 2-layer architecture.
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)


# Evaluate the trained model on the test set:
score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score[1])