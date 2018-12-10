from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
from collections import Counter
from six.moves import cPickle
import gensim.models.word2vec as w2v
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys
import io
import re
import json


# Encoding the words
# The embedding lookup requires that we pass in integers to the neural network. The easiest way to do this is to
# create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews
# into integers so they can be passed into the network.

# Create a dictionary that maps vocab words to integers
from collections import Counter

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# Convert the reviews to integers, same shape as reviews list, but with integers
reviews_ints = []
for each in tweets:
    tweet_ints.append([vocab_to_int[word] for word in each.split()])
print(len(tweet_ints))


# Convert labels to 1s and 0s for 'positive' and 'negative'
# print(labels_org)
labels = np.array([1 if l == "positive" else 0 for l in labels_org.split()])
# print(labels)
print(len(labels))


# If you built labels correctly, you should see the next output.
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Tweets can be of variable length i'e with zero length tokens to maximum retweet length that could be
# way too many steps for the RNN. So truncate the tweet tokens to 200 steps. For tweets shorter than 200,  pad with 0s.
# For tweets longer than 200, we can truncate them to the first 200 characters.


# Filter out that review with 0 length
reviews_ints = [r[0:200] for r in reviews_ints if len(r) > 0]

review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Now, create an array features that contains the data we'll pass to the network. The data should come from review_ints,
#  since we want to feed integers to the network. Each row should be 200 elements long. For reviews shorter than
# 200 words, left pad with 0s. That is, if the tweet is ['crypto', 'prices', 'falling'], [117, 18, 128] as integers,
# the row will look like [0, 0, 0, ..., 0, 117, 18, 128]. For tweets longer than 200, use the first 200 words as the
# feature vector.

seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
# print(features[:10,:100])
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
features[:10,:100]

print("length of features: ", len(features))
print("type of features: ", type(features))
print("feature 41", features[41])
print("Length of feature 41: ", len(features[41]))
print("feature 41 integer encodings", tweets_ints[41])
print("feature 41 integer encodings: ", len(tweets_ints[41]))


# Split data into training, validation, and test sets.
split_frac = 0.8

split_index = int(split_frac * len(features))

train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape),
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))
