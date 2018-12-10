from data_tokenize import DataTokenize
from collections import Counter
import pandas as pd
import nltk

class WordList(DataTokenize):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    whitelist = ["n't", "not"]
    wordlist = []

    def build_wordlist(self, min_occurrences=5, max_occurences=50000, stopwords=nltk.corpus.stopwords.words("english"),
                       whitelist=None):
        self.wordlist = []
        whitelist = self.whitelist if whitelist is None else whitelist
        import os
        if os.path.isfile("data/wordlist.csv"):
            word_df = pd.read_csv("data/wordlist.csv")
            word_df = word_df[word_df["occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "word"])
            return


        words = Counter()
        for idx in self.processed_data.index:
            words.update(self.processed_data.loc[idx, "summary"])

        whitelist = ["n't", "not"]
        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]

        for word in words:
            if type(word) is not str:
                words.remove(word)


        word_df = pd.DataFrame(
            data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                  "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
            columns=["word", "occurrences"])

        word_df.to_csv("data/wordlist.csv", index_label="idx")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]