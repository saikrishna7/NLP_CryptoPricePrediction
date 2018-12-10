from extra_features import ExtraFeatures
import pandas as pd
import numpy as np
import nltk

class TwitterData(ExtraFeatures):

    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def build_final_model(self, word2vec_provider, stopwords=nltk.corpus.stopwords.words("english")):
        try:
            self.processed_data.head()
        except Exception:
            print("its not a dataframe")
        print("first 5 rows before building final data model", self.processed_data[:5])
        whitelist = self.whitelist
        print("whitelist: ", self.whitelist)
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        similarity_columns = ["bad_similarity", "good_similarity", "information_similarity"]
        label_column = []
        if not self.is_testing:
            label_column = ["sentiment"]

        columns = label_column + ["timestamp", "sentiment_score", "close"] + extra_columns + similarity_columns + list(
            map(lambda i: "word2vec_{0}".format(i), range(0, word2vec_provider.dimensions))) + list(
            map(lambda w: w + "_bow", self.wordlist))
        # print("columns of self.processed_data: ", self.processed_data.columns)
        labels = []
        rows = []

        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "sentiment"]
                labels.append(current_label)
                current_row.append(current_label)

            current_row.append(pd.to_datetime(self.processed_data.loc[idx, "timestamp"]))
            current_row.append(self.processed_data.loc[idx, "sentiment_score"])
            current_row.append(self.processed_data.loc[idx, "close"])
            # current_row.append(self.processed_data.loc[idx, "id"])

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # average similarities with words
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            for main_word in map(lambda w: w.split("_")[0], similarity_columns):
                current_similarities = [abs(sim) for sim in
                                        map(lambda word: word2vec_provider.get_similarity(main_word, word.lower()), tokens) if sim is not None]
                if len(current_similarities) <= 1:
                    current_row.append(0 if len(current_similarities) == 0 else current_similarities[0])
                    continue
                max_sim = max(current_similarities)
                min_sim = min(current_similarities)
                current_similarities = [((sim - min_sim) / (max_sim - min_sim)) for sim in
                                        current_similarities]  # normalize to <0;1>
                current_row.append(np.array(current_similarities).mean())

            # add word2vec vector
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            current_word2vec = []
            for _, word in enumerate(tokens):
                vec = word2vec_provider.get_vector(word.lower())
                if vec is not None:
                    current_word2vec.append(vec)
            # print("shape of current_word2vec: ", len(current_word2vec))

            # print(type(current_word2vec))
            # print(np.array(current_word2vec).shape)
            # print(current_word2vec)
            if np.array(current_word2vec).shape == (0,):
                continue
            else:
                # print("current_word2vec: ", current_word2vec)
                # print("np.array(current_word2vec).mean(axis=0): ", np.array(current_word2vec).mean(axis=0))
                averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))
                current_row += averaged_word2vec

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)
        # print(rows[0:2])
        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels