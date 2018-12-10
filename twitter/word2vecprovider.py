# The word2vec allows to transform words into vectors of numbers. Those vectors represent abstract features,
# that describe the word similarities and relationships (i.e co-occurence).

# A pre trained word2vec model which is trained on over 2 billion of tweets with 200 dimensions (one vector consists
# of 200 numbers) is used in this notebook.


import gensim


class Word2VecProvider(object):
    word2vec = None
    dimensions = 0

    def load(self, path_to_word2vec):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=False)
        self.word2vec.init_sims(replace=True)
        self.dimensions = self.word2vec.vector_size

    def get_vector(self, word):
        if word not in self.word2vec.vocab:
            return None

        return self.word2vec.syn0norm[self.word2vec.vocab[word].index]

    def get_similarity(self, word1, word2):
        if word1 not in self.word2vec.vocab or word2 not in self.word2vec.vocab:
            return None
        return self.word2vec.similarity(word1, word2)


