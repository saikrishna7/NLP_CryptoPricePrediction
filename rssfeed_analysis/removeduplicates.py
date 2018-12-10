import pandas as pd
import time

class RemoveDuplicates:
    shingles=[]
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    import itertools


    # from lsh import lsh, minhash # https://github.com/mattilyra/lsh

    # a pure python shingling function that will be used in comparing
    # LSH to true Jaccard similarities


    def get_shingles(self, char_ngram=5):
        """Create a set of overlapping character n-grams.

        Only full length character n-grams are created, that is the first character
        n-gram is the first `char_ngram` characters from text, no padding is applied.

        Each n-gram is spaced exactly one character apart.

        Parameters
        ----------

        text: str
            The string from which the character n-grams are created.

        char_ngram: int (default 5)
            Length of each character n-gram.
        """
        for index, tweet in self.processed_data[1:].iterrows():
            # print("tweet['text'] type:", type(len(tweet["text"])))
            # print("char_ngram type: ",type(char_ngram))
            # print(len(tweet["text"]))
            # print(char_ngram)
            self.shingles.append(
                set(tweet["text"][head:head + char_ngram] for head in range(0, len(tweet["text"]) - char_ngram)))
        print(self.shingles[:30])

    def jaccard(self, set_a, set_b):
        """Jaccard similarity of two sets.

        The Jaccard similarity is defined as the size of the intersection divided by
        the size of the union of the two sets.

        Parameters
        ---------
        set_a: set
            Set of arbitrary objects.

        set_b: set
            Set of arbitrary objects.
        """
        intersection = set_a & set_b
        union = set_a | set_b
        try:
            return len(intersection) / len(union)
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)
        return False

    duplicates = []

    def print_duplicates(self):
        start = time.time()
        self.get_shingles(5)
        for i_doc in range(len(self.shingles)):
            for j_doc in range(i_doc + 1, len(self.shingles)):
                jaccard_similarity = self.jaccard(self.shingles[i_doc], self.shingles[j_doc])
                is_duplicate = jaccard_similarity >= 0.75
                if is_duplicate:
                    self.duplicates.append((i_doc, j_doc, jaccard_similarity))
        print(len(self.duplicates))
        print(pd.DataFrame(self.duplicates, columns=['Document ID', 'Document ID', 'Jaccard Similarity']))
        end = time.time()
        print("total time taken:", end-start)
        return self.processed_data

