from data_cleaning import DataCleaning
import nltk

class DataTokenize(DataCleaning):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    words = set(nltk.corpus.words.words())

    def stem(self, stemmer=nltk.PorterStemmer()):
        def stem_and_join(row):
            row["summary"] = list(map(lambda str: stemmer.stem(str.lower()), row["summary"]))
            return row

        self.processed_data = self.processed_data.apply(stem_and_join, axis=1)

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenize_row(row):
            row["summary"] = tokenizer(row["summary"])
            row["tokenized_text"] = [] + row["summary"]
            return row

        self.processed_data = self.processed_data.apply(tokenize_row, axis=1)