from word_list import WordList
import pandas as pd

class BagOfWords(WordList):
    def __init__(self, data, words, is_testing):
        self.processed_data = data
        self.wordlist = words
        self.is_testing = is_testing
        # print("head of wordlist in bag of words: ", self.wordlist)
        # print("type of data received in bag of words class : ", type(self.processed_data))
    def build_data_model(self):
        label_column = []
        columns=[]
        # if not self.is_testing:
        #     label_column = ["label"]
        label_column = ["label"]   # For now calculating the sentiment for test data as well
        for word in self.wordlist:
            if type(word) is not str:
                self.wordlist.remove(word)
                print(word)

        columns = label_column + list(map(lambda w: w + "_bow", self.wordlist))


        # except Exception as e:
        #     print(label_column)
        #     print(self.wordlist)
        # print("columns: ", columns)
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            # if not self.is_testing:
            #     # add label
            #     current_label = self.processed_data.loc[idx, "sentiment"]
            #     labels.append(current_label)
            #     current_row.append(current_label)
            current_label = self.processed_data.loc[idx, "sentiment"]
            labels.append(current_label)
            current_row.append(current_label)
            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels
