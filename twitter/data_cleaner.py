import re as regex
import nltk
import pandas as pd


class DataCleaner:
    def __init__(self, is_testing):
        self.is_testing = is_testing


    words = set(nltk.corpus.words.words())

    def iterate(self):
        for cleanup_method in [self.remove_duplicates,
                               self.remove_noneng_tweets,
                               self.remove_urls,
                               self.remove_usernames,
                               self.remove_na,
                               self.remove_special_chars,
                               self.remove_numbers,
                               self.processHashtags,
                               self.processHandles,
                               self.processUrls,
                               self.processRepeatings
                                ]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets.loc[:, "text"].replace(regexp, "", inplace=True)
        return tweets

    def remove_duplicates(self, tweets):
        print("removing duplicate tweets")
        tweets.drop_duplicates(subset=["text"], inplace=True)
        print("Number of tweets after removing duplicates: ",tweets.shape)
        return tweets


    def remove_noneng_tweets(self, tweets):
        tweets_text=[]
        print("removing non english tweets for ", ("Train set", "Test set")[self.is_testing == True])

        try:
            for i in range(len(tweets)):
                tweets_text.append(" ".join(w for w in nltk.wordpunct_tokenize(str(tweets["text"][i])) if w.lower() in self.words or w.isalpha()))
            tweets["text"]=tweets_text
        except Exception as e:
            print("exception: ", e)
        return tweets

    def remove_urls(self, tweets):
        print("removing urls")
        return self.remove_by_regex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))

    def remove_na(self, tweets):
        print("removing NA values")
        return tweets[tweets["text"] != "Not Available"]

    def remove_special_chars(self, tweets):  # it unrolls the hashtags to normal words
        print("removing special characters")
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)
        return tweets

    def remove_usernames(self, tweets):
        print("removing user names")
        return self.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, tweets):
        print("removing numbers")
        return self.remove_by_regex(tweets, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

    # Hashtags
    hash_regex = regex.compile(r"#(\w+)")

    def hash_repl(self,match):
        return '__HASH_' + match.group(1).upper()

    # Handels
    hndl_regex = regex.compile(r"@(\w+)")

    def hndl_repl(self,match):
        return '__HNDL'  # _'+match.group(1).upper()

    # URLs
    url_regex = regex.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

    # Spliting by word boundaries
    word_bound_regex = regex.compile(r"\W+")

    # Repeating words like hurrrryyyyyy
    rpt_regex = regex.compile(r"(.)\1{1,}", regex.IGNORECASE);

    def rpt_repl(self,match):
        return match.group(1) + match.group(1)

    def processHashtags(self, tweets):
        # print(tweets["text"])
        for index,tweet in tweets[1:].iterrows():
            tweet["text"] = regex.sub(self.hash_regex, self.hash_repl, tweet["text"])
        return tweets

    def processHandles( self, tweets):
        for index,tweet in tweets[1:].iterrows():
            tweet["text"] =  regex.sub(self.hndl_regex, self.hndl_repl, tweet["text"])
        return tweets

    def processUrls( self, tweets):
        for index,tweet in tweets[1:].iterrows():
            tweet["text"] =  regex.sub( self.url_regex, ' __URL ', tweet["text"])
        return tweets

    # def processPunctuations( self, tweets):
    #     for index,tweet in tweets[1:].iterrows():
    #         tweet["text"] =  regex.sub(self.word_bound_regex , self.punctuations_repl, tweet["text"])
    #     return tweets

    def processRepeatings( 	self, tweets):
        for index, tweet in tweets[1:].iterrows():
            tweet["text"] = regex.sub(self.rpt_regex, self.rpt_repl, tweet["text"])
        return tweets



