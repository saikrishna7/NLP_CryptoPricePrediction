import re as regex
import nltk
import pandas as pd


class DataCleaner:
    def __init__(self, is_testing):
        self.is_testing = is_testing

    words = set(nltk.corpus.words.words())

    def iterate(self):
        for cleanup_method in [self.remove_duplicates,
                               self.remove_noneng_rssfeeds,
                               self.remove_urls,
                               self.clean_text,
                               self.remove_special_chars,
                               self.processHashtags,
                               self.processHandles,
                               self.processUrls,
                               self.processRepeatings
                                ]:
            yield cleanup_method


    @staticmethod
    def remove_by_regex(rssfeeds, regexp):
        rssfeeds.loc[:, "summary"].replace(regexp, "", inplace=True)
        return rssfeeds

    def remove_duplicates(self, tweets):
        print("removing duplicate tweets")
        tweets.drop_duplicates(subset=["summary"], inplace=True)
        print("Number of tweets after removing duplicates: ",tweets.shape)
        return tweets

    def clean_text(self, rssfeeds):
        print("removing unwanted text")
        print(rssfeeds.head())
        for index,rssfeed in rssfeeds[1:].iterrows():
            rssfeed["author"] = regex.sub('\/u/', '', rssfeed.author)
            rssfeed.title = regex.sub('\/u/.*$', '', rssfeed.title)
            rssfeed.summary = regex.sub('\/u/.*$', '', rssfeed.summary)
            if "submitted by" in rssfeed["summary"]:
                rssfeed["summary"] = rssfeed["title"]
            if "submitted by" in rssfeed["title"]:
                rssfeed["title"] = rssfeed["author"]
        print("after cleaning data: ", rssfeeds.head())
        print("Number of rssfeeds after removing duplicates: ", rssfeeds.shape)
        return rssfeeds


    def remove_noneng_rssfeeds(self, rssfeeds):
        rssfeeds_text=[]
        print("removing non english rssfeeds")
        print("filtered rssfeed")

        try:
            for i in range(len(rssfeeds)):
                rssfeeds_text.append(" ".join(w for w in nltk.wordpunct_tokenize(str(rssfeeds["summary"][i])) if w.lower() in self.words or w.isalpha()))
                rssfeeds["summary"]=rssfeeds_text
        except Exception as e:
            print("exception: ",e)
        return rssfeeds

    def remove_urls(self, rssfeeds):
        print("removing urls")
        return self.remove_by_regex(rssfeeds, regex.compile(r"http.?://[^\s]+[\s]?"))

    # def remove_na(self, rssfeeds):
    #     print("removing NA values")
    #     return rssfeeds[rssfeeds["summary"] != "Not Available"]

    def remove_special_chars(self, rssfeeds):  # it unrolls the hashtags to normal words
        print("removing special characters")
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            rssfeeds.loc[:, "summary"].replace(remove, "", inplace=True)
        return rssfeeds

    # def process_usernames(self, rssfeeds):
    #     print("processing user names")
    #     rssfeeds.author = rssfeeds.author[3:]
    #     # print(rssfeeds.head())
    #     return rssfeeds

    # def remove_numbers(self, rssfeeds):
    #     print("removing numbers")
    #     return self.remove_by_regex(rssfeeds, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

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

    def processHashtags(self, rssfeeds):
        # print("head: ", rssfeeds.head())
        for index,rssfeed in rssfeeds[1:].iterrows():
            rssfeed["summary"] = regex.sub(self.hash_regex, self.hash_repl, rssfeed["summary"])
        return rssfeeds

    def processHandles( self, rssfeeds):
        for index,rssfeed in rssfeeds[1:].iterrows():
            rssfeed["summary"] =  regex.sub(self.hndl_regex, self.hndl_repl, rssfeed["summary"])
        return rssfeeds

    def processUrls( self, rssfeeds):
        for index,rssfeed in rssfeeds[1:].iterrows():
            rssfeed["summary"] =  regex.sub( self.url_regex, ' __URL ', rssfeed["summary"])
        return rssfeeds


    def processRepeatings( 	self, rssfeeds):
        for index, rssfeed in rssfeeds[1:].iterrows():
            rssfeed["summary"] = regex.sub(self.rpt_regex, self.rpt_repl, rssfeed["summary"])
            # print("rssfeeds columns in process repearing: \n******************\n", rssfeeds.columns, "\n******************\n")
            # print(rssfeeds.head())
        return rssfeeds



