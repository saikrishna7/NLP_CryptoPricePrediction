from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import shutil
import os

class RemoveNoiseTweets:

    intervals = 25000
    def __init__(self, csvfile,is_testing_set):
        self.file = csvfile
        self.is_testing_set = is_testing_set

    def removenoise(self):
        for k in np.arange(0, 0):
            print("iteration: ", k, "\n"*10)
            shutil.rmtree("data/clean/")
            os.mkdir("data/clean")
            # if k==0:
            #     df = pd.read_csv(self.file)
            # else:
            if self.is_testing_set:
                df = pd.read_csv(self.file)
                RemoveNoiseTweets.intervals = 100
            else:
                df = pd.read_csv(self.file)
            print("data dimensions: ",df.shape)
            print("data columns: ",df.columns)
            if "level_0" in df.columns:
                df.drop(['level_0'],axis=1, inplace=True)
            if "index" in df.columns:
                df.drop(['index'],axis=1,inplace=True)
            text_data = pd.DataFrame(df.text)
            # print(text_data.head())
            # print(type(text_data))

            print("length of text_data: ", len(text_data))
            print("reading the file",self.file, "for cleaning")
            print("data dimensions: ", df.shape)
            for i in np.arange(0, len(text_data), step=RemoveNoiseTweets.intervals):
                temp_df = df[i:i+RemoveNoiseTweets.intervals]
                print(temp_df.head())
                temp_df.reset_index(inplace=True)
                # print("shape of temp df: ", temp_df.shape)
                documents=text_data[i:i+RemoveNoiseTweets.intervals]
                # documents.text = re.sub(r"\n","",documents.text)

                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents.text)
                print("tfidf_matrix.shape: ", tfidf_matrix.shape)

                cos_sim = np.zeros((len(documents), len(documents)))
                print("##################")
                arr = np.array([])
                max_cos = []
                indexes = []
                for index in np.arange(0, len(documents)):
                    # print("index: ", index)
                    cos_sim[index, :] = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix)
                    temp = [i for i, x in enumerate(cos_sim[index, :]) if x>0.7]
                    temp = [a for a in temp if a > index]
                    # print(index, ": ", temp,"\n")
                    if len(temp)>0:
                        indexes.append(temp)
                if len(indexes)==0:
                    return
                else:
                    dup_recs = np.unique(np.concatenate(indexes))

                print("number of duplicate docs: ", len(dup_recs))
                # print("duplicate docs: ", indexes)
                print("duplicate docs: ", dup_recs)

                print("before dropping rows: ", temp_df.shape)
                temp_df = temp_df.drop(labels = dup_recs, axis=0)
                print("after dropping similar rows: ", temp_df.shape)

                temp_df = temp_df.groupby('id').filter(lambda x: len(x) < 50)
                temp_df = temp_df.loc[~temp_df['screen_name'].isin(["Online_machine", "Mig0008", "altcoinuniverse", "BitcoinBidder",
                                                               "skylarprentice1", "sophiepmmartin9", "BlockWatcher",
                                                               "blockchainbot", "zFlashio", "Winkdexer", "BigBTCTracker",
                                                               "michaeljamiw", "ldnblockchain", "cryptoknocks", "CapitalCreator_",
                                                               "EurVirtualCoins", "r_topisto", "bitcoinnetwork3", "InvestAltcoins",
                                                               "susansreviews", "cryptobuddha1", "CryptoCriterion", "12earnmoney",
                                                               "suresh_p12", "laweddingmakeup", "Helg2012", "Monu80067960", "gaohz_",
                                                               "BroMinercom", "FactomStockPr", "skylarprentice1", "free4coin",
                                                               "help_me_plzplz_", "claytongarner5", "TeleMiner", "bitcoinkiosk_",
                                                               "birnihigo18", "CryptoBuza", "so6i79", "play_game_girl", "quyendoattt",
                                                               "coinok", "johnbitcoinss", "deuZige", "saddington", "prds71",
                                                               "zomerushintoi",
                                                               "hodlhodlnews", "WhaleStreetNews", "simulacra10", "coinlerinkrali",
                                                               "PlanetZiggurat", "dotcomlandlords", "bot", "BCash_Market",
                                                               "BitcoinMarket2",
                                                               "bitcoinest", "mclynd", "Affiliyfuture1", "bitcoins_future",
                                                               "WallStForsale", "Coin_Manager","CryptoTopCharts", "stone22stone",
                                                               "CryptoNewswire","CoinWatcherBot","btctickerbot","ksantacr_","realyoungjohn",
                                                               "CashClamber", "techpearce","mustcoins", "infocryptos","bitcoinnewsfeed",
                                                               "cryptinsight", "coffeegaram"])]
                temp_df = temp_df.loc[~temp_df['text'].
                  str.contains("Market rank is|1 bitcoin =|1 BTC Price:|FREE TOKEN|latest Cryptocurrency News|Free|bot |free bitcoin|#Earn #Bitcoin|Earn Bitcoin")]
                print("shape of data after removing tweets with count > 50 and noise tweets: ", temp_df.shape)

                temp_df.to_csv("data/clean/subset"+str(i)+".csv", sep=',', encoding='utf-8', index=False)

            filenames = []
            for (dirpath, dirnames, files) in os.walk("data/clean"):
                filenames.extend(files)
                break
            if len(filenames)==0:
                return
            os.chdir("data/clean/")
            combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
            if "index" in combined_csv.columns:
                combined_csv.drop(['index'],axis=1,inplace=True)
            combined_csv.reset_index(inplace=True, drop=True)
            os.chdir("../../")
            if self.is_testing_set:
                combined_csv.to_csv("data/clean_test.csv", index=False)
            else:
                combined_csv.to_csv("data/clean_train.csv", index=False)
            print(len(combined_csv))


    # os.chdir("data")
    # filenames=["subset0.csv","subset25000.csv","subset50000.csv","subset75000.csv","subset100000.csv","subset125000.csv","subset150000.csv","subset175000.csv","subset200000.csv","subset225000.csv","subset250000.csv","subset275000.csv",""]
#from os import walk

#filenames=[]
#for(dirpath, dirnames,files) in walk("data/clean"):
#    filenames.extend(filenames)
#    break
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
#combined_csv.to_csv("data/combined_csv1.csv", index=False)

