import boto3
import os
import time
import getpass
import botocore
from datetime import datetime, timedelta, date
import pprint
import pandas as pd
import json
import csv
import config
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil
class GetTrainData:

    s3_client = boto3.client('s3',
                      region_name = config.region,
                      aws_access_key_id = config.aws_key_id,
                      aws_secret_access_key = config.aws_key
                      )

    bucket_name='es-twitter-jupyter'

    paginator = s3_client.get_paginator('list_objects')


    def datetime_handler(self, x):
        if isinstance(x, datetime):
            return x.isoformat()
        raise TypeError("Unknown type: ",x)

    def json_serial(self, obj):
        """JSON serializer for objects not serializable by default json code"""

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError ("Type %s not serializable" % type(obj))


    def get_list_of_files_from(self, files_path):
        print("returning files from path: ", files_path)
        response = GetTrainData.s3_client.list_objects(
            Bucket=GetTrainData.bucket_name,
            Prefix=files_path
        )

        if(response['Contents']==None):
            print("There are no files to be downloaded")
            return False
        return response


    def get_current_hour_datetime(self):
        dt=datetime.utcnow() - timedelta(hours = 1)
        dt = dt.strftime("%Y-%m-%d-%H")
        return dt


    def clean_datafiles(self, filetype):
        df_ = pd.read_csv("data/"+filetype+".csv", dtype={"timestamp": str,
                                                          "followers_count": float, "favourites_count": float,
                                                          "id": float, "screen_name": str, "text": str})
        print(df_.head())
        df_.drop_duplicates(subset=["text"], inplace=True)
        df_.sort_values(by=['timestamp'], inplace=True)
        df_ = df_[:-1]
        df_.to_csv("data/clean_"+filetype+".csv", sep=',', encoding='utf-8', index=False)
        os.remove("data/"+filetype+".csv")



    def create_empty_datafiles(self):
        with open('data/train.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("timestamp", "followers_count", "favourites_count", "id", "screen_name", "text"))
        with open('data/test.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("timestamp", "followers_count", "favourites_count", "id", "screen_name", "text"))


    def main(self):
        if os.path.isfile("data/train.csv"):
            GetTrainData.clean_datafiles(self, "train")
            GetTrainData.clean_datafiles(self, "test")
        else:
            GetTrainData.create_empty_datafiles(self)
            for months in GetTrainData.paginator.paginate(Bucket=GetTrainData.bucket_name, Delimiter='/', Prefix="twitter_firehose/twitter/raw-data/2018/"):
                print("months:", months.get('CommonPrefixes'))
                for month in months.get('CommonPrefixes'):
                    print("month: ", month.get('Prefix'))
                    for result in GetTrainData.paginator.paginate(Bucket=GetTrainData.bucket_name, Delimiter='/', Prefix=month.get('Prefix')):
                        print("list of days in current month:", result.get('CommonPrefixes'))
                        for day in result.get('CommonPrefixes'):
                            print("Getting files for the day: ", day)
                            s3_response = GetTrainData.get_list_of_files_from(self, files_path = day.get('Prefix'))
                            with open("data/train.csv", "a") as outfile:
                                with open("data/test.csv", "a") as test:
                                    tweets = {}
                                    for item in s3_response['Contents']:
                                        try:
                                            response = GetTrainData.s3_client.get_object(Bucket=GetTrainData.bucket_name, Key=item['Key'])
                                            data = response["Body"].read().decode('utf-8')

                                            # clean trailing comma
                                            if data.endswith(',\n'):
                                                data = data[:-2]

                                            # each element of 'data' is an individual JSON object.
                                            # i want to convert it into an *array* of JSON objects
                                            # which, in and of itself, is one large JSON object
                                            # basically... add square brackets to the beginning
                                            # and end, and have all the individual business JSON objects
                                            # separated by a comma
                                            data_json_str = "[" + data + "]"
                                            tweets = json.loads(data_json_str)  # , default=json_serial

                                            # now, load it into pandas
                                            data_df = pd.io.json.json_normalize(tweets)
                                            current_hour = GetTrainData.get_current_hour_datetime(self)
                                            if current_hour in item["Key"]:
                                                print(item['Key'])
                                                print(current_hour)
                                                data_df.to_csv("data/test.csv", header=None, sep=',', mode='a', encoding='utf-8',
                                                               index=False)
                                            else:
                                                data_df.to_csv("data/train.csv", header=None, sep=',', mode='a', encoding='utf-8')


                                        except botocore.exceptions.ClientError as e:
                                            if e.response['Error']['Code'] == "404":
                                                print("The object does not exist.")
                                            else:
                                                print('bad json: ', response)

            GetTrainData.clean_datafiles(self, "train")
            GetTrainData.clean_datafiles(self, "test")




    # def remove_duplicate_tweets(self, filetype):
    #     for i in np.arange(0, 4):
    #         os.chdir("../../")
    #         shutil.rmtree("data/clean/")
    #         os.mkdir("data/clean")
    #         if i == 0:
    #             df = pd.read_csv("data/clean_train.csv")
    #         else:
    #             df = pd.read_csv("data/clean_train"+i-1+".csv")
    #         if "level_0" in df.columns:
    #             df.drop(['level_0'], axis=1, inplace=True)
    #         text_data = pd.DataFrame(df.text)
    #         print(text_data.head())
    #         print(type(text_data))
    #
    #         print("length of text_data: ", len(text_data))
    #         for i in np.arange(0, len(text_data), step=25000):
    #             temp_df = df[i:i + 25000]
    #             print(temp_df.head())
    #             temp_df.reset_index(inplace=True)
    #             # print("shape of temp df: ", temp_df.shape)
    #             documents = text_data[i:i + 25000]
    #             # documents.text = re.sub(r"\n","",documents.text)
    #
    #             tfidf_vectorizer = TfidfVectorizer()
    #             tfidf_matrix = tfidf_vectorizer.fit_transform(documents.text)
    #             print("tfidf_matrix.shape: ", tfidf_matrix.shape)
    #
    #             cos_sim = np.zeros((len(documents), len(documents)))
    #             print("##################")
    #             arr = np.array([])
    #             max_cos = []
    #             indexes = []
    #             for index in np.arange(0, len(documents)):
    #                 # print("index: ", index)
    #                 cos_sim[index, :] = cosine_similarity(tfidf_matrix[index:index + 1], tfidf_matrix)
    #                 temp = [i for i, x in enumerate(cos_sim[index, :]) if x > 0.7]
    #                 temp = [a for a in temp if a > index]
    #                 # print(index, ": ", temp,"\n")
    #                 if len(temp) > 0:
    #                     indexes.append(temp)
    #
    #             dup_recs = np.unique(np.concatenate(indexes))
    #
    #             print("number of duplicate docs: ", len(dup_recs))
    #             # print("duplicate docs: ", indexes)
    #             print("duplicate docs: ", dup_recs)
    #
    #             print("before dropping rows: ", temp_df.shape)
    #             temp_df = temp_df.drop(labels=dup_recs, axis=0)
    #             print("after dropping similar rows: ", temp_df.shape)
    #
    #             temp_df = temp_df.groupby('id').filter(lambda x: len(x) < 50)
    #             temp_df = temp_df.loc[
    #                 ~temp_df['screen_name'].isin(["Online_machine", "Mig0008", "altcoinuniverse", "BitcoinBidder",
    #                                               "skylarprentice1", "sophiepmmartin9", "BlockWatcher",
    #                                               "blockchainbot", "zFlashio", "Winkdexer", "BigBTCTracker",
    #                                               "michaeljamiw", "ldnblockchain", "cryptoknocks", "CapitalCreator_",
    #                                               "EurVirtualCoins", "r_topisto", "bitcoinnetwork3", "InvestAltcoins",
    #                                               "susansreviews", "cryptobuddha1", "CryptoCriterion", "12earnmoney",
    #                                               "suresh_p12", "laweddingmakeup", "Helg2012", "Monu80067960", "gaohz_",
    #                                               "BroMinercom", "FactomStockPr", "skylarprentice1", "free4coin",
    #                                               "help_me_plzplz_", "claytongarner5", "TeleMiner", "bitcoinkiosk_",
    #                                               "birnihigo18", "CryptoBuza", "so6i79", "play_game_girl",
    #                                               "quyendoattt",
    #                                               "coinok", "johnbitcoinss", "deuZige", "saddington", "prds71",
    #                                               "zomerushintoi",
    #                                               "hodlhodlnews", "WhaleStreetNews", "simulacra10", "coinlerinkrali",
    #                                               "PlanetZiggurat", "dotcomlandlords", "bot", "BCash_Market",
    #                                               "BitcoinMarket2",
    #                                               "bitcoinest", "mclynd", "Affiliyfuture1", "bitcoins_future",
    #                                               "WallStForsale", "Coin_Manager", "CryptoTopCharts", "stone22stone",
    #                                               "CryptoNewswire", "CoinWatcherBot", "btctickerbot", "ksantacr_",
    #                                               "realyoungjohn",
    #                                               "CashClamber", "techpearce", "mustcoins", "infocryptos",
    #                                               "bitcoinnewsfeed",
    #                                               "cryptinsight", "coffeegaram"])]
    #             temp_df = temp_df.loc[~temp_df['text'].
    #                 str.contains(
    #                 "Market rank is|1 bitcoin =|1 BTC Price:|FREE TOKEN|latest Cryptocurrency News|Free|bot |free bitcoin|#Earn #Bitcoin|Earn Bitcoin")]
    #             print("shape of data after removing tweets with count > 50 and noise tweets: ", temp_df.shape)
    #
    #             temp_df.to_csv("data/clean/subset" + str(i) + ".csv", sep=',', encoding='utf-8', index=False)
    #
    #         filenames = []
    #         for (dirpath, dirnames, files) in os.walk("data/clean"):
    #             filenames.extend(files)
    #             break
    #         os.chdir("data/clean/")
    #         combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
    #         combined_csv.to_csv("../../data/clean_train" + i + ".csv", index=False)
    #         print(len(combined_csv))