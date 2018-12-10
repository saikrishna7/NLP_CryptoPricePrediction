import boto3
import os
import time
import getpass
import botocore
from datetime import datetime, timedelta, date
import pprint
import pandas as pd
import config
import json
import csv

class GetTestData:

    s3_client = boto3.client('s3',
                      region_name = config.region,
                      aws_access_key_id = config.aws_key_id,
                      aws_secret_access_key = config.aws_key
                      )

    s3_resource = boto3.resource('s3',
                             region_name=config.region,
                             aws_access_key_id=config.aws_key_id,
                             aws_secret_access_key=config.aws_key
                             )

    bucket_name='es-twitter-jupyter'

    def clean_datafiles(self):
        df_ = pd.read_csv("data/test.csv") #, dtype={"timestamp": datetime, "followers_count": float, "favourites_count": float, "id": float, "screen_name": str, "text": str}
        df_.rename(index=str, columns={"created_at": "timestamp"}, inplace=True)
        print("cleaned test data:", df_.head())
        df_.drop_duplicates(subset=["text"], inplace=True)
        df_.sort_values(by=['timestamp'], inplace=True)
        df_ = df_[:-1]
        df_.to_csv("data/clean_test.csv", sep=',', encoding='utf-8', index=False)
        print("cleaned test data:", df_.head())
        print(df_.shape)
        # os.remove("data/test.csv")

    def get_time(self, hours):
        dt = datetime.utcnow() - timedelta(hours=hours)
        dt = dt.strftime("%Y/%m/%d/%H")
        return dt

    def get_list_of_files_from(self, date, duration=0):
        order_files=[]
        if duration>1:
            files_path = "twitter_firehose/twitter/raw-data/" + date + "/"
            response = GetTestData.s3_client.list_objects(
                Bucket=GetTestData.bucket_name,
                Prefix=files_path
            )
            files = []
            if 'Contents' in response.keys():
                # print("response.keys(): ", response.keys())
                for file in response["Contents"]:
                    files.append(file["Key"])
                return files
        else:
            files_path = "twitter_firehose/twitter/raw-data/" + date + "/"
            response = GetTestData.s3_client.list_objects(
                Bucket=GetTestData.bucket_name,
                Prefix=files_path
            )
            files = {}
            if 'Contents' in response.keys():
                # print("response.keys(): ", response.keys())
                for file in response["Contents"]:
                    files[file["Key"]] = file["LastModified"]

                order_files = sorted(files, reverse=True)
                if (len(response['Contents'])) > 3:
                    return order_files[0:3]
                else:
                    return order_files
            else:
                print("getting files for previous hour")
                order_files = GetTestData.get_list_of_files_from(self, GetTestData.get_time(self, 1))
                return order_files

    # First parameter in download_file() function has the file name to be downloaded.
    # Second parameter has the name of directory where the file has to be saved snd the file name itself.
    def create_empty_testcsv(self):
        print("creating empty csv")
        if os.path.isfile('data/test.csv'):
            if os.path.isfile('data/old_test.csv'):
                os.remove("data/old_test.csv")
                os.rename('data/test.csv', 'data/old_test.csv')
            print("Renaming old test data file to old_test.csv")
            time.sleep(5)
        with open('data/test.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("created_at", "followers_count", "favourites_count", "id", "screen_name", "text"))


    def main(self, duration=24):
        duration=int(duration)
        print("execution begins here")
        GetTestData.create_empty_testcsv(self)
        s3_response = GetTestData.get_list_of_files_from(self, GetTestData.get_time(self, duration), duration)

        """
        If there are no recent files in S3 bucket, download the last 3 files that are created"""
        unsorted = []
        if s3_response is None:
            from boto3 import client
            conn = client('s3')  # again assumes boto.cfg setup, assume AWS S3
            for key in conn.list_objects(Bucket=GetTestData.bucket_name)['Contents']:
                unsorted.append(key['Key'])
            # print("unsorted.append(key['Key']): ", unsorted)
            s3_response=unsorted
        """"""

        with open("data/test.csv", "a") as test:
            tweets = {}

            for item in s3_response[0:10]:
                try:
                    print("file name: ", item)
                    response = GetTestData.s3_client.get_object(Bucket=GetTestData.bucket_name, Key=item)
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

                    if unsorted is None:
                        withinhour = int((datetime.today() - timedelta(1)).timestamp())
                        print("withinhour: ", withinhour)
                        print(withinhour)
                        data_df["epoch"] = (
                                pd.to_datetime(data_df["created_at"], infer_datetime_format=True) - datetime(1970, 1,
                                                                                                             1)).dt.total_seconds()
                        # print(data_df.head())
                        print("shape before filtering: ", data_df.shape)
                        data_df = data_df[data_df['epoch'] > withinhour]
                        data_df.drop("epoch", axis=1, inplace=True)

                        print("shape after filtering: ", data_df.shape)
                    # print(data_df.head())
                    data_df.to_csv("data/test.csv", header=None, sep=',', mode='a', encoding='utf-8', index=False)

                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        print('bad json: ', response)
            GetTestData.clean_datafiles(self)
