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

class GetDashboardTestData:

    s3_client = boto3.client('s3',
                      region_name = config.region,
                      aws_access_key_id = config.aws_key_id,
                      aws_secret_access_key = config.aws_key
                      )

    bucket_name='es-rssfeed-jupyter'
    header = ["author", "title", "timestamp_ms", "summary"]

    def clean_datafiles(self):
        df_ = pd.read_csv("data/test.csv") #, dtype={"timestamp": datetime, "followers_count": float, "favourites_count": float, "id": float, "screen_name": str, "text": str}
        df_.drop_duplicates(subset=["summary"], inplace=True)
        df_.sort_values(by=['timestamp_ms'], inplace=True)
        df_ = df_[:-1]
        df_.to_csv("data/clean_test.csv", sep=',', encoding='utf-8', index=False)
        os.remove("data/test.csv")

    def get_time(self, hours):
        dt = datetime.utcnow() - timedelta(hours=hours)
        dt = dt.strftime("%Y/%m/%d/%H")
        return dt

    def get_list_of_files_from(self, hour=0):
        order_files=[]
        files_path = "rssfeed_firehose/rssfeed/raw-data/" + hour + "/"
        # get_last_modified = lambda obj: int(obj['LastModified'].strftime('%S'))
        response = GetTestData.s3_client.list_objects(
            Bucket=GetTestData.bucket_name,
            Prefix=files_path
        )
        files = {}
        # print(response)
        # if(response['Contents']==None):
        if 'Contents' in response.keys():
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
        if os.path.exists('data/test.csv'):
            if os.path.exists('data/old_test.csv'):
                os.remove("data/old_test.csv")
                os.rename('data/test.csv', 'data/old_test.csv')
            print("Renaming old test data file to old_test.csv")
            time.sleep(5)
        with open('data/test.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("author", "summary", "timestamp_ms", "title"))



    def main(self):
        print("execution begins here")
        GetTestData.create_empty_testcsv(self)
        s3_response = GetTestData.get_list_of_files_from(self, GetTestData.get_time(self, 0))
        print(s3_response)
        with open("data/test.csv", "a") as test:
            rssfeeds = {}
            for item in s3_response:
                try:
                    response = GetTestData.s3_client.get_object(Bucket=GetTestData.bucket_name, Key=item)
                    # data = response["Body"].read().decode('utf-8')
                    print("file name: ", item)
                    ######################################
                    s3_file_content = response['Body'].read().decode('utf-8')

                    # clean trailing comma
                    if s3_file_content.endswith(',\n'):
                        s3_file_content = s3_file_content[:-2]
                    rssfeeds_str = '[' + s3_file_content + ']'

                    df = pd.read_json(rssfeeds_str, orient='records')
                    print(df.head())
                    # now, load it into pandas
                    # data_df = pd.io.json.json_normalize(rssfeeds)
                    withinhour = int((datetime.today() - timedelta(1)).timestamp())
                    print(withinhour)
                    df["epoch"] = (
                                pd.to_datetime(df["timestamp_ms"], infer_datetime_format=True) - datetime(1970, 1,
                                                                                                             1)).dt.total_seconds()
                    print("shape before filtering: ", df.shape)
                    df = df[df['epoch'] > withinhour]
                    print("shape after filtering: ", df.shape)
                    df.drop(['epoch'], axis=1, inplace=True)
                    # print(df.head())
                    print("dataframe size before deduplication: ", df.shape)

                    df.drop_duplicates(subset=["summary"], inplace=True)
                    df.sort_values(by=['timestamp_ms'], inplace=True)
                    if len(df.columns) != 4:
                        print(df.columns)
                        print(df.head())
                        df.to_csv("data/test.csv", header=True, columns=GetTestData.header, sep=',', mode='a', encoding='utf-8',
                                  index=False)
                    else:
                        df.to_csv("data/test.csv", header=True, sep=',', mode='a', encoding='utf-8', index=False)
                    print("dataframe size after deduplication: ", df.shape)

                    # data_df.to_csv("data/test.csv", header=None, sep=',', mode='a', encoding='utf-8', index=False)

                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        print('bad json: ', response)
        GetTestData.clean_datafiles(self)


