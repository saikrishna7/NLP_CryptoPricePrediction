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


class GetTrainData:

    s3_client = boto3.client('s3',
                      region_name = config.region,
                      aws_access_key_id = config.aws_key_id,
                      aws_secret_access_key = config.aws_key
                      )

    bucket_name='es-rssfeed-jupyter'

    paginator = s3_client.get_paginator('list_objects')

    header = ["author", "summary", "timestamp_ms", "title"]

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
        df_ = pd.read_csv("data/"+filetype+".csv", dtype={"author": str, "title": str,
                                                          "timestamp_ms": str, "summary": str})
        df_ = df_[1:]
        try:
            df_.drop_duplicates(subset=["summary"], inplace=True)
            df_ = df_[df_["summary"].str.contains("submitted by") == False]
            df_ = df_[~df_['summary'].isnull()]

        except Exception as e:
            print("error occured: ", e)

        df_.sort_values(by=['timestamp_ms'], inplace=True)
        df_ = df_[:-1]
        df_.reset_index(drop=True)
        df_.to_csv("data/clean_" + filetype + ".csv", sep=',', encoding='utf-8', index=False)
        os.remove("data/" + filetype + ".csv")

    def create_empty_traincsv(self):
        with open('data/train.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("author", "summary", "timestamp_ms", "title"))
        with open('data/test.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("author", "summary", "timestamp_ms", "title"))

    def main(self):
        GetTrainData.create_empty_traincsv(self)
        for months in GetTrainData.paginator.paginate(Bucket=GetTrainData.bucket_name, Delimiter='/', Prefix="rssfeed_firehose/rssfeed/raw-data/2018/"):
            print("months:", months.get('CommonPrefixes'))
            for month in months.get('CommonPrefixes'):
                print("month: ", month.get('Prefix'))
                for result in GetTrainData.paginator.paginate(Bucket=GetTrainData.bucket_name, Delimiter='/', Prefix=month.get('Prefix')):
                    print("list of days in current month:", result.get('CommonPrefixes'))
                    for day in result.get('CommonPrefixes'):
                        print("Getting files for the day: ", day)
                        s3_response = GetTrainData.get_list_of_files_from(self, files_path=day.get('Prefix'))
                        with open("data/train.csv", "a") as outfile:
                            with open("data/test.csv", "a") as test:
                                for item in s3_response['Contents']:
                                    try:
                                        response = GetTrainData.s3_client.get_object(Bucket=GetTrainData.bucket_name, Key=item["Key"])
                                        # data = response["Body"].read().decode('utf-8')
                                        print("file name: ", item["Key"])
                                        ######################################
                                        s3_file_content = response['Body'].read().decode('utf-8')

                                        # clean trailing comma
                                        if s3_file_content.endswith(',\n'):
                                            s3_file_content = s3_file_content[:-2]
                                        rssfeeds_str = '[' + s3_file_content + ']'

                                        df = pd.read_json(rssfeeds_str, orient='records')

                                        print("dataframe size before deduplication: ", df.shape)

                                        df.drop_duplicates(subset=["summary"], inplace=True)
                                        df.sort_values(by=['timestamp_ms'], inplace=True)
                                        if len(df.columns) != 4:
                                            df.to_csv("data/train.csv", header=True, columns=GetTrainData.header, sep=',', mode='a',
                                                      encoding='utf-8',
                                                      index=False)
                                        else:
                                            df.to_csv("data/train.csv", header=True, sep=',', mode='a', encoding='utf-8',
                                                      index=False)
                                        print("dataframe size after deduplication: ", df.shape)


                                    except botocore.exceptions.ClientError as e:
                                        if e.response['Error']['Code'] == "404":
                                            print("The object does not exist.")
                                        else:
                                            print('bad json: ', response)

        GetTrainData.clean_datafiles(self, "train")
        GetTrainData.clean_datafiles(self, "test")
