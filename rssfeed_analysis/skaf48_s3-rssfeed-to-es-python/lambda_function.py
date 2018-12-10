

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import boto3
import rssfeed_to_es
from textblob import TextBlob
from elasticsearch import Elasticsearch
import pandas as pd
s3 = boto3.client('s3')


"""

If you want to include a package in lambda deployment package include the package folder in the deployment package zip file

Steps that worked


1. Follow steps in https://stackoverflow.com/questions/34749806/using-moviepy-scipy-and-numpy-in-amazon-lambda

2. Once the packages are installed on ec2 instance, use  mobaxterm software to ssh into the instance and download the files. 

3. Add the files to the lambda deployment zip file

"""


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    print("bucket: ",bucket)
    print("key: ",key)
    # Getting s3 object
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
              
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
    
    # Parse s3 object content (JSON)
    try:
        s3_file_content = response['Body'].read()
        #clean trailing comma
        if s3_file_content.endswith(',\n'):
            s3_file_content = s3_file_content[:-2]
        rssfeeds_str = '[' + s3_file_content + ']'

        df = pd.read_json(rssfeeds_str, orient='records')
        # print(df.head())

        # print("dataframe size before deduplication: ", df.shape)

        df.drop_duplicates(subset=["summary"], inplace=True)
        df.sort_values(by=['timestamp_ms'], inplace=True)
        # print("dataframe size after deduplication: ", df.shape)

        rssfeeds_str = df.to_json(orient='records')
        rssfeeds = json.loads(rssfeeds_str)
   
    except Exception as e:
        print(e)
        print('Error loading json from object {} in bucket {}'.format(key, bucket))
        raise e
    
    # Load data into ES
    try:
        rssfeed_to_es.load(rssfeeds)

    except Exception as e:
        print(e)
        print('Error loading data into ElasticSearch')
        raise e    


if __name__ == '__main__':
    event = {
	    'Records': [
		    {
			    's3': {
				    'bucket': {
					    'name': 'YOUR_BUCKET'
				    },
				    'object': {
					    'key': 'YOUR_KEY'
				    }
			    }
		    }
	    ]
    }
    lambda_handler(event, None)