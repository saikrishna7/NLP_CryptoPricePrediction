import feedparser
from bs4 import BeautifulSoup
from bs4.element import Comment
import json
import time
import os
import uuid
import boto3
import botocore
import random
import time
import json
import datetime
import getpass
from moto import mock_kinesis
from botocore.exceptions import ClientError
import os
import sys
import config

# Set the username from system
system_user_name=getpass.getuser()

s3 = boto3.resource('s3',
                  region_name = config.region,
                  aws_access_key_id = config.aws_key_id,
                  aws_secret_access_key = config.aws_key
                  )

# aws_key_id = os.environ.get('aws_access_key_id')
# aws_key    = os.environ.get('aws_secret_access_key')

iam = boto3.client('iam',
                  region_name = config.region,
                  aws_access_key_id = config.aws_key_id,
                  aws_secret_access_key = config.aws_key)

lamb = boto3.client('lambda',
                  region_name = config.region,
                  aws_access_key_id = config.aws_key_id,
                  aws_secret_access_key = config.aws_key)

client = boto3.client('es',
                region_name = config.region,
                  aws_access_key_id = config.aws_key_id,
                  aws_secret_access_key = config.aws_key)

# Create an S3 bucket for this application
# s3.create_bucket(Bucket=bucket_name)

domain_name='es-rssfeed-jupyter'
bucket_name='es-rssfeed-jupyter'

# Create a new Elasticsearch domain
def create_elastic_search_domain():
    response = client.create_elasticsearch_domain(
        DomainName=domain_name,
        ElasticsearchVersion='5.5',
        ElasticsearchClusterConfig={
            'InstanceType': 'm3.medium.elasticsearch',
            'InstanceCount': 2,
            'DedicatedMasterEnabled': False,
            'ZoneAwarenessEnabled': False
        },
        EBSOptions={
            'EBSEnabled': True,
            'VolumeType': 'standard',
            'VolumeSize': 10
        },
        AccessPolicies="""{
                          "Version": "2012-10-17",
                          "Statement": [
                            {
                              "Effect": "Allow",
                              "Principal": {
                                "AWS": "*"
                              },
                              "Action": "es:*",
                              "Resource": "arn:aws:es:us-east-1:714861692883:domain/es-rssfeed-jupyter/*"
                            }
                          ]
                        }""",
        
        AdvancedOptions={
            'rest.action.multi.allow_explicit_index': 'True', 
            'indices.fielddata.cache.size': '40',
            'indices.query.bool.max_clause_count': '1024'
        }    
    )

# create_elastic_search_domain()

def poll_until_completed(client, domain_id):
    delay = 2
    while True:
        # Get the es domain details
        es_domain = client.describe_elasticsearch_domains(DomainNames=[domain_id,])
        # Get the current status of cluster
        status = es_domain['DomainStatusList'][0]['Processing']
        # Get current system time 
        now = str(datetime.datetime.now().time())
        
        # Below Condition keeps checking if the cluster is in available state or in final-snapshot. If yes, then break the loop
        if status == False:
            print("Domain is active")
            break
        else:
            # Print the message about domain status at current time
            print("domain %s is still processing at %s" % (domain_id, now))

        # If the cluster status is not in available or final-snapshot then wait for time and go through one more iteration.
        delay *= random.uniform(1.1, 2.0)
        time.sleep(delay)
    return True


# Call the above function to make sure elastic search domain is active
# poll_until_completed(client, domain_id=domain_name)  # Can't use the cluster until it is available




# Create an IAM role for Firehose
# Below function will create an AWS role for performing lambda operation
def create_firehose_role(name, policies=None):
    """ Create a role with an optional inline policy """
    policydoc = {
                  "Version": "2012-10-17",
                  "Statement": {
                                  "Effect": "Allow",
                                  "Principal": {"Service": "firehose.amazonaws.com"},
                                  "Action": "sts:AssumeRole"
                            } 
                }
    roles = [r['RoleName'] for r in iam.list_roles()['Roles']]
    if name in roles:
        print('IAM role %s exists' % (name))
        role = iam.get_role(RoleName=name)['Role']
    else:
        print('Creating IAM role %s' % (name))
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(policydoc))['Role']
        # Give some time for the role to be created. 
        time.sleep(60)

    # attach managed policy
    if policies is not None:
        for p in policies:
            iam.attach_role_policy(RoleName=role['RoleName'], PolicyArn=p)
    return role

# Create an IAM role to for Amazon Kinesis Firehose Full Access
# Call above function to create the role with predefined role type and access policy

role = create_firehose_role(system_user_name + '_firehose_rssfeed_delivery_role', 
                   policies=['arn:aws:iam::aws:policy/AmazonKinesisFirehoseFullAccess'])



lambda_name = "skaf48_s3-rssfeed-to-es-python"
file_name = "skaf48_s3-rssfeed-to-es-python.zip"


def create_lambda_role(name, policies=None):
    """ Create a role with an optional inline policy """
    policydoc = {
                  "Version": "2012-10-17",
                  "Statement": {
                                  "Effect": "Allow",
                                  "Principal": {"Service": "lambda.amazonaws.com"},
                                  "Action": "sts:AssumeRole"
                            } 
                }
    roles = [r['RoleName'] for r in iam.list_roles()['Roles']]
    if name in roles:
        print('IAM role %s exists' % (name))
        role = iam.get_role(RoleName=name)['Role']
    else:
        print('Creating IAM role %s' % (name))
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(policydoc))['Role']
        # Give some time for the role to be created. 
        time.sleep(60)
    # attach managed policy
    if policies is not None:
        for p in policies:
            iam.attach_role_policy(RoleName=role['RoleName'], PolicyArn=p)
    return role



# Create an IAM role to execute lambda function
# Call above function to create the role with predefined role type and access policy

role = create_lambda_role(system_user_name + '_lambda-s3-rssfeed-execution-role', 
                   policies=['arn:aws:iam::aws:policy/AWSLambdaExecute'])


def delete_function(function_name):
    if function_name in [f['FunctionName'] for f in lamb.list_functions()['Functions']]:
        print("function_name: ",function_name)
        response = lamb.delete_function(FunctionName=function_name)
    else:
        print("Lambda function %s doesn't exist" % (function_name))


delete_function_resp = delete_function(lambda_name)

def create_function(name, bucket, key, lsize=128, timeout=120, update=False):
    if name in [f['FunctionName'] for f in lamb.list_functions()['Functions']]:
        if update:
            print('Updating %s lambda function code' % (name))
            return lamb.update_function_code(FunctionName=name, S3Bucket=bucket, S3Key=key)
        else:
            print('Lambda function %s exists' % (name))
            for f in lamb.list_functions()['Functions']:
                if f['FunctionName'] == name:
                    lfunc = f
    else:
        print('Creating %s lambda function' % (name))
        lfunc = lamb.create_function(
            FunctionName=name,
            Runtime='python2.7',
            Role= role['Arn'], # "arn:aws:iam::714861692883:role/service-role/skaf48_lambda-s3-rssfeed-execution-role",
            Handler='lambda_function.lambda_handler',
            Description='lambda function to capture rssfeed data',
            Timeout=timeout,
            MemorySize=lsize,
            Publish=True,
            Code={'S3Bucket':bucket, 'S3Key':key},
            )
        lfunc['Role'] = role
    return lfunc

lfunc = create_function(lambda_name,bucket=bucket_name, key=file_name, update=True)


def add_lambda_permissions():
    response = lamb.add_permission(
        FunctionName=lambda_name,
        StatementId=time.strftime(system_user_name+"%d%m%Y%H%M%S"), # some-unique-id 
        Action='lambda:InvokeFunction', 
        Principal='s3.amazonaws.com',
        SourceArn='arn:aws:s3:::'+bucket_name,   # ARN of the source bucket, Change the value with your pawprint
        SourceAccount='714861692883'   # bucket-owner-account-id
    )
add_lambda_permissions()


# Get the lambda function details. We need the lambda arn in next function call. 

lambda_details = lamb.get_function(
    FunctionName=lambda_name
)


def s3_bucket_notifications_configuration():
    s3_client = boto3.client('s3')
    response = s3_client.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration={
            'LambdaFunctionConfigurations': [
                {
                    'LambdaFunctionArn': lambda_details["Configuration"]["FunctionArn"],
                    'Events': ['s3:ObjectCreated:*']
                }
            ]
        }
    )

s3_bucket_notifications_configuration()


def create_stream(client, stream_name):
    print("creating Kinesis stream",stream_name)
    return client.create_delivery_stream(
        DeliveryStreamName=stream_name,
        S3DestinationConfiguration={
        'RoleARN': 'arn:aws:iam::714861692883:role/skaf48_firehose_rssfeed_delivery_role',
        'BucketARN': 'arn:aws:s3:::es-rssfeed-jupyter',
        'Prefix': stream_name+'/rssfeed/raw-data/',
        'BufferingHints': {
            'SizeInMBs': 20,
            'IntervalInSeconds': 900
        },
        'CloudWatchLoggingOptions': {
            'Enabled': True,
            'LogGroupName': 'kinesisfirehose',
            'LogStreamName': 'rssfeed_firehose'
        }
    }
    )


# Functions from: https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def poll_until_count_completed(max_count,firehose_client, stream_name, url):
    delay = 2
    count=0
    local_count=0
    data=[]
    while True:
        feed = feedparser.parse( url )

        if (feed['bozo'] == 1):
            print("Error Reading/Parsing Feed XML Data")
        else:
            for item in feed["items"]:
                ele = {"title": item[ "title" ], "author" : item[ "author" ], "timestamp_ms" : item["date"], "summary" : text_from_html(item["summary"])}
                data.append(ele)
            try:
                firehose_client.put_record(DeliveryStreamName=stream_name,
                                        Record={'Data': json.dumps(data) + '\n'})

            except Exception as e:
                print("*"*50)
                print("\n\n\n\n\n\n\n")
                print("Failed Kinesis Put Record {}".format(str(e)))
                print("\n\n\n\n\n\n\n")
                print("*"*50)
                

            for item in feed[ "items" ]:
                dttm = item[ "date" ]
                title = item[ "title" ]
                summary_text = text_from_html(item[ "summary" ])
                link = item[ "link" ]

                print("====================")
                print("Title: {} ({})\nTimestamp: {}".format(title,link,dttm))
                print("--------------------\nSummary:\n{}".format(summary_text))

        # local_count=local_count+1
        # if local_count==1000:
        #     filename = str(uuid.uuid4())
        #     s3.Bucket(bucket_name).put_object(Key=file_name, Body=json.dump(data))
        #     data=[]
        #     local_count=0
        count+=1
        if(count==max_count):
            break



def poll_until_time_completed(time_limit,firehose_client, stream_name, url):
    start_time = time.time() #grabs the system time
    count=0
    data=[]
    while True:
        feed = feedparser.parse( url )
        if (feed['bozo'] == 1):
            print("Error Reading/Parsing Feed XML Data")
        else:
            if (time.time() - start_time) < time_limit:
                feed = feedparser.parse(url)

                for item in feed["items"]:
                    ele = {"title": item[ "title" ], "author" : item[ "author" ], "timestamp_ms" : item["date"], "summary" : text_from_html(item["summary"])}
                    try:
                        firehose_client.put_record(DeliveryStreamName=stream_name,
                                            Record={'Data': json.dumps(ele) + ',\n'})

                    except Exception as e:
                        print("Failed Kinesis Put Record {}".format(str(e)))

                for item in feed["items"]:
                    dttm = item["date"]
                    title = item["title"]
                    summary_text = text_from_html(item["summary"])
                    link = item["link"]

                    print("====================")
                    print("--------------------\n Summary:\n{}".format(summary_text))
                # count+=1
                # if count==1000:
                #     filename = str(uuid.uuid4())
                #     s3.Bucket(bucket_name).put_object(Key=file_name, Body=json.dump(data))
                #     count=0
            else:
                break
        


def main(search_name):
    stream_name = search_name[0]

    firehose_client = boto3.client('firehose', region_name='us-east-1',
                          aws_access_key_id=config.aws_key_id,
                          aws_secret_access_key=config.aws_key
                          )
    try:
        create_stream(firehose_client,stream_name)

    except:
        pass

    print('Please wait until the stream is created...')
    time.sleep(60)
    stream_status = firehose_client.describe_delivery_stream(DeliveryStreamName=stream_name)
    if stream_status['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE':
        print("\n ==== KINESES ONLINE ====")

    a_reddit_rss_url = 'https://www.reddit.com/r/Bitcoin/new/.rss?sort=new'
    # max_count=3
    # poll_until_count_completed(max_count,firehose_client, a_reddit_rss_url)  # Can't use it until it's COMPLETED

    time_limit=100000000

    poll_until_time_completed(time_limit,firehose_client, stream_name, a_reddit_rss_url)  # Can't use it until it's COMPLETED
 
if __name__ == '__main__':
    main(sys.argv[1:])
