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

# Set the username from system
system_user_name=getpass.getuser()

aws_key_id = os.environ.get('aws_access_key_id')
aws_key    = os.environ.get('aws_secret_access_key')

client = boto3.client('es')
iam = boto3.client('iam')
s3 = boto3.resource('s3')
lamb = boto3.client('lambda')

domain_name='es-twitter-jupyter'


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
	                          "Resource": "arn:aws:es:us-east-1:714861692883:domain/es-twitter-jupyter/*"
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
        # Get the es doåmain details
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
poll_until_completed(client, domain_id=domain_name)  # Can't use the cluster until it is available

# Create an S3 bucket for this application
s3.create_bucket(Bucket=domain_name)


# Create an IAM role for Firehose
# Below function will create an AWS role for performing lambda operation
def create_role(name, policies=None):
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

    # attach managed policy
    if policies is not None:
        for p in policies:
            iam.attach_role_policy(RoleName=role['RoleName'], PolicyArn=p)
    return role

# Create an IAM role to for Amazon Kinesis Firehose Full Access
# Call above function to create the role with predefined role type and access policy
role = create_role(system_user_name + '_firehose_delivery_role', 
                   policies=['arn:aws:iam::aws:policy/AmazonKinesisFirehoseFullAccess'])



lambda_name = "skaf48_s3-twitter-to-es-python"
bucket_name = "es-twitter-jupyter"
file_name = "skaf48_s3-twitter-to-es-python.zip"

# Create an IAM role to execute lambda function
# Call above function to create the role with predefined role type and access policy
role = create_role(system_user_name + '_lambda-s3-execution-role', 
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
			Role=role['Arn'],
			Handler='lambda_function.lambda_handler',
			Description='lambda function to capture twitter stream data',
			Timeout=timeout,
			MemorySize=lsize,
			Publish=True,
			Code={'S3Bucket':bucket, 'S3Key':key},
			)
		lfunc['Role'] = role
	return lfunc

lfunc = create_function(lambda_name,bucket=bucket_name, key=file_name, update=True)


# def create_function(name, zfile, lsize=512, timeout=120, update=False):
# 	""" Create, or update if exists, lambda function """
# 	# print("role:",role)

# 	with open(zfile, 'rb') as zipfile:
# 	    if name in [f['FunctionName'] for f in lamb.list_functions()['Functions']]:
# 	        if update:
# 	            print('Updating %s lambda function code' % (name))
# 	            return lamb.update_function_code(FunctionName=name, ZipFile=zipfile.read())
# 	        else:
# 	            print('Lambda function %s exists' % (name))
# 	            for f in lamb.list_functions()['Functions']:
# 	                if f['FunctionName'] == name:
# 	                    lfunc = f
# 	    else:
# 	        print('Creating %s lambda function' % (name))
# 	        lfunc = lamb.create_function(
# 	            FunctionName=name,
# 	            Runtime='python3.6',
# 	            Role=role['Arn'],
# 	            Handler='lambda_function.lambda_handler',
# 	            Description='lambda function to capture twitter stream data',
# 	            Timeout=timeout,
# 	            MemorySize=lsize,
# 	            Publish=True,
# 	            Code={'ZipFile': zipfile.read()},
# 	        )
# 	    lfunc['Role'] = role
# 	return lfunc

# file_name = 'skaf48_s3-twitter-to-es-python.zip'
# lfunc = create_function(lambda_name, file_name, update=True)




def add_lambda_permissions():
	response = lamb.add_permission(
	    FunctionName=lambda_name,
	    StatementId=time.strftime(system_user_name+"%d%m%Y%H%M%S"), # some-unique-id 
	    Action='lambda:InvokeFunction', 
	    Principal='s3.amazonaws.com',
	    SourceArn='arn:aws:s3:::'+domain_name,   # ARN of the source bucket, Change the value with your pawprint
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
	    Bucket=domain_name,
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
        'RoleARN': 'arn:aws:iam::714861692883:role/skaf48_firehose_delivery_role',
        'BucketARN': 'arn:aws:s3:::es-twitter-jupyter',
        'Prefix': stream_name+'/twitter/raw-data/'
    }
    )
    

def main(search_name):
    stream_name = search_name[0]
    client = boto3.client('firehose', region_name='us-east-1',
                          aws_access_key_id=aws_key_id,
                          aws_secret_access_key=aws_key
                          )
    try:
        create_stream(client,stream_name)
        print('Please wait...')
        time.sleep(60)
    except:
        pass

    stream_status = client.describe_delivery_stream(DeliveryStreamName=stream_name)
    if stream_status['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE':
        print("\n ==== KINESES ONLINE ====")
 
if __name__ == '__main__':
    main(sys.argv[1:])




