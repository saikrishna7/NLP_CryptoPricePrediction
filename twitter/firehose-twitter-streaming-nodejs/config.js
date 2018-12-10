/***
Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
***/

'use strict';

var config = module.exports = {
 firehose : {
  DeliveryStreamName: 'twitter_firehose', /* required */
  S3DestinationConfiguration: {
    BucketARN: 'arn:aws:s3:::es-twitter-jupyter>', /* required if stream not exists */
    RoleARN: 'arn:aws:iam::714861692883:role/skaf48_firehose_delivery_role', /* required if stream not exists */
    BufferingHints: {
      IntervalInSeconds: 900,
      SizeInMBs: 10
    },
    CompressionFormat: 'UNCOMPRESSED', /* 'UNCOMPRESSED | GZIP | ZIP | Snappy' */
    EncryptionConfiguration: {
      NoEncryptionConfig: 'NoEncryption'
    },
    Prefix: 'twitter/raw_data'  /* if stream not exists. example: twitter/raw-data/  */
  }
  },
  twitter: {
      consumer_key: 'KZT7UkCSyLhVO18Wqx6OJISDY',
      consumer_secret: 'X6hfBxJZz3jLqo8VeX451d7zW8u8v6yDqpiWTUWoq7hnGQTrp2',
      access_token: '908803963557941248-sRHYClIfMteyPMnwF4hWkARyuHNkJRT',
      access_token_secret: 'FgGi0GshGh8Xbi0Tmkbks0G4Jvd20J5tTThCLJzxd0UVB'
 },

 // screen_names:['BTCTN','coindesk','Cointelegraph','officialmcafee'],
 track_words:['Bitcoin','crypto','cryptocurrency','blockchain','SmartContracts'],
 // screen_names:['BTCTN','BitcoinTre','BitcoinMagazine','coindesk','BTCFoundation','GDAX','ethereumproject','VitalikButerin','Cointelegraph','iamjosephyoung','BittrexExchange','binance_2017', 'officialmcafee', 'Ripple', 'BITCOlNCASH','CardanoStiftung','litecoin','LiteCoinNews','LTCFoundation','NEO_Blockchain','NEOnewstoday','StellarOrg','EOS_io','NEMofficial','iotatokennews','Dashpay','hitbtc','monerocurrency','justinsuntron','LiskHQ','vechainofficial','eth_classic','QtumOfficial','helloiconworld'],
 // track_words:['#Bitcoin','crypto', 'cryptocurrency', 'blockchain','#XRP','#Ethereum','#BCH','#Cardano','#Ada','#Litecoin','#LTC','#Crypto','$NEO','$XLM','#XLM','#EOS','$EOS','#XEM','XEM','#IOTA','#DigitalCash','#DASH','#XMR','#xmr','Monero','#TRON','#TRX','$TRX','#Lisk','#LSK','#VeChain','#EthereumClassic','$ETC','#Qtum','#SmartContracts','#icx','$icx'],
 locations: ['-127.33,23.34,-55.52,49.56'], //US   (All the world:'-180,-90,180,90; New York City:-74,40,-73,41; San Francisco:-122.75,36.8,-121.75,37.8, US:-127.33,23.34,-55.52,49.56)
 waitBetweenDescribeCallsInSeconds: 2,
 recordsToWritePerBatch: 100,
 waitBetweenPutRecordsCallsInMilliseconds: 50,
 region: 'us-east-1'   
};
