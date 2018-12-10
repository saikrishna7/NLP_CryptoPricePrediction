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

var config = require('./config');
var moment = require('moment');
var Twit = require('twit');
var util = require('util');
var logger = require('./util/logger');

function twitterStreamProducer(firehose) {
  var log = logger().getLogger('producer');
  var waitBetweenPutRecordsCallsInMilliseconds = config.waitBetweenPutRecordsCallsInMilliseconds;
  var T = new Twit(config.twitter)

  // Creates a new kinesis stream if one doesn't exist.
  function _createStreamIfNotCreated(callback) {
    firehose.describeDeliveryStream({DeliveryStreamName: config.firehose.DeliveryStreamName}, function(err, data) {
      if (err) {
        firehose.createDeliveryStream(config.firehose, function(err, data) {
          if (err) {
            // ResourceInUseException is returned when the stream is already created.
            if (err.code !== 'ResourceInUseException') {
              console.log(err);
              callback(err);
              return;
            }
            else {
              var msg = util.format('%s stream is already created! Re-using it.', config.firehose.DeliveryStreamName);
              console.log(msg);
              log.info(msg);
            }
          }
          else {
            var msg = util.format('%s stream does not exist. Created a new stream with that name.', config.firehose.DeliveryStreamName);
            console.log(msg);
            log.info(msg);
          }
          // Poll to make sure stream is in ACTIVE state before start pushing data.
          _waitForStreamToBecomeActive(callback);
        });
      }
      else {
        var msg = util.format('%s stream is already created! Re-using it.', config.firehose.DeliveryStreamName);
        console.log(msg);
        log.info(msg);
      }

      // Poll to make sure stream is in ACTIVE state before start pushing data.
      _waitForStreamToBecomeActive(callback);
    });

    
  }

  // Checks current status of the stream.
  function _waitForStreamToBecomeActive(callback) {
    firehose.describeDeliveryStream({DeliveryStreamName: config.firehose.DeliveryStreamName}, function(err, data) {
      if (!err) {
        if (data.DeliveryStreamDescription.DeliveryStreamStatus === 'ACTIVE') {
          log.info('Current status of the stream is ACTIVE.');
          callback(null);
        }
        else {
          var msg = util.format('Current status of the stream is %s.', data.DeliveryStreamDescription.DeliveryStreamStatus);
          console.log(msg);
          log.info(msg);
          setTimeout(function() {
            _waitForStreamToBecomeActive(callback);
          }, 1000 * config.waitBetweenDescribeCallsInSeconds);
        }
      }
    });
  }


  function _sendToFirehose() {
    // var locations = [ '-180,-90,180,90' ]; //all the world
   
    var stream = T.stream('statuses/filter', {locations: config.locations, track : config.track_words, language:'en'});
    
    var records = [];
    var record = {};
    var retweet={};
    var extended_tweet={};
    var recordParams = {};
    // var dateFormat = require('dateformat');
    
    stream.on('tweet', function (tweet) {
       // if (tweet.coordinates){
            // if (tweet.coordinates !== null){ 
              
              if(tweet["text"].includes("bitcoin"))
              {
                if(tweet["retweeted_status"]!=null)
                {
                  retweet=tweet["retweeted_status"]
                  extended_tweet=retweet["extended_tweet"]

                  let obj  = JSON.stringify(extended_tweet);
                  if(typeof obj!='undefined')
                  {
                    console.log("******************")
                    var newStr = String(obj.match(':.*,"display_text_range'));
                    var final_text = newStr.replace(',"display_text_range', "").substr(2);
                    var retweet_json = {};
                    
                    // var dateandtime = new Date(retweet["created_at"])
                    retweet_json["created_at"] = String(moment(new Date(retweet["created_at"])).format('YYYY-MM-DD HH:m:s')); 
                    retweet_json["id"] = retweet["user"]["id"]
                    retweet_json["screen_name"] = retweet["user"]["screen_name"]
                    retweet_json["text"] = final_text
                    retweet_json["followers_count"] = retweet["user"]["followers_count"]
                    retweet_json["favourites_count"] = retweet["user"]["favourites_count"]
                    console.log(retweet_json)

                    // console.log(new Date(retweet["created_at"]).toISOString()+"\t"+retweet["user"]["id"]+"\t"+retweet["user"]["screen_name"]+"\t"+final_text+"\t"+retweet["user"]["followers_count"]+"\t"+retweet["user"]["favourites_count"]+"\t"+retweet["user"]["retweet_count"]);
                    console.log("******************\n\n")
                      recordParams = {
                        DeliveryStreamName: config.firehose.DeliveryStreamName,
                        Record: {
                          
                          Data: JSON.stringify(retweet_json)+',\n'
                          // new Date(retweet["created_at"]).toISOString()+"\t"+retweet["user"]["id"]+"\t"+retweet["user"]["screen_name"]+"\t"+final_text+"\t"+retweet["user"]["followers_count"]+"\t"+retweet["user"]["favourites_count"]+"\t"+retweet["user"]["retweet_count"]+',\n'
                        }
                      };
                    
                      firehose.putRecord(recordParams, function(err, data) {
                      if (err) {
                        log.error(err);
                      }
                    });  
                  }
                   
                }
                else
                {
                    var tweet_json = {}
                    tweet_json["created_at"] = String(moment(new Date(tweet["created_at"])).format('YYYY-MM-DD HH:m:s')); 
                    tweet_json["id"] = tweet["user"]["id"]
                    tweet_json["screen_name"] = tweet["user"]["screen_name"]
                    tweet_json["text"] = tweet["text"]
                    tweet_json["followers_count"] = tweet["user"]["followers_count"]
                    tweet_json["favourites_count"] = tweet["user"]["favourites_count"]
                    console.log(tweet_json)

                    // console.log(new Date(tweet["created_at"]).toISOString()+"\t"+tweet["user"]["id"]+"\t"+tweet["user"]["screen_name"]+"\t"+tweet["text"]+"\t"+tweet["user"]["followers_count"]+"\t"+tweet["user"]["favourites_count"]+"\t"+0);
                    recordParams = {
                      DeliveryStreamName: config.firehose.DeliveryStreamName,
                      Record: {
                        Data: JSON.stringify(tweet_json)+',\n'
                        // new Date(tweet["created_at"]).toISOString()+"\t"+tweet["user"]["id"]+"\t"+tweet["user"]["screen_name"]+"\t"+tweet["text"]+"\t"+tweet["user"]["followers_count"]+"\t"+tweet["user"]["favourites_count"]+"\t"+0+',\n'
                      }
                    };
                  
                    firehose.putRecord(recordParams, function(err, data) {
                    if (err) {
                      log.error(err);
                    }
                  });
                }
                    
              }
              
              
             
          // }
        // }
    });
  }


  return {
    run: function() {
      log.info(util.format('Configured wait between consecutive PutRecords call in milliseconds: %d',
          waitBetweenPutRecordsCallsInMilliseconds));
      _createStreamIfNotCreated(function(err) {
        if (err) {
          log.error(util.format('Error creating stream: %s', err));
          return;
        }

        _sendToFirehose();
      });
    }
  };
}

module.exports = twitterStreamProducer;
