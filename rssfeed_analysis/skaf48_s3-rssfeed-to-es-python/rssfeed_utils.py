#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from textblob import TextBlob

class Sentiments:
    POSITIVE = 'Positive'
    NEGATIVE = 'Negative'
    NEUTRAL = 'Neutral'
    CONFUSED = 'Confused'
    
id_field = 'id_str'
emoticons = {Sentiments.POSITIVE:'😀|😁|😂|😃|😄|😅|😆|😇|😈|😉|😊|😋|😌|😍|😎|😏|😗|😘|😙|😚|😛|😜|😝|😸|😹|😺|😻|😼|😽',
             Sentiments.NEGATIVE : '😒|😓|😔|😖|😞|😟|😠|😡|😢|😣|😤|😥|😦|😧|😨|😩|😪|😫|😬|😭|😾|😿|😰|😱|🙀',
             Sentiments.NEUTRAL : '😐|😑|😳|😮|😯|😶|😴|😵|😲',
             Sentiments.CONFUSED: '😕'
             }


rssfeed_mapping = {'properties':
                        {'timestamp_ms': {
                                  'type': 'date'
                                  },
                        'author': {
                                'type' : 'string'
                            },
                        'summary': {
                                'type': 'string'
                            },
                        'title':{
                                'type': 'string'
                            },
                        'sentiments': {
                              'type': 'keyword'
                            }
                        }
                }

rssfeed_mapping_v5 = {'properties':
                        {'timestamp_ms': {
                                  'type': 'date'
                                  },
                        'author': {
                                'type' : 'text'
                            },
                        'summary': {
                                'type': 'text'
                            },
                        'title':{
                                'type': 'text'
                            },
                        'sentiments': {
                              'type': 'keyword'
                            }
                        }
                }


def _sentiment_analysis(rssfeed):
    rssfeed['emoticons'] = []
    rssfeed['sentiments'] = []
    _sentiment_analysis_by_emoticons(rssfeed)
    if len(rssfeed['sentiments']) == 0:
        _sentiment_analysis_by_text(rssfeed)


def _sentiment_analysis_by_emoticons(rssfeed):
    for sentiment, emoticons_icons in emoticons.iteritems():
        matched_emoticons = re.findall(emoticons_icons, rssfeed['text'].encode('utf-8'))
        if len(matched_emoticons) > 0:
            rssfeed['emoticons'].extend(matched_emoticons)
            rssfeed['sentiments'].append(sentiment)
    
    if Sentiments.POSITIVE in rssfeed['sentiments'] and Sentiments.NEGATIVE in rssfeed['sentiments']:
        rssfeed['sentiments'] = Sentiments.CONFUSED
    elif Sentiments.POSITIVE in rssfeed['sentiments']:
        rssfeed['sentiments'] = Sentiments.POSITIVE
    elif Sentiments.NEGATIVE in rssfeed['sentiments']:
        rssfeed['sentiments'] = Sentiments.NEGATIVE

def _sentiment_analysis_by_text(rssfeed):
    blob = TextBlob(rssfeed['text'].decode('ascii', errors="replace"))
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity < 0:
        sentiment = Sentiments.NEGATIVE
    elif sentiment_polarity <= 0.2:
        sentiment = Sentiments.NEUTRAL
    else:
        sentiment = Sentiments.POSITIVE
    rssfeed['sentiments'] = sentiment
    
def get_rssfeed(doc):
    rssfeed = {}
    rssfeed['author'] = doc['author']
    rssfeed['timestamp_ms'] = doc['timestamp_ms']
    rssfeed['text'] = doc['summary']
    rssfeed['title'] = doc['title']
    _sentiment_analysis(rssfeed)
    return rssfeed

def get_rssfeed_mapping(es_version_number_str):
    major_number = int(es_version_number_str.split('.')[0])
    if major_number >= 5:
        return rssfeed_mapping_v5
    return rssfeed_mapping