import boto3
import pandas as pd

client = boto3.client('comprehend')

data = pd.read_csv("data/train.csv")
sentiment=[]
sentiment_score=[]

for index, row in data.iterrows():
    response = client.detect_sentiment(Text=row["text"], LanguageCode='en')
    # print("sentiment: ", response["Sentiment"])
    sentiment.append(response["Sentiment"])
    sentiment_score.append(response["SentimentScore"][response["Sentiment"][:1]+response["Sentiment"][1:].strip().lower()])

print("Distribution of sentiments: ", pd.Series(sentiment).value_counts())
data["sentiment"]=sentiment
data["sentiment_score"]=sentiment_score

print("first 5 rows with sentiment: ", data.head())

with open("data/amazon_sentiments_data.csv","w") as outfile:
    data.to_csv(outfile, sep=",")


