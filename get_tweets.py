import requests
import pandas as pd
from datetime import datetime, timedelta


url = "https://twitter154.p.rapidapi.com/search/search"
url_continuation = "https://twitter154.p.rapidapi.com/search/search/continuation"

headers = {
	"X-RapidAPI-Key": "6ec7934ea2msh8a8cb22698520c3p1f877djsn745040c35697",
	"X-RapidAPI-Host": "twitter154.p.rapidapi.com"
}


all_tweets = []
zero_tweets_count_limit = 1
continuation_token = ""
query = "https://thegrayzone.com/2023/10/11/beheaded-israeli-babies-settler-wipe-out-palestinian/"

# Define start and end dates
start_date = datetime(2023, 10, 11)
end_date = datetime(2023, 10, 22)

# Initialize an empty list
date_list = []

# Loop through the range of dates and append to the list
while start_date <= end_date:
    date_list.append(start_date.strftime("%Y-%m-%d"))
    start_date += timedelta(days=1)

# Print the list of dates
# print(date_list)

for i in range(0, len(date_list)):

    print("date :", date_list[i])

    zero_tweet_counts = 0

    if i < (len(date_list) - 1):

        continuation_token = ""
        querystring = {"query":"{query}".format(query=query),"section":"latest","min_retweets":"0","min_likes":"0","limit":"20","start_date":"{date}".format(date=date_list[i]),"language":"en","end_date":"{date}".format(date=date_list[i+1])}

        response = requests.get(url,headers=headers, params=querystring)

        json_response = response.json()

        # print(json_response)
        print(json_response['continuation_token'])

        all_tweets.extend(json_response['results'])

        print(len(json_response['results']))
        print(len(all_tweets))

        if(len(json_response['results']) == 0):
            zero_tweet_counts+=1

        if('continuation_token' in json_response.keys()):
            continuation_token = json_response['continuation_token']
        else:
            print("continuation_token was not present")

        while(continuation_token != ""):

            if(zero_tweet_counts >= zero_tweets_count_limit):
                print("Not finding any tweets")
                break

            querystring = {"query":"{query}".format(query=query),"continuation_token":"{continuation_token}".format(continuation_token=continuation_token),"section":"latest","min_retweets":"0","min_likes":"0","limit":"20","start_date":"{date}".format(date=date_list[i]),"language":"en","end_date":"{date}".format(date=date_list[i+1])}
            response = requests.get(url_continuation,headers=headers,params=querystring)
            json_response = response.json()


            print(json_response['continuation_token'])
            
            if('results' in json_response.keys()):
                all_tweets.extend(json_response['results'])

            print(len(json_response['results']))
            print(len(all_tweets))

            if(len(json_response['results']) == 0):
                zero_tweet_counts+=1
            
            old_token = continuation_token
            if('continuation_token' in json_response.keys()):
                continuation_token = json_response['continuation_token']
            else:
                continuation_token = ""
                print("continuation_token not present, breaking out of the loop")

            if(old_token == continuation_token):
                print("search exhausted, same token repeating")
                break
            



tweeters_df = pd.DataFrame(all_tweets)

tweeters_df.to_json("BetOn_1xBet.json",orient='records')

