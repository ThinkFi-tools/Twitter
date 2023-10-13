import requests
import json
import time
import csv
import pandas as pd
import tensorflow_hub as hub
from nltk.tokenize import TweetTokenizer
from transformers import pipeline
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

## Proton Mail Account Header
# headers = {
#     'X-RapidAPI-Key': "6ec7934ea2msh8a8cb22698520c3p1f877djsn745040c35697",
#     'X-RapidAPI-Host': "twitter154.p.rapidapi.com"
# }

## ThinkFi Account Header
headers = {
    'X-RapidAPI-Key': 'd19b1d8a53msh6f826ca9bd285e6p1942d8jsn63fb36296581',
    'X-RapidAPI-Host': 'twitter154.p.rapidapi.com'
  }

def get_username_from_userid(userid):
	url = "https://twitter154.p.rapidapi.com/user/id"
	querystring = {"user_id":userid}
	
	response = requests.get(url,headers=headers,params=querystring)
	data_dict = json(response.text)
	return data_dict['username']


# Returns all the user details mentioned in the "columns" list as a dictionary
def get_user_details(username,userid = None):
	url = "https://twitter154.p.rapidapi.com/user/details"
	querystring = {"username":username}
	columns = ['username','user_id','creation_date', 'name','userame','is_verified','is_private','follower_count','following_count','number_of_tweets','description','profile_pic_url']
	response = requests.get(url, headers=headers, params=querystring)
	data_dict = json.loads(response.text)
	res={key:value for key,value in data_dict.items() if key in columns}
	return res

def get_first_result(username,size):
    url = "https://twitter154.p.rapidapi.com/user/tweets"
    querystring = {"username":username,"limit":str(min(100,size)),"include_replies":"false","include_pinned":"false"}
    
    response = requests.get(url, headers=headers, params=querystring)
    data_dict = json.loads(response.text)
    cnt = 0
    while 1:
        try:
            results = data_dict['results']
            break
        except:
            if cnt == 1:
                 print("not getting data for " + username)
                 return [], ""
            time.sleep(1)
            print("trying again for " + username)
            cnt += 1

    ct = data_dict['continuation_token']
    return results, ct

def get_next_result(ct, username,size):
    url = "https://twitter154.p.rapidapi.com/user/tweets/continuation"
    params = {"continuation_token":ct,"username":username,"limit":str(min(100,size)),"include_replies":"false"}
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    cnt = 0
    while 1:
        try:
            results = data['results']
            break
        except:
            if cnt == 1:
                 print("not getting data for " + username)
                 return [], ""
            time.sleep(1)
            print("trying again for " + username)
            cnt += 1
    ct = data['continuation_token']
    return results, ct

def get_number_of_tweets(username, size):
    user_detail = get_user_details(username)
    cnt = 0
    while 1:
        try:
            total_tweets = user_detail['number_of_tweets']
            break
        except:
            if cnt == 1:
                 print("not getting data for " + username)
                 return []
            print("trying again for " + username)
            time.sleep(1)
            cnt += 1
            print(user_detail)
    print(username)
    res, ct = get_first_result(username,size)
    tweets = res
    while len(tweets) <= size and len(tweets) <= total_tweets:
        res, ct = get_next_result(ct, username,size)
        if len(res) == 0:
            break
        tweets = tweets + res
    return tweets

def users_csv(filename, users, columns_user = ['username','creation_date', 'user_id', 'name','follower_count','following_count','is_private','is_verified','profile_pic_url','description','number_of_tweets']):
    
	csv_list_user = []

	for user in users:
		row =[]
		for key in columns_user:
			if(key in user.keys()):
				row.append(user[key])
			else :
				row.append(" ")
		csv_list_user.append(row)
			
	with open(filename, 'w') as file:
		dw = csv.DictWriter(file, delimiter=',',fieldnames=columns_user)

		writer = csv.writer(file)
		dw.writeheader()
		writer.writerows(csv_list_user)


def tweets_csv(filename,tweets_list, columns_tweet = ['tweet_id', 'creation_date', 'text', 'language', 'favorite_count', 'retweet_count', 'author_id', 'username']):
    csv_tweet = []
    for tweets in tweets_list:
        for tweet in tweets:
            row = []
            for key in columns_tweet:
                try:
                    if key == 'author_id':
                        row.append(tweet['user']['user_id'])
                    elif key == 'username':
                        row.append(tweet['user']['username'])
                    elif key == 'creation_date':
                         row.append(datetime.datetime.strptime(tweet['creation_date'], "%a %b %d %H:%M:%S %z %Y"))
                    else:
                        row.append(tweet[key])
                except:
                    row.append(" ")
            csv_tweet.append(row)
			
    with open(filename, 'w') as file: 
        dw = csv.DictWriter(file, delimiter=',',fieldnames=columns_tweet)
        writer = csv.writer(file)
        dw.writeheader()
        writer.writerows(csv_tweet)

def get_users_info(filename, usernames):
	users_info = []
	for username in usernames:
		users_info.append(get_user_details(username))
	
	users_csv(filename,users_info)
    
def get_users_timelines(filename, usernames, limit):
	users_timeline = []
	for username in usernames:
		users_timeline.append(get_number_of_tweets(username=username,size=limit))
	tweets_csv(filename,users_timeline)
	
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tt = TweetTokenizer()

sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-roberta-base-sentiment")
model = SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')

def get_embeding(input):
  return model.encode(input)

def get_count(lst, word):
    cnt = 0
    for x in [s.lower() for s in lst]:
        if x == word.lower():
            cnt+=1
    return cnt


def get_f_ratio(data, followers, following):
    data['f_ratio'] = data[followers]/ np.clip(data[following], 1e-7, 1e10)
    return data

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)
    return text

def clean_stopWords(text):
    return " ".join([w.lower() for w in text.split() if w.lower() not in stop_words and len(w) > 1])

def tokenize(text):
    return tt.tokenize(text)

def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def get_translate(text, source=None):
    translation = GoogleTranslator(target='en').translate(text=text)
    return translation

def get_ppm_count(lst):
    cnt = 0
    l = [s.lower() for s in lst]
    for x in range(len(l)-1):
        if l[x] == 'ppm':
            cnt+=1

    return cnt

def get_sentiment(text):
    return sentiment_task(text)[0]['label']



# user = get_user_details("namtaus")

# user_csv(user, columns_user)


# FOR TWEETS

# tweets = get_number_of_tweets(username="namtaus", size=10000)

# tweet_csv(tweets, columns_tweet)


