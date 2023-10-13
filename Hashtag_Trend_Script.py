import streamlit as st
from os.path import exists
import functions as fnc
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud
import functions as fnc
from absl import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, OrderedDict
from itertools import chain
from collections import Counter, OrderedDict
from ast import literal_eval
import spacy
from deep_translator import GoogleTranslator
from streamlit_tags import st_tags
import time


st.set_page_config(layout="wide")
st.title('Hashtag Trend Analysis')
trend = st.text_input('enter query to find data')
start = st.date_input('enter start date', datetime.datetime.now() - datetime.timedelta(1))
end = st.date_input('enter end date', datetime.datetime.now())
st.write('searched for', trend)
words = st_tags(
    label='## Enter Words for sentiment analysis:',
    text='Press enter to add more',
    maxtags=100)
path = './datasets/'+trend+'..'+str(start)+'..'+str(end)


def convert_df(df):
     return df.to_csv()

def get_count(lst, word):
    cnt = 0
    for x in lst:
        if word.lower() in x:
            cnt+=1
    return cnt

def get_top_k(lst, k):
    cnt = 0
    res = {}
    all = list(chain.from_iterable(lst))
    for i in range(len(all)):
        all[i] = str(all[i])
    for key, value in OrderedDict((Counter(all).most_common())).items(): 
        if cnt == k:
            break
        if str(value).isnumeric():
            res[key] = str(value)
        else :
            res[value] = str(key)
        cnt += 1
    return res

def check_trend(lst, word):
    for x in lst:
        if word.lower() in x:
            return 1
    return 0

def get_count_list(lst, word_dic):
    tokens = lst.split()
    cnt = 0
    for x in word_dic:
        if '@'+x in tokens:
            cnt+=1
    return cnt


def present_data(data, dataX, authorsX, X, start):
    scores = {idx : 10 for idx in dataX.author_id}
    fig = plt.figure(figsize=(15,5))
    st.header('account creation density vs their tweets on #'+trend+' for accounts with<'+X+' tweets')
    sns.kdeplot(data[data.author_id.isin(dataX.author_id)].created_at, fill=True, color='r', label = 'tweets on #'+trend+', from these acccounts')
    sns.kdeplot(dataX[dataX.created_at.dt.date >= datetime.date(start.year, start.month, start.day)].created_at, fill=True, palette='magma', label='accounts created during timeline')
    sns.kdeplot(data.created_at, color='g', fill=False, label='total tweets on #'+trend)
    plt.legend()
    st.pyplot(fig)
    dataX['author_username'] = dataX.author_id.apply(lambda x : authorsX[authors.id == x]['username'].values[0])

    st.download_button(
        label="Download <"+X+" authors as CSV",
        data=convert_df(dataX),
        file_name=trend+'_authors_<'+X+'.csv',
        mime='text/csv',
    )

    lang = list(GoogleTranslator().get_supported_languages(as_dict=True).values())
    dataX_nottext = dataX[~dataX.lang.isin(lang)]
    dataX_text = dataX[dataX.lang.isin(lang)]

    try:
        fig = plt.figure(figsize=(20,4))
        st.header('comparison of total and non text tweets of users')
        plt.xticks(rotation=90)
        sns.barplot(x=dataX.author_username.value_counts().index, y=dataX.author_username.value_counts().values, label='all tweets', color='orange')
        sns.barplot(x=dataX_nottext.author_username.value_counts().index , y=dataX_nottext.author_username.value_counts().values, label='no text tweets', color='limegreen')
        plt.ylabel('count')
        plt.xlabel('author username')
        plt.legend()
        st.pyplot(fig)

        st.download_button(
            label="Download <"+X+" authors non text tweets as CSV",
            data=convert_df(dataX_nottext),
            file_name=trend+'_authors_<'+X+'_nontext.csv',
            mime='text/csv',
        )
    except:
        st.write('non text tweets not found!!!!')

    try: 
        fig = plt.figure(figsize=(20,4))
        plt.xticks(rotation=90)
        st.header('comparison of total and 9-5 tweets of users')
        sns.barplot(x=dataX.author_username.value_counts().index, y=dataX.author_username.value_counts().values, label='all tweets', color='orange')
        sns.barplot(x=dataX[(dataX.tweet_time.dt.hour >= 9) & (dataX.tweet_time.dt.hour <= 17)].author_username.value_counts().index , y=dataX[(dataX.tweet_time.dt.hour >= 9) & (dataX.tweet_time.dt.hour <= 17)].author_username.value_counts().values, label='9-5 tweets', color='royalblue')
        plt.ylabel('count')
        plt.xlabel('author username')
        plt.legend()
        st.pyplot(fig)

        st.download_button(
            label="Download <"+X+" authors 9-5 tweets as CSV",
            data=convert_df(dataX[(dataX.tweet_time.dt.hour >= 9) & (dataX.tweet_time.dt.hour <= 17)]),
            file_name=trend+'_authors_<50_9-5.csv',
            mime='text/csv',
        )
    except:
        st.write('9-5 tweets not found!!!!')


    threshold = int(X)*0.5
    lst = []
    for id,val in zip(dataX[(dataX['tweet_time'].dt.hour >= 9) & (dataX['tweet_time'].dt.hour <= 19)]['author_id'].value_counts().index, dataX[(dataX['tweet_time'].dt.hour >= 9) & (dataX['tweet_time'].dt.hour <= 19)]['author_id'].value_counts().values):
        if val > threshold:
            lst.append(id)

    #adding score if more than 50% tweets are between 9-5
    for i in dataX[(dataX['author_id'].isin(lst))].author_id.unique():
        scores[i] += 10

    #adding score if more thatn 50% tweets are non textual tweets
    for idx,val in zip(dataX_nottext.author_id.value_counts().index,dataX_nottext.author_id.value_counts().values):
        if val > threshold:
            scores[idx] += 10

    top_mentions = get_top_k(data.mentions, 10).keys()
    dataX['top_mentions_cnt'] = dataX.text.apply(lambda x : get_count_list(x,top_mentions))
    fig = plt.figure(figsize=(6,3))
    sns.histplot(dataX[dataX.top_mentions_cnt > 0]['top_mentions_cnt'], shrink=0.9, fill=True)
    st.pyplot(fig)

    # adding scores if account has mentioned more than top 2 accounts mentioned in this hashtag
    freq = {}
    for idx,val in zip(dataX.author_id, dataX.top_mentions_cnt):
        if val > 2:
            if idx in freq:
                if freq[idx] > 4:
                    scores[idx] += 10
                    freq[idx] = 1
                else:
                    freq[idx] += 1
            else:
                freq[idx] = 1
                scores[idx] += 10

    if 'sentiment' not in dataX_text.columns:
        start = time.time()
        with st.spinner('Text Pre Processing...'):
            translated = []
            for x,lang in zip(dataX_text.text, dataX_text.lang):
                if lang != 'en':
                    translated.append(fnc.get_translate(x,lang))
                else :
                    translated.append(x)
            dataX_text['text'] = translated
            try:
                dataX_text['text'] =  dataX_text['text'].apply(lambda x : fnc.clean_text(x))
            except:
                print("couldn't remove website/links")
            try:
                dataX_text['text_tokens'] = dataX_text['text'].apply(lambda x : fnc.tokenize(x))
            except:
                print('tokens not created')
            dataX_text['sentiment'] = dataX_text['text'].apply(lambda x : fnc.get_sentiment(x))
        st.success('Time Taken -> '+str(time.time()-start))


    fig = plt.figure(figsize=(15,5))
    negative = dataX_text.text_tokens
    negative = [" ".join(negative.values[i]) for i in range(len(negative))]
    negative = [" ".join(negative)][0]
    wc = WordCloud(min_font_size=3,max_words=200,width=1600,height=720, colormap = 'Set1', background_color='black').generate(negative)

    st.header('Frequent Words Used By authors')
    plt.imshow(wc,interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    st.pyplot(fig)

    fig = plt.figure(figsize=(25,4))
    st.header('counts of neutral, negtive and positive tweets for authors')
    plt.xticks(rotation=90)
    sns.histplot(data=dataX_text, x='author_username', hue=dataX_text.sentiment, multiple='dodge', palette='bright', discrete=True, shrink=.9)
    st.pyplot(fig)

    author_sentiments = {'username' : [], 'positive tweets' : [], 'neutral tweets' : [], 'negative tweets': []}

    for idx in dataX_text.author_username.unique():
        author_sentiments['username'].append(idx)
        for ind, va in zip(dataX_text[dataX_text.author_username == idx].sentiment.value_counts().index, dataX_text[dataX_text.author_username == idx].sentiment.value_counts().values):
            if ind == 'Positive':
                author_sentiments['positive tweets'].append(va)
            elif ind == 'Negative':
                author_sentiments['negative tweets'].append(va)
            else:
                author_sentiments['neutral tweets'].append(va)

        if len(author_sentiments['positive tweets']) < len(author_sentiments['username']):
            author_sentiments['positive tweets'].append(0)
        if len(author_sentiments['negative tweets']) < len(author_sentiments['username']):
            author_sentiments['negative tweets'].append(0)
        if len(author_sentiments['neutral tweets']) < len(author_sentiments['username']):
            author_sentiments['neutral tweets'].append(0)

    sentiment_df = pd.DataFrame(author_sentiments)
    st.download_button(
        label="Download All Sentiment Counts of Authors as CSV",
        data=convert_df(sentiment_df),
        file_name=trend+'_authors_<'+X+'_sentiments.csv',
        mime='text/csv',
    )
    
    for word in words:
        dataX_text['word_present'] = dataX_text.text_tokens.apply(lambda x : get_count(x,word))
        sent = {}
        cnt = 0
        for idx,x,sen in zip(dataX_text.author_username, dataX_text.word_present, dataX_text.sentiment):
            if sen == 'Positive' and x == 1:
                sen = 1
            elif sen == 'Negative' and x == 1:
                sen = -1
            else:
                sen = 0
            if idx in sent:
                sent[idx] += sen
            else:
                sent[idx] = sen
            cnt+=1
        
        for x in sent:
            sent[x] /= cnt
        fig = plt.figure(figsize=(20,5))
        st.header('average tweet sentiment of authors who have used the word ' +word)
        plt.xticks(rotation=90)
        plt.xlabel('author_username')
        plt.ylabel('average sentiment of tweets')
        sns.barplot(x=list(sent.keys()), y=list(sent.values()))
        st.pyplot(fig)

    similar_tweets = {'pair of authors ids' : 'list of similar tweets their similarity value'}
    nlp = spacy.load("en_core_web_lg")
    ids = dataX_text.tweet_id.values
    corpus = dataX_text.text.values
    logging.set_verbosity(logging.ERROR)
    text_embeddings = fnc.get_embeding(corpus)
    freq = {}
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            sim = cosine_similarity(np.array(text_embeddings[i]).reshape(1,-1), np.array(text_embeddings[j]).reshape(1,-1))
            if sim > 0.7 and dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0] != dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]:
                idxs = []
                idxs.append(dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0])
                idxs.append(dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0])
                idxs.sort()
                for i in range(len(idxs)):
                    idxs[i] = authorsX[authorsX.id == idxs[i]]['username'].values[0]
                idxs = tuple(idxs)
                if idxs in similar_tweets.keys():
                    similar_tweets[idxs].append(sim[0][0])
                    if freq[idxs] > 4:
                        scores[dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0]] += 10
                        scores[dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]] += 10
                        freq[idxs] = 0
                    freq[idxs] += 1
                else :
                    scores[dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0]] += 10
                    scores[dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]] += 10
                    similar_tweets[idxs] = [sim[0][0]]
                    freq[idxs] = 1

    similar_Tweets_df = pd.DataFrame(list(similar_tweets.items()))
    st.download_button(
        label="Download Similar Tweets as CSV",
        data=convert_df(similar_Tweets_df),
        file_name=trend+'_authors_<'+X+'_similarities.csv',
        mime='text/csv',
    )

    fig = plt.figure(figsize=(20,5))
    st.header('final caculated scores for each author')
    plt.xticks(rotation=90)
    plt.xlabel('author_id')
    plt.ylabel('scores')
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    st.pyplot(fig)

if 'button1' not in st.session_state:
    st.session_state.button1 = False
if 'button2' not in st.session_state:
    st.session_state.button2 = False
if 'button3' not in st.session_state:
    st.session_state.button3 = False
if 'button4' not in st.session_state:
    st.session_state.button4 = False
if 'button5' not in st.session_state:
    st.session_state.button5 = False
if 'button6' not in st.session_state:
    st.session_state.button6 = False
if 'button7' not in st.session_state:
    st.session_state.button7 = False

if st.button('Load Data') or st.session_state.button1:
    # st.session_state.button1 = True
    data_load_state = st.text('Loading data...')
    if exists(path+'.csv') :
        print('data found')
        data = pd.read_csv(path+'.csv',  parse_dates=['created_at'])
    else :
        print('getting data')
        with st.spinner('Getting Data...'):
            fnc.get_Tweets(
                filename=path,
                query=trend, 
                tweet_field=['created_at','author_id','source', 'lang', 'entities'], 
                user_field=['username','verified','created_at'], 
                start_date=datetime.datetime(start.year, start.month, start.day),
                end_date=datetime.datetime(end.year, end.month, end.day))
        st.success('Done!!')
        data = pd.read_csv(path+'.csv', parse_dates=['created_at'])
    if data is None:
        st.error('Data Could Not Be Loaded!!')
    else:
        data_load_state.text("Done Data Loaded!!")
        data = pd.read_csv(path+'.csv', parse_dates=['created_at'])
        data.mentions = data.mentions.apply(literal_eval)
        data.hashtags = data.hashtags.apply(literal_eval)
        data.ex_links = data.ex_links.apply(literal_eval)
        st.dataframe(data, 10000, 500)
        st.download_button(
            label="Download dataset as CSV",
            data=convert_df(data),
            file_name=trend+'.csv',
            mime='text/csv',
        )
        st.header('Tweet Timeline')
        fig = plt.figure(figsize=(30,6))
        plt.xlabel('Tweet Date')
        plt.ylabel('Number Of Tweets')
        sns.lineplot(x=data['created_at'].dt.date.value_counts().index, y=data['created_at'].dt.date.value_counts().values, color='r', markers='.', linestyle='-.')
        st.pyplot(fig)
        st.header('Number Of Unique Authors Tweeted during this timeline')
        st.write(str(len(data['author_id'].unique())))
        cnt = 0
        st.header('Top 10 Accounts Mentioned')

        all_mentions = get_top_k(data.mentions, 10)
        pd_all_mentions = pd.DataFrame(list(all_mentions.items()))
        st.download_button(
            label="Download Top Mentions as CSV",
            data=convert_df(pd_all_mentions),
            file_name=trend+'_top_mentions.csv',
            mime='text/csv',
        )
        for k in all_mentions:
            st.write('@'+k+' :: '+all_mentions[k])
        st.header('Top 10 Hashtags Used')
        all_hashtags = get_top_k(data.hashtags, 10)
        pd_all_hashtags = pd.DataFrame(list(all_hashtags.items()))
        st.download_button(
            label="Download Top Hashtags as CSV",
            data=convert_df(pd_all_hashtags),
            file_name=trend+'_top_hashtags.csv',
            mime='text/csv',
        )
        for k in all_hashtags:
            st.write('#'+k+' :: '+all_hashtags[k])

threshold1 = st.number_input('Enter Threshold To filter Accounts(Number of Tweets)', min_value=1, format='%i')
threshold2 = st.number_input('Enter Threshold To filter Accounts(Number of # in one tweet)', min_value=1, format='%i')
path_authors = path+'..authors..'+str(threshold1) + '..'+str(threshold2)

if st.button('Load Filtered Authors Data') or st.session_state.button2:
    # st.session_state.button2 = True

    if not exists(path+'.csv'):
        st.error('Data Not Found!!!')
    else :

        data = pd.read_csv(path+'.csv', parse_dates=['created_at'])
        data['text_tokens'] = data['text'].apply(fnc.tokenize)
        data['#trend_cnt'] = data['text_tokens'].apply(lambda x : fnc.get_count(x,trend))


        author_list_tweet_cnt = []
        for author_id,val in zip(data['author_id'].value_counts().index, data['author_id'].value_counts().values):
            if(val >= threshold1):
                if author_id not in author_list_tweet_cnt:
                    author_list_tweet_cnt.append(author_id)
        authors_list_hashtag_cnt = []
        for i in range(data.shape[0]):
            if data['#trend_cnt'][i] >= threshold2:
                if data['author_id'][i] not in authors_list_hashtag_cnt :
                    authors_list_hashtag_cnt.append(data['author_id'][i])

        for author in authors_list_hashtag_cnt:
            if author not in author_list_tweet_cnt:
                author_list_tweet_cnt.append(author)
        
        data_load_state = st.text('Loading data...')
        if not exists(path_authors+'.csv'):
            print('getting Authors data')
            with st.spinner('Getting Authors Data...'):
                fnc.get_user_info(filename=path_authors, user_ids=author_list_tweet_cnt, user_field=['created_at','protected','verified', 'public_metrics', 'description','profile_image_url'])
            st.success('Done!!')
        authors = pd.read_csv(path_authors+'.csv', parse_dates=['created_at'])

        authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))    

        st.download_button(
            label="Download Filtered Authors Data as CSV",
            data=convert_df(authors),
            file_name=trend+'_auhtors.csv',
            mime='text/csv',
        )

        data_load_state.text("Done Data Loaded!!")
        fig = plt.figure(figsize=(30,8))
        st.subheader('density of accounts creation Date(filtered based on number of tweets and number of hashtags) and their tweets on #'+trend)
        sns.kdeplot(data[data.author_id.isin(author_list_tweet_cnt)]['created_at'], fill=True, color='r', label='#'+trend+' Tweets')
        sns.kdeplot(authors[authors.created_at.dt.date >= datetime.date(start.year, start.month, start.day)].created_at, fill=True, color='b', label='filtered accounts creation date')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure(figsize=(30,8))
        plt.xticks(rotation=90)
        st.subheader('Account Creation Pattern for Filtered Accounts')
        sns.histplot(authors.sort_values('short_date')['short_date'], fill=True, label='filtered accounts creation date', shrink=0.8)
        plt.legend()
        st.pyplot(fig)
        # st.dataframe(data)

        val10 = []
        val50 = []
        val100 = []
        val200 = []
        for ids,value in zip(authors.id.values, authors.tweets_count.values):
            if value < 10:
                val10.append(ids)
            elif value < 50:
                val50.append(ids)
            elif value < 100:
                val100.append(ids)
            else:
                val200.append(ids)
        
        st.write('authors with <10 total tweets :: ' + str(len(val10)))
        st.write('authors with <50 total tweets :: ' + str(len(val50)))
        st.write('authors with <100 total tweets :: ' + str(len(val100)))
        st.write('authors with <200 total tweets :: ' + str(len(val200)))



path_timeline = path_authors+'..timeline'

if st.button('Get Info About Authors With <10 Tweets') or st.session_state.button4:
    # st.session_state.button4 = True
    try :
        authors = pd.read_csv(path_authors+'.csv',parse_dates=['created_at'])
        authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    except:
        st.error('authors data could not be loaded!!!')
    val10 = []
    val50 = []
    val100 = []
    val200 = []
    for ids,value in zip(authors.id.values, authors.tweets_count.values):
            if value < 10:
                val10.append(ids)
            elif value < 50:
                val50.append(ids)
            elif value < 100:
                val100.append(ids)
            else:
                val200.append(ids)
    if not exists(path_timeline+'..<10.csv'):
        with st.spinner('Getting Timeline Data...'):
            fnc.get_timeline(filename=path_timeline+'..<10', ids=val10,tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'])
        st.success('Done!!')
    if not exists(path+'.csv'):
        st.error('Trend Data Not Found!!')

    data = pd.read_csv(path+'.csv',  parse_dates=['created_at'])
    data.mentions = data.mentions.apply(literal_eval)
    data.hashtags = data.hashtags.apply(literal_eval)
    data.ex_links = data.ex_links.apply(literal_eval)
    data10 = pd.read_csv(path_timeline+'..<10.csv',  parse_dates=['tweet_time'])
    data10['created_at'] = pd.to_datetime(data10['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0]))
    present_data(data,data10,authors,'10', start)


if st.button('Get Info About Authors With <50 Tweets') or st.session_state.button5:
    # st.session_state.button5 = True
    try :
        authors = pd.read_csv(path_authors+'.csv', parse_dates=['created_at'])
        authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    except:
        st.error('authors data could not be loaded!!!')
    val10 = []
    val50 = []
    val100 = []
    val200 = []
    for ids,value in zip(authors.id.values, authors.tweets_count.values):
            if value < 10:
                val10.append(ids)
            elif value < 50:
                val50.append(ids)
            elif value < 100:
                val100.append(ids)
            else:
                val200.append(ids)
    if not exists(path_timeline+'..<50.csv'):
        with st.spinner('Getting Timeline Data...'):
            fnc.get_timeline(filename=path_timeline+'..<50', ids=val50,tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'])
        st.success('Done!!')
    if not exists(path+'.csv'):
        st.error('Trend Data Not Found!!')

    data = pd.read_csv(path+'.csv', parse_dates=['created_at'])
    data.mentions = data.mentions.apply(literal_eval)
    data.hashtags = data.hashtags.apply(literal_eval)
    data.ex_links = data.ex_links.apply(literal_eval)
    data50 = pd.read_csv(path_timeline+'..<50.csv',  parse_dates=['tweet_time'])
    data50['created_at'] = pd.to_datetime(data50['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0]))
    present_data(data,data50,authors, '50', start)

if st.button('Get Info About Authors With <100 Tweets') or st.session_state.button6:
    # st.session_state.button6 = True
    try :
        authors = pd.read_csv(path_authors+'.csv', parse_dates=['created_at'])
        authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    except:
        st.error('authors data could not be loaded!!!')
    val10 = []
    val50 = []
    val100 = []
    val200 = []
    for ids,value in zip(authors.id.values, authors.tweets_count.values):
            if value < 10:
                val10.append(ids)
            elif value < 50:
                val50.append(ids)
            elif value < 100:
                val100.append(ids)
            else:
                val200.append(ids)
    if not exists(path_timeline+'..<100.csv'):
        with st.spinner('Getting Timeline Data...'):
            fnc.get_timeline(filename=path_timeline+'..<100', ids=val100,tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'])
        st.success('Done!!')
    if not exists(path+'.csv'):
        st.error('Trend Data Not Found!!')

    data = pd.read_csv(path+'.csv', parse_dates=['created_at'])
    data.mentions = data.mentions.apply(literal_eval)
    data.hashtags = data.hashtags.apply(literal_eval)
    data.ex_links = data.ex_links.apply(literal_eval)
    data100 = pd.read_csv(path_timeline+'..<100.csv', names = ['author_id', 'tweet_id','source', 'tweet_time', 'text', 'lang', 'likes', 'retwets'], parse_dates=['tweet_time'])
    data100['created_at'] = pd.to_datetime(data100['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0]))
    present_data(data,data100,authors,'100', start)

if st.button('Get Info About Authors With <200 Tweets') or st.session_state.button7:
    # st.session_state.button7 = True
    try :
        authors = pd.read_csv(path_authors+'.csv',parse_dates=['created_at'])
        authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    except:
        st.error('authors data could not be loaded!!!')
    val10 = []
    val50 = []
    val100 = []
    val200 = []
    for ids,value in zip(authors.id.values, authors.tweets_count.values):
            if value < 10:
                val10.append(ids)
            elif value < 50:
                val50.append(ids)
            elif value < 100:
                val100.append(ids)
            else:
                val200.append(ids)
    if not exists(path_timeline+'..<200.csv'):
        with st.spinner('Getting Timeline Data...'):
            fnc.get_timeline(filename=path_timeline+'..<200', ids=val200,tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'])
        st.success('Done!!')
    if not exists(path+'.csv'):
        st.error('Trend Data Not Found!!')

    data = pd.read_csv(path+'.csv',  parse_dates=['created_at'])
    data.mentions = data.mentions.apply(literal_eval)
    data.hashtags = data.hashtags.apply(literal_eval)
    data.ex_links = data.ex_links.apply(literal_eval)
    data200 = pd.read_csv(path_timeline+'..<200.csv', parse_dates=['tweet_time'])
    data200['created_at'] = pd.to_datetime(data200['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0]))
    present_data(data,data200,authors,'200', start)