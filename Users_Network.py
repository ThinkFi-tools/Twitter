import streamlit as st
from os.path import exists
import functions as fnc
import pandas as pd
import functions as fnc
import warnings
warnings.filterwarnings('ignore')
from streamlit_tags import st_tags
import networkx as nx
from pyvis.network import Network
import time
from twitterUsernameviaUserID import getHandles as gH
import os

st.set_page_config(layout="wide")
st.title('Twitter Users Network Analysis')

dirname = st.text_input('Enter directory name to store data: ', 'datasets')
dir = './'+dirname

if st.button('make directory'):
    if not os.path.exists(dir):
        os.makedirs(dirname)
        st.success('Directory Created!!!')
    else :
        st.warning('Directory Already Exists!!')

usernames = st_tags(
    label='## Enter Usernames without @:',
    text='Press enter to add more',
    maxtags=200)
ids = st_tags(
    label='## Enter User Ids:',
    text='Press enter to add more',
    maxtags=200)
ids_new = []
for id in ids:
    ids_new += id.split(' ')
ids = ids_new

usernames_new = []
for username in usernames:
    usernames_new += username.split(' ')
usernames = usernames_new
limit = st.number_input("enter number of followers to scrape", min_value=10, max_value=25000, value=10)

usernames_common_followers = st_tags(
    label='## Enter Usernames For common Followers info:',
    text='Press enter to add more',
    maxtags=200)

usernames_common_followings = st_tags(
    label='## Enter Usernames For common Followings info:',
    text='Press enter to add more',
    maxtags=200)

for i in range(len(ids)):
    ids[i] = int(ids[i])

def convert_df(df):
    return df.to_csv()

if st.button('Get Followers List'):

    if len(ids) > 0:
        usernames += fnc.get_username_from_id(ids)
        ids.clear()

    print(len(usernames))
    with st.spinner('Getting Authors Connections...'):
        for idx in usernames:
            followers_filepath = dirname+'/'+idx+'_followers.csv'
            if not exists(followers_filepath):
                while 1:
                    try:
                        fnc.get_followers(idx, followers_filepath, limit=limit)
                        break
                    except Exception as e:
                        print(e)
                        with st.spinner("waiting!!!"):
                            time.sleep(905)
    st.success('Done!!')

    for idx in usernames:
        # id_followers = pd.DataFrame(followers[idx])
        # id_following = pd.DataFrame(following[idx])
        id_followers = pd.read_csv(dirname+'/'+idx+'_followers.csv', names=['follower_id'])
        st.download_button(
            label="Download Followers of "+idx,
            data=convert_df(id_followers),
            file_name=idx+'_followers.csv',
            mime='text/csv',
        )

if st.button('Get Following List'):
    if len(ids) > 0:
        usernames += fnc.get_username_from_id(ids)
        ids.clear()

    with st.spinner('Getting Authors Connections...'):
        for idx in usernames:
            following_filepath = dirname+'/'+idx+'_following.csv'
            followings = []
            if not exists(following_filepath):
                fnc.get_following(idx, following_filepath, limit=limit)
    st.success('Done!!')

    for idx in usernames:
        # id_followers = pd.DataFrame(followers[idx])
        # id_following = pd.DataFrame(following[idx])
        id_following = pd.read_csv(dirname+'/'+idx+'_following.csv', names=['following_id'])
        st.download_button(
            label="Download Following of "+idx,
            data=convert_df(id_following),
            file_name=idx+'_following.csv',
            mime='text/csv',
        )

if st.button('Get Followers Network Graphs'):

    if len(ids) > 0:
        usernames += fnc.get_username_from_id(ids)
        ids.clear()

    # Creating Network Graph
    G_followers = nx.Graph()
    cnt = 0

    for idx in usernames:
        G_followers.add_node(idx,color='red', title=idx)

    followers = {}
    for idx in usernames:
        followers_filepath = dirname+'/'+idx+'_followers.csv'
        if not exists(followers_filepath):
                followers = fnc.get_followers(idx, followers_filepath, limit=limit)
        id_followers = pd.read_csv(followers_filepath, names=['follower_id'])
        followers[idx] = id_followers

    common_followers = {}
    for idx in usernames:
        start = time.time()
        for node in followers[idx].follower_id.values:
            lst = set()
            for idx2 in usernames:
                if node in followers[idx2].follower_id.values:
                    lst.add(idx2)
            if len(lst) > 1:
                if node in common_followers.keys():
                    common_followers[node].union(lst)
                else :
                    common_followers[node] = lst
                if not G_followers.has_node(str(node)):
                    G_followers.add_node(str(node),color='white', title=str(node))
                for source in lst:
                    G_followers.add_edge(source, str(node))
        print("done followers time taken "+idx+" :- ", time.time()-start)

    df = pd.DataFrame(list(common_followers.items()),columns = ['user id','follows'])
    df.to_csv(dirname+'/Users_Common_followers.csv', index=False)  
    st.download_button(
        label="Download Common Followers",
        data=convert_df(df),
        file_name='Users_Common_followers.csv',
        mime='text/csv',
    )
    
    scale=10 # Scaling the size of the nodes by 10*degree
    d = dict(G_followers.degree)
    #Updating dict
    d.update((x, scale*y) for x, y in d.items())
    #Setting up size attribute
    nx.set_node_attributes(G_followers,d,'size')
    st.header('Common Followers of Users')
    net = Network(height="1000px", width="2000px",notebook=True, font_color='#10000000', bgcolor="#222222")
    net.barnes_hut()
    net.from_nx(G_followers)
    nx.write_gexf(G_followers, dirname+'/followers.gexf')
    st.success('Common followers graphs gefx file successfully saved!!')


if st.button('Get Followings Network Graph'):
    if len(ids) > 0:
        usernames += fnc.get_username_from_id(ids)
        ids.clear()

    # Creating Network Graph
    G_following = nx.Graph()
    cnt = 0

    for idx in usernames:
        G_following.add_node(idx,color='red', title=idx)

    following = {}
    for idx in usernames:
        following_filepath = dirname+'/'+idx+'_following.csv'
        if not exists(following_filepath):
            followings =fnc.get_following(idx, following_filepath, limit=limit)
        id_following = pd.read_csv(following_filepath, names=['following_id'])
        following[idx] = id_following

    common_following = {}
    for idx in usernames:
        start = time.time()
        for node in following[idx].following_id.values:
            lst = set()
            for idx2 in usernames:
                if node in following[idx2].following_id.values:
                    lst.add(idx2)
            if len(lst) > 1:
                if node in common_following.keys():
                    common_following[node].union(lst)
                else :
                    common_following[node] = lst
                if not G_following.has_node(str(node)):                  
                    G_following.add_node(str(node),color='white', title=str(node))
                for source in lst:
                    G_following.add_edge(source, str(node))
        print("done followings time taken "+idx+" :- ", time.time()-start)

    scale=10 # Scaling the size of the nodes by 10*degree
    d = dict(G_following.degree)
    #Updating dict
    d.update((x, scale*y) for x, y in d.items())
    #Setting up size attribute
    nx.set_node_attributes(G_following,d,'size')
    st.header('Common Followings of Users')
    net = Network(height="1000px", width="2000px",notebook=True, font_color='#10000000', bgcolor="#222222")
    net.barnes_hut()
    net.from_nx(G_following)
    nx.write_gexf(G_following, dirname+'/following.gexf')
    st.success('Common following graphs gefx file successfully saved!!')


    df = pd.DataFrame(list(common_following.items()),columns = ['user id','followed by'])
    df.to_csv(dirname+'/Users_Common_followings.csv', index=False) 
    st.download_button(
        label="Download Common Followings",
        data=convert_df(df),
        file_name='Users_Common_followings.csv',
        mime='text/csv',
    )

if st.button('Get Common Followers/Followings info:'):

    followers = {}
    following = {}
    for idx in usernames_common_followers:
        followers_filepath = dirname+'/'+idx+'_followers.csv'
        if not exists(followers_filepath):
            fnc.get_followers(idx, followers_filepath, limit=limit)
        id_followers = pd.read_csv(followers_filepath, names=['follower_id'])
        followers[idx] = id_followers
    
    for idx in usernames_common_followings:
        following_filepath = dirname+'/'+idx+'_following.csv'
        if not exists(following_filepath):
            fnc.get_following(idx, following_filepath, limit=limit)
            while(not exists(following_filepath)):
                print('waiting for file to save..')
        id_following = pd.read_csv(following_filepath, names=['following_id'])
        following[idx] = id_following

    if len(usernames_common_followers) > 1:
        common_follower_nodes = set()
        for idx in usernames_common_followers:
            for node in followers[idx].follower_id.values:
                cnt = 0
                for idx2 in usernames_common_followers:
                    if node in followers[idx2].follower_id.values:
                        cnt += 1
                if cnt > 1:
                    common_follower_nodes.add(node)
        
        with st.spinner('getting common Followers info'):
            fnc.get_user_info(dirname+'/common_followers_info',user_ids=list(common_follower_nodes), user_field=['created_at','protected','verified', 'public_metrics', 'description', 'profile_image_url'])
        st.success('Done!!')

        common_followers_df = pd.read_csv(dirname+'/common_followers_info.csv', parse_dates=['created_at'])

        st.download_button(
            label="Download Common Followers Data as CSV",
            data=convert_df(common_followers_df),
            file_name='common_followers_info.csv',
            mime='text/csv',
        )

    if len(usernames_common_followings) > 1:
        common_following_nodes = set()
        for idx in usernames_common_followings:
            for node in following[idx].following_id.values:
                flag = 1
                for idx2 in usernames_common_followings:
                    if node not in following[idx2].following_id.values:
                        flag = 0
                        break
                if flag:
                    common_following_nodes.add(node)

        with st.spinner('getting common Following info'):
            fnc.get_user_info(dirname+'/common_followings_info',user_ids=list(common_following_nodes), user_field=['created_at','protected','verified', 'public_metrics', 'description', 'profile_image_url'])
        st.success('Done!!')

        common_followings_df = pd.read_csv(dirname+'/common_followings_info.csv',parse_dates=['created_at'])

        st.download_button(
            label="Download Common Following Data as CSV",
            data=convert_df(common_followings_df),
            file_name='common_followings_info.csv',
            mime='text/csv',
        )