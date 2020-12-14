# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

import pyyoutube
import pandas as pd
import operator
from fuzzywuzzy import fuzz
import os
import plotly.express as px
import spacy
import time
import re
import ast


# + [markdown] heading_collapsed=true
# # Data Download

# + hidden=true
API_KEY = "AIzaSyCtIIspulHKgjyLLrZN-4Ev7Q9Eedy6oSA"  # replace this with your api key.

# + hidden=true
"""
Retrieve some videos info from given channel.
Use pyyoutube.api.get_channel_info to get channel video uploads playlist id.
Then use pyyoutube.api.get_playlist_items to get playlist's videos id.
Last use get_video_by_id to get videos data.
"""

def get_videos(channel_id, channel_name=None):
    api = pyyoutube.Api(api_key=API_KEY)

    if channel_name:
        channel_res = api.get_channel_info(channel_name=channel_name)
    else:
        channel_res = api.get_channel_info(channel_id=channel_id)

    
    playlist_id = channel_res.items[0].contentDetails.relatedPlaylists.uploads

    playlist_item_res = api.get_playlist_items(
        playlist_id=playlist_id, count=None 
    )
    
    print(len(playlist_item_res.items))
    
    videos = []
    for item in playlist_item_res.items:
        video_id = item.contentDetails.videoId
        print(video_id)
        video_res = api.get_video_by_id(video_id=video_id, 
                                        parts='contentDetails,statistics,status,snippet')
        videos.append(video_res.items[0])

    return videos

# + hidden=true
channel_name = 'teamcoco'
videos = get_videos(None, channel_name)

# + hidden=true
video_infos = []

for video in videos:
    v = video
    video_dict = {
        'video_id':v.id,
        'title':v.snippet.title,
        'description':v.snippet.description,
        'channel':v.snippet.channelTitle,
        'tags':v.snippet.tags,
        'publishedAt':v.snippet.publishedAt,
        'duration': v.contentDetails.duration,
        'views':v.statistics.viewCount,
        'likes':v.statistics.likeCount,
        'dislikes':v.statistics.dislikeCount,
        'comments':v.statistics.commentCount,
    }
    
    video_infos.append(video_dict)
    

videos_df = pd.DataFrame(video_infos)

# + hidden=true
videos_df.to_csv('./data/raw/ConanOBrien.csv', index=False)
# -

# # Data Prep

nlp = spacy.load('en_core_web_sm')

# +
all_hosts = []
for f in os.listdir('./data/raw/'):
    df = pd.read_csv('./data/raw/{}'.format(f))
    df.loc[:, 'host'] = ' '.join(f.split('.csv')[0].split('_'))
    all_hosts.append(df)
    
latenight_df_raw = pd.concat(all_hosts).reset_index().drop(columns=['index'])


# -

def replace_host_name(text, host):
    if text and host:
        first_name = host.split(' ')[0]
        if first_name in text and host not in text:
            return text.replace(first_name, host)
    return text


# +
latenight_df_raw.loc[:, 'date'] = latenight_df_raw['publishedAt'].str[:10]
latenight_df_raw.loc[:, 'month'] = latenight_df_raw['publishedAt'].str[:7]


latenight_df_raw.loc[:, 'description_text'] = latenight_df_raw['description']\
                                                .str.replace(r'https?:\/\/.*', '')\
                                                .str.split('\n\n', expand=True).iloc[:, 0]\
                                                .fillna('')

latenight_df_raw.loc[:, 'description_text'] = latenight_df_raw.apply(lambda x:
                                                                         replace_host_name(x['description_text'], x['host']), axis=1) 

latenight_df_raw.loc[:, 'title_adj'] = latenight_df_raw.apply(lambda x:
                                                                replace_host_name(x['title'], x['host']), axis=1) 
    
latenight_df_raw.head(3)


# +
def prep_text(x, check=False):
    doc = nlp(x)
    text = ''
    for token in doc:
        if token.text in [',', '.', '-']:
            text += token.text
        elif token.pos_ in ('PROPN', 'NOUN'):
            text += ' ' + token.text.capitalize()  
        else:
             text += ' ' + token.text.lower()
        if check:
            print(token.text, token.pos_, token.dep_)
            print(text)
            
    # Replace special characters
    text = text.replace(' & ', ' and ').strip()
        
    return text


def find_entities(x):
    doc = nlp(x)
    ents = {}
    for ent in doc.ents:
        ents[ent.text] = ent.label_
            
    return ents


# -

latenight_df_raw.shape

latenight_df = latenight_df_raw.loc[latenight_df_raw['month'] <= '2020-02']#.sample(5000, random_state=42)

# +
start = time.time()

latenight_df.loc[:, 'title_adj'] = latenight_df['title_adj'].str.lower().apply(lambda x: prep_text(x))

# Parse title due to letter capitalizing
latenight_df.loc[:, 'ents_title_adj'] = latenight_df['title_adj'].apply(lambda x: find_entities(x))

latenight_df.loc[:, 'ents_description'] = latenight_df['description_text'].apply(lambda x: find_entities(x))

end = time.time()

print(end-start)

# +
# # Add csv
# latenight_df.to_csv('./data/all_videos.csv', index=False)

# Read csv
latenight_df = pd.read_csv('./data/all_videos.csv')

# +
latenight_df.loc[:, 'date'] = latenight_df['publishedAt'].str[:10]

latenight_df.loc[:, 'tags_list'] = latenight_df['tags'].fillna('[]').apply(lambda x: ast.literal_eval(x))

latenight_df.head()
# -



# +
def gen_entities(title_ents, description_ents):
    all_dicts = [title_ents, title_ents, description_ents]
    
    unique_dict = {}
    for d in all_dicts:
        for k, v in d.items():
            if k in unique_dict: 
                unique_dict[k].append(v)
            else:
                unique_dict[k] = [v]
    
    ents_dict = {}
    for k, v in unique_dict.items():
#         if 'PERSON' in v:
        ents_dict[k] = v
    
    ents_dict_len = {}
    for k, v in ents_dict.items():
        key_adj = k.replace('\'s', '').strip()
        ents_dict_len[key_adj] = len(v)
                          
    return ents_dict_len


def get_possible_guest(ocurr_dict):
    
    values_ordered = sorted(ocurr_dict.items(), key=lambda kv: (-kv[1], -len(kv[0].split(' '))))
#     print(values_ordered)

    final_dict = {}

    for t in values_ordered:
        name = t[0].lower()
        freq = t[1]
        if final_dict:
            other_key = ''
            for key in final_dict.keys():
                dist = fuzz.partial_ratio(name, key)
                if dist >= 80:
                    other_key = key
                    break
            if other_key:
                final_dict[other_key] += freq
            else:
                final_dict[name] = freq
        else:
            final_dict[name] = freq    
    
    if final_dict:
        max_value = max(final_dict.values())
        max_refs = [k for k, v in final_dict.items() if v == max_value]
        return max_refs
    else:
        return ''


# -
row = latenight_df.loc[27369]
text = row['tags_list']
# host = row['host']
# text = replace_host_name(text, host)
# get_possible_guest(text)
print(text)

# +
latenight_df.loc[:, 'all_ents'] = latenight_df.apply(lambda x: gen_entities(x['ents_title_adj'], 
                                                        x['ents_description']), axis=1)


latenight_df.loc[:, 'main_subject'] = latenight_df['all_ents'].apply(lambda x: get_possible_guest(x))

latenight_df.iloc[10:]
# -

# ## Analysis

# +
df = latenight_df.groupby(['channel', 'month'], as_index=False).agg({'video_id':'nunique'})

px.area(df, x='month', y='video_id', color='channel')
# -

latenight_df.sort_values(by='views', ascending=False)


