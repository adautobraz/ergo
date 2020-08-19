# -*- coding: utf-8 -*-
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

# +
import pandas as pd
import requests
import re
import json

with open('../.config', 'rb') as f:
    credentials = json.load(f)
# -

API_KEY = credentials['LastFM']['API_KEY']
USER_AGENT = credentials['LastFM']['USER_AGENT']

headers = {'user_agent':USER_AGENT}

all_acts = pd.read_csv('./data/all_artists.csv')['artist'].tolist()


# +
def get_artist_tags(artist):
    payload = {
        'api_key':API_KEY,
        'method':'artist.getTopTags',
        'format':'json',
        'autocorrect':1,
        'artist':artist
    }

    r = requests.get('http://ws.audioscrobbler.com/2.0/', headers=headers, params=payload)
    if 'error' not in r.json().keys():
        response_dict = r.json()['toptags']
        all_tags = {d['name']: d['count'] for d in response_dict['tag']}
        tags_dict = {'artist':response_dict['@attr']['artist'],
                     'tags':all_tags}
        
        return tags_dict
    else:
        print('Não achei artista: {}'.format(artist))
        return {}
    
    
def get_artist_albums(artist):
    payload = {
        'api_key':API_KEY,
        'method':'artist.getTopAlbums',
        'format':'json',
        'autocorrect':1,
        'artist':artist
    }

    r = requests.get('http://ws.audioscrobbler.com/2.0/', headers=headers, params=payload)
    return r
#     if 'error' not in r.json().keys():
#         response_dict = r.json()['toptags']
#         all_tags = {d['name']: d['count'] for d in response_dict['tag']}
#         tags_dict = {'artist':response_dict['@attr']['artist'],
#                      'tags':all_tags}
        
#         return tags_dict
#     else:
#         print('Não achei artista: {}'.format(artist))
#         return {}


# -

r['album']

# + code_folding=[]
payload = {
    'album':'tribalistas',
    'api_key':API_KEY,
    'method':'album.getInfo',
    'format':'json',
    'autocorrect':1,
    'artist':'Tribalistas'
}

r = requests.get('http://ws.audioscrobbler.com/2.0/', headers=headers, params=payload).json()

# +
tags_per_artist = {}
errors = []

for act in all_acts:
    print(act)
    artists_array = []
    artists_tags = []
    
    if 'vs' in act:
        artists_array = act.split('vs')
    elif 'feat' in act:
        artists_array = act.split('feat')
    else:
        artists_array = [act]
        
    for artist in artists_array:
        artists_tags.append(get_artist_tags(artist.strip()))
    if artists_tags:
        tags_per_artist[act] = artists_tags
    else:
        errors.append(act)
# -

with open('./data/lastfm_tags_per_act.json', 'w') as w:
    json.dump(tags_per_artist, w)

with open('./data/lastfm_tags_per_act.json', 'r') as r:
    tags_per_artist = json.load(r)
