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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
import copy
import random
from difflib import SequenceMatcher

import json

with open('../.config', 'rb') as f:
    creds = json.load(f)

# +
client_credentials_manager = SpotifyClientCredentials(client_id=creds['spotify']['client_id'],
                                                     client_secret = creds['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

data_folder = Path('./data/')


# +
# Funções 

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items_all = results['artists']['items']
    
    results = sp.search(q='artist:' + name, type='artist', market='BR')
    items_br = results['artists']['items']
    
    if len(items_br) > 0 and len(items_all) > 0:
        if similar(items_br[0]['name'], name) > similar(items_all[0]['name'], name):
            return items_br[0]
        else:
            return items_all[0]
    else:
        return None


def get_artist_singles_info(artist):

    singles = []
    results = sp.artist_albums(artist['id'], album_type='single')
    singles.extend(results['items'])

    while results['next']:
        results = sp.next(results)
        singles.extend(results['items'])

    singles_info = []
    for single in singles:
        single_dict = {'name':single['name'], 'release_date':single['release_date'], 'total_tracks':single['total_tracks']}
        singles_info.append(single_dict)
        
    if singles_info:
        a = pd.DataFrame.from_dict(singles_info)

        # Configurar datas
        a.loc[:, 'adjusted_name'] = a['name'].str.lower().str.strip()

        # Manter a última data do álbum
        a.sort_values(by='release_date', inplace=True)
        a.drop_duplicates(keep='last', inplace=True)

        # Agrupar dados de álbum
        singles_per_artist = a.groupby('adjusted_name').agg({'release_date':'max', 'total_tracks':'max', 'name':'max'})

        return singles_per_artist.to_dict(orient='index')
    
    else:
        return None
    
def get_artist_albums_info(artist):

    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])

    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])

    albums_info = []
    for album in albums:
        album_dict = {'name':album['name'], 'release_date':album['release_date'], 'total_tracks':album['total_tracks']}
        albums_info.append(album_dict)
        
    if albums_info:
        a = pd.DataFrame.from_dict(albums_info)

        # Configurar datas
        a.loc[:, 'adjusted_name'] = a['name'].str.lower().str.strip()

        # Manter a última data do álbum
        a.sort_values(by='release_date', inplace=True)
        a.drop_duplicates(keep='last', inplace=True)

        # Agrupar dados de álbum
        albuns_per_artist = a.groupby('adjusted_name').agg({'release_date':'max', 'total_tracks':'max', 'name':'max'})

        return albuns_per_artist.to_dict(orient='index')
    
    else: 
        return None
    
def get_artist_infos(name, uri):
    if uri:
        artist = sp.artist(uri)
    else:                    
        artist = get_artist(name)
    if artist:     
        artist_info_dict = {
            'genres':artist['genres'],
            'followers':artist['followers']['total'],
            'popularity':artist['popularity'],
            'uri':artist['uri'],
            'name':artist['name'],
            'albums_infos': get_artist_albums_info(artist),
            'single_infos': get_artist_singles_info(artist)
        }
        return artist_info_dict
    else:
        return None


# +
# audio_analysis = sp.audio_features('1GZH9Sv6zCIse2GKihRHKy')
# -

info_artist_dict = {}

# +
all_artists = {f:[f] for f in pd.read_csv(data_folder/'all_artists.csv').loc[:, 'artist'].tolist()}

splits = ['vs', 'feat']

for k, v in all_artists.items():
    for s in splits:
        if s in k:
            all_artists[k] = [a.strip() for a in k.split(s)]
            break
            
artists_to_get_info = list(all_artists.keys())

# +
# # Alterações nos termos de busca
all_artists['Robert Plant and The Sensational Space Shifters'] = ['Robert Plant']
all_artists['The Fever 333'] = ['FEVER 333']
all_artists['Lirinha feat Eddie'] = ['Cordel do Fogo Encantando', 'Banda Eddie']

all_artists['Seed'] = ['Seeed'] 
all_artists['Mix Hel'] = ['Mixhell']
all_artists['Lennox'] = ['Lenox Hortale']

all_artists['Daniel Brandão'] = []
all_artists['Mø'] = ['MØ']
all_artists['Rhythm Monkeys'] = []
all_artists['Classic'] = []
all_artists['School of Rock'] = []
# -

while len(artists_to_get_info) > 0:
    k = artists_to_get_info.pop()
    print(k)
    artists_list = all_artists[k]
    infos = []
    for a in artists_list:
        artist_info = get_artist_infos(a, None)
        if artist_info:
            infos.append(artist_info)   
    if infos:   
        info_artist_dict[k] = infos
    else:
        print('Não achei infos sobre:{}'.format(k))

# +
# Casos para corrigir manualmente
uris_to_correct = {
    'Tiê':['spotify:artist:5rTjH3aABAmPM5B6DZebZ7'],
    'Balls':['spotify:artist:0Cw7oXPvcDWk5bHaDSsMs8'],
    'Baia': ['spotify:artist:7JZvzvSDhw5PZu1SMBZSr0'],
    'Mø':['spotify:artist:0bdfiayQAKewqEvaU6rXCv'],
    'Ludov':['spotify:artist:1oJr9YeNcCuMjUtB6lQDlp'],
    'Griz':['spotify:artist:25oLRSUjJk4YHNUsQXk7Ut'],
    'The Outs': ['spotify:artist:67HEXQpQHNTCY4TQoblxnn'],
    'Marrero': ['spotify:artist:7lOh5oz88EerqqVhRlEcif'],    
    'Jørd': ['spotify:artist:2dhLVCzAEMbAu1SSkAoOGV'],
    'Iza': ['spotify:artist:3zgnrYIltMkgeejmvMCnes'],
    'Nas':['spotify:artist:20qISvAhX20dpIbOOzGK3q'],
    'Maz':['spotify:artist:6gYwbDKcqhLitCTlgF1oZn'],
    'Mika':['spotify:artist:5MmVJVhhYKQ86izuGHzJYA'],
    'Edgar':['spotify:artist:0ZeTpaHiNCZFAuQ7v1fZ7Z'],
    'The Hu': ['spotify:artist:0b2B3PwcYzQAhuJacmcYgc']
}

for k, uris in uris_to_correct.items():
    all_info = []
    for u in uris:
        all_info.append(get_artist_infos(k, u))   
    info_artist_dict[k] = all_info
# -



# +
# Checar se artistas corretos estão sendo selecionados
artists_to_get_info = []

for k, v in info_artist_dict.items():
    for a in v:
        if k.lower() != a['name'].lower():
            artists_to_get_info.append(k)
            print('{} - {}'.format(k, a['name']))

# +
# Alterações nos termos de busca 
# -

with open(data_folder/'artists_discography_summary.json', 'w') as fp:
    json.dump(info_artist_dict, fp)

info_artist_dict


