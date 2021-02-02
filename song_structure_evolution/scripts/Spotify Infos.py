# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# +
import sys
sys.path.append('..')

import json
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from ergo_utilities import songs_info

import unicodedata as ud
import re
import os

# +
with open('../.config') as f:
    credentials=json.load(f)

client_credentials_manager = SpotifyClientCredentials(client_id=credentials['spotify']['client_id'],
                                                    client_secret = credentials['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
pd.set_option('max_columns', None)


# +
def search_for_track_by_artist(track, artist):
    query = f"{track} {artist}"
    results = sp.search(q=query)
    items = results['tracks']['items']
    if len(items) > 0:
        most_popular = pd.DataFrame(items).sort_values(by='popularity', ascending=False).iloc[0].to_dict()
        return most_popular
    else:
        return {}
    
def get_tracks_infos_df(tracks_artists_array, to_print=False):
    
    tracks_info = []
    for p in tracks_artists_array:
        
          # compare composed
        
        artist_name = ud.normalize('NFKC',str(p[1])).lower()
        track_name = ud.normalize('NFKC',str(p[0])).lower()
        
        if to_print:
            print(f"{track_name} - {artist_name}")
        
        track = search_for_track_by_artist(track_name, artist_name)
        
        if not track:
            # If there's no track, let's normaliza track name
            track_name = re.split('[\(\[]', track_name)[0].strip()
            if 'feat' in artist_name or 'and' in artist_name:
                artist_name =artist_name.split('feat')[0].strip()
                artist_name = artist_name.split('and')[0].strip()
                
            if to_print:
                print(f"{track_name} - {artist_name}")
                
            track = search_for_track_by_artist(track_name, artist_name)
            
        tracks_info.append(track)
        
    return tracks_info


# -
all_billboard_charts = [f for f in os.listdir('./data/billboard_charts/raw_wikipedia/') if '.csv' in f and f not in os.listdir('./data/billboard_charts/spotify_ref/')]

all_billboard_charts

for file in all_billboard_charts:
    print(file)
    
    chart_df = pd.read_csv(f"./data/billboard_charts/raw_wikipedia/{file}")
    songs_dict = chart_df.to_dict(orient='index')

    tracks = [(v['song_title'], v['artists']) for k,v in songs_dict.items()]
    tracks_info = get_tracks_infos_df(tracks)

    tracks_infos_df = pd.json_normalize(tracks_info, errors='ignore')

    spotify_infos = pd.concat([chart_df, tracks_infos_df], axis=1)
    
    spotify_infos.loc[:, 'year'] = file.split('.csv')[0]
    
    spotify_infos.to_csv(f"./data/billboard_charts/spotify_ref/{file}", index=False)
    spotify_infos.head()

dict_ = songs.loc[34].to_dict()
test = [(dict_['song_title'], dict_['artists'])]
tracks_info = get_tracks_infos_df(test, True)
