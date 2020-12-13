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

# +
with open('../.config') as f:
    credentials=json.load(f)

client_credentials_manager = SpotifyClientCredentials(client_id=credentials['spotify']['client_id'],
                                                    client_secret = credentials['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
pd.set_option('max_columns', None)


# +
def search_for_track(track, artist):
    results = sp.search(q=track + '+' + artist, type='track')
    items = results['tracks']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None
    
def get_tracks_infos_df(tracks_artists_array):
    
    tracks_info = []
    for p in tracks_artists_array:
        artist_name = p[1].lower()
        if 'feat' in artist_name:
            artist_name = artist_name.split('featuring')[0].strip()
            
        track_name = p[0]
        
        track = search_for_track(track_name, artist_name)
        
        tracks_info.append(track)
        
    return tracks_info
# +
songs = pd.read_csv('./data/charts_2010_2020.csv')
songs.head()

keys_id = songs.iloc[:, :2].drop_duplicates()
keys_id.loc[:, 'key'] = keys_id['artist'].str.lower().str.replace(' ', '_') + '__' + keys_id['song'].str.lower().str.replace(' ', '_')
keys_id = keys_id.set_index('key').to_dict(orient='index')
# -

# for k, v in keys_id:
tracks = [(v['song'], v['artist']) for k,v in keys_id.items()][:3]
# tracks
tracks_info = get_tracks_infos_df(tracks)

tracks

# +
# df = pd.json_normalize(tracks_info)
# df.columns = [c.replace('.', '_') for c in df.columns.tolist()]

# df['artists'] = df['artists'].apply(lambda x: [a['name'] for a in x])

results = pd.DataFrame(tracks_info)
results.head()