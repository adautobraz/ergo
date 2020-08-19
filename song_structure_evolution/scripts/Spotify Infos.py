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
    
    track_info = []
    for i in range(0, len(artists)):
        track_info.
        
    track = search_for_track(track, artist)
    
    
r = search_for_track('cardigan', 'Taylor Swift')

# -




