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
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
import copy
import random
from difflib import SequenceMatcher
import re
import json
import plotly.express as px
import lyricsgenius

with open('../.config') as f:
    credentials=json.load(f)

client_credentials_manager = SpotifyClientCredentials(client_id=credentials['spotify']['client_id'],
                                                     client_secret = credentials['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

genius = lyricsgenius.Genius(credentials['genius']['client_access_token'])


pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# +
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import numpy as np
# -

# ## Lyrics Extract Data

# +
all_songs = pd.read_csv('./data/pop_divas_df_valid_albums.csv').set_index('id')

all_songs = all_songs.loc[:, ['name', 'album_name', 'artist']]
all_songs.head()

# +
from pathlib import Path
import os
Path("./data/lyrics/").mkdir(parents=True, exist_ok=True)
path = "/Users/adautobrazdasilvaneto/Documents/ergo/pop_divas_super_album/data/lyrics/"
data_folder = Path(path)

genius.skip_non_songs = True # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)"] # Exclude songs with these words in their title

# +
artists = all_songs['artist'].unique().tolist()
artists.remove('Taylor Swift')

artists

# +
artist = artists[4]
print(data_folder)
os.chdir(data_folder)
Path(data_folder/"{}/".format(artist)).mkdir(parents=True, exist_ok=True)
os.chdir(data_folder/"{}/".format(artist))

df = all_songs.loc[all_songs['artist'] == artist]
albums = df['album_name'].unique().tolist()

for album in albums:
    artist_genius = genius.search_artist(artist, sort='popularity', get_full_info=True, max_songs=1)

    songs = df.loc[df['album_name'] == album, 'name'].tolist()

    for s in songs:
        song_lyric = genius.search_song(s, artist_genius.name)
        if song_lyric:
            artist_genius.add_song(song_lyric)

    archive = 'all_songs__{}.json'.format(album)
    artist_genius.save_lyrics(filename=archive, overwrite=True,
)


# +
with open(')

data['songs'][1].keys()

songs_infos = []

for s in data['songs']:

    writers = [w['name'] for w in s['writer_artists']]
    producers = [p['name'] for p in s['producer_artists']]

    info = {'song': s['title'],
            'writers':writers,
            'producers':producers,
            'lyrics':s['lyrics']
           }
    songs_infos.append(info)
    
df = pd.DataFrame(songs_infos)

df.head()
# -

for f in data['songs'][1:]:
    f

# +
artist = genius.search_song("Welcome To New York", "Taylor Swift")
print(artist)

artist.save_lyrics()
# -

artist
