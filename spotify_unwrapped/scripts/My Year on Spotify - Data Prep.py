# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import json 
import plotly.express as px
import os
import numpy as np
import os 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re 
from fuzzywuzzy import fuzz

pd.set_option('max_columns', None)

# %%
with open('../credentials.json') as f:
    creds = json.load(f)

# %%
files = [f for f in os.listdir('./data/raw/') if 'StreamingHistory' in f]
dfs = []

for f in files:  
    df = pd.read_json(f'./data/raw/{f}', orient='records')
    dfs.append(df)
    
stream_df = pd.concat(dfs, axis=0).reset_index().iloc[:, 1:]

stream_df.loc[:, 'endTime'] = pd.to_datetime(stream_df['endTime'], infer_datetime_format=True)
stream_df['week'] = stream_df['endTime'].dt.to_period('W').apply(lambda r: r.start_time)
stream_df['date'] = stream_df['endTime'].dt.to_period('D').apply(lambda r: r.start_time)
stream_df['hour'] = stream_df['endTime'].dt.hour

stream_df.loc[:, 'hPlayed'] = stream_df['msPlayed']/(24*60*1000)

stream_df.head()


# %% [markdown]
# # Songs Characteristics 

# %%
def clean_string(x):
    return re.sub('[//]+', '', x).replace(' ', '_').lower()


# %%
spotify_info_path = './data/spotify_api_data/song_search'

# %% [markdown]
# ## Search songs

# %%
auth_manager = SpotifyClientCredentials(client_id=creds['spotify']['client_id'],
                                             client_secret=creds['spotify']['client_secret'],
                                             proxies=None, requests_session=True, requests_timeout=None, cache_handler=None)

# %%
sp = spotipy.Spotify(auth_manager=auth_manager)

df_all_songs = stream_df\
                .loc[:, ['artistName', 'trackName']]\
                .drop_duplicates()

df_all_songs.loc[:, 'song_id'] = df_all_songs.apply(lambda x: "{}__{}".format(clean_string(x['trackName']), clean_string(x['artistName'])), axis=1)

all_songs = df_all_songs\
                .to_dict(orient='records')

saved_songs = os.listdir(spotify_info_path)

for i in range(0, len(all_songs)): 
    if i%10 == 0:
        print(i)
    
    s = all_songs[i]
    artist = s['artistName']
    track = s['trackName']
    song_id = s['song_id']
    filename = f'{song_id}.json'
    
    if not filename in saved_songs:

        q = f'{track} {artist}'

        results = sp.search(q=q, type='track')
            
        with open(f'{spotify_info_path}/{filename}', 'w') as f:
            json.dump(results, f)    

# %% [markdown] heading_collapsed=true
# ## Compile audio URIs

# %% hidden=true
all_responses = {}

for f in os.listdir(spotify_info_path):
    if f.endswith('.json'):
        key = f.split('.json')[0]
        artist_name = key.split('__')[-1].replace('_', ' ')
        with open(f'{spotify_info_path}/{f}', 'r') as file:
            response = json.load(file)
            if response['tracks']['items']:
                results_match = []
                search_results = response['tracks']['items']
                for s in search_results:
                    album_artist_name = s['album']['artists'][0]['name'].lower()
                    results_match.append(fuzz.ratio(artist_name, album_artist_name))
                    
                best_match = results_match.index(max(results_match))
                all_responses[key] = search_results[best_match]

# %% hidden=true
track_infos_df = pd.DataFrame.from_dict(all_responses, orient='index')

track_infos_df.loc[:, 'artist_name'] = track_infos_df['artists'].apply(lambda x: x[0]['name'] if x else '')

track_infos_df.loc[:, 'album_name'] = track_infos_df['album'].apply(lambda x: x['name'] if x else '')
track_infos_df.loc[:, 'album_id'] = track_infos_df['album'].apply(lambda x: x['id'] if x else '')
track_infos_df.loc[:, 'album_type'] = track_infos_df['album'].apply(lambda x: x['album_type'] if x else '')
track_infos_df.loc[:, 'image_url'] = track_infos_df['album'].apply(lambda x: x['images'][0]['url'] if x else '')
track_infos_df.loc[:, 'release_date'] = track_infos_df['album'].apply(lambda x: x['release_date'] if x else '')

track_infos_df.loc[:, 'release_year'] = track_infos_df['release_date'].str[:4]

track_infos_df = track_infos_df.loc[:, ['id', 'duration_ms', 'name', 'popularity', 'uri', 'artist_name', 'album_name', 'album_id', 'image_url', 'release_year', 'album_type']]
track_infos_df.index.name = 'ref_id'

track_infos_df.to_csv('./data/prep/songs_spotify_response_info.csv', index=True)

# %% [markdown]
# ## Get audio features

# %%
uris = track_infos_df['uri'].unique().tolist()
all_results = []
sp = spotipy.Spotify(auth_manager=auth_manager)

i = 0
while i <= len(uris)//50:
    if i % 10 == 0:
        print(i)
    start = int(max(0, (i-1)*50))
    end = min(int(i*50), len(uris))
    tracks = uris[start:end]
    results = sp.audio_features(tracks=tracks)
    all_results.append(results)
    i += 1

# %%
features_df = pd.DataFrame(all_results).stack().to_frame().reset_index().iloc[:, 2:]
features_df = features_df[0].apply(pd.Series)
features_df.to_csv('./data/prep/audio_features.csv', index=False)
features_df.head()

# %% [markdown]
# ## Prepare full data spotify

# %%
df_features = pd.read_csv('./data/prep/audio_features.csv')\
                .loc[:, ['uri', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']] 

df_info = pd.read_csv('./data/prep/songs_spotify_response_info.csv')

df_streams = stream_df.copy()
df_streams.loc[: , 'ref_id'] =  df_streams.apply(lambda x: "{}__{}".format(clean_string(x['trackName']), clean_string(x['artistName'])), axis=1)

df = pd.merge(left = df_streams, right=df_info, on='ref_id', how='left')
df = pd.merge(left = df, right=df_features, on='uri', how='left')
df.to_csv('./data/prep/stream_history.csv', index=False)
