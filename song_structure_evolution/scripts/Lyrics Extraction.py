# -*- coding: utf-8 -*-
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
# # %load_ext autoreload
# # %reload_ext autoreload
# # %autoreload 2

# +
import sys
import pandas as pd
from pathlib import Path
import os
import time
import re
import numpy as np
from fuzzywuzzy import fuzz

pd.set_option('max_columns', None)
# -

project_path = Path(os.getcwd())
data_path = project_path/'data'

os.chdir(project_path)

# +
# sys.path.append('..')
from ergo_utilities import lyrics_info

# # %load_ext autoreload
# # %reload_ext autoreload
# # %autoreload 2

# +
all_charts = os.listdir(data_path/'billboard_charts/raw_wikipedia/')

all_dfs = []
for f in all_charts:
    df = pd.read_csv(data_path/f"billboard_charts/raw_wikipedia/{f}")
    
    df.loc[:, 'year'] = f.split('.csv')[0]
    all_dfs.append(df)
    
charts_df = pd.concat(all_dfs)\
            .reset_index()\
            .rename(columns={'index':'order'})

charts_df.to_csv('./data/billboard_charts_1946_2020.csv', index=False)

# +
charts_df = pd.read_csv('./data/billboard_charts_1946_2020.csv').sort_values(by=['year', 'order'], ascending=False)

charts_df.loc[:, 'song_id'] = charts_df['year'].astype(str) + '_' + charts_df['order'].astype(str)
charts_df.loc[:, 'song_sum'] = charts_df.groupby(['artists'])['year'].transform('sum')

charts_df.loc[:, 'song_sorter'] = 100 - charts_df['order'] 

charts_df.set_index('song_id', inplace=True)
charts_df.rename(columns={'song_title':'name', 'artists':'artist'}, inplace=True)

charts_df.sort_values(by=['song_sum'], ascending=False, inplace=True)

# charts_df.loc[:, 'artist_adj'] = charts_df['artist'].apply(lambda x: str(x).split(',')[0].split('feat')[0].split('and')[0].strip())
charts_df.loc[:, 'artist_adj'] = charts_df['artist'].apply(lambda x: re.split('and|feat|[,/(//&/[]', str(x))[0].strip())

charts_df.loc[:, 'song_adj'] = charts_df['name'].apply(lambda x: re.split('[/(///[]', re.sub(r'\([^)]*\)', '', str(x)))[0].strip())


charts_df.head()

# +
# all_artists = charts_df['artist_adj'].unique().tolist()
# all_artists

lyrics_path = data_path/'lyrics/raw/'

all_songs = charts_df\
                .sort_values(by=['year', 'song_sorter'], ascending=True)\
                .loc[charts_df['year'] >= 1960]\
                .loc[charts_df['order'] <= 100]

all_years = os.listdir(lyrics_path/'genius/per_year')
found_songs = []
for y in all_years:
    found_songs.extend([f.split('.json')[0] for f in os.listdir(lyrics_path/'genius/per_year/{}'.format(y)) if '.json' in f])

not_found_songs = [f for f in all_songs.index.tolist() if f not in found_songs]

song_info_dict = all_songs.to_dict(orient='index')

not_found_df = charts_df.loc[charts_df.index.isin(not_found_songs)]

# +
all_charts_sp = [f for f in os.listdir(data_path/'billboard_charts/spotify_ref/') if '.csv' in f]
spotify_refs_df = pd.concat([pd.read_csv(data_path/f'billboard_charts/spotify_ref/{f}').reset_index() for f in all_charts_sp], axis=0)

spotify_refs_df.loc[:, 'song_id'] = spotify_refs_df['year'].astype(str) + '_' + spotify_refs_df['index'].astype(str)

# +
not_found_sp = not_found_df.join(spotify_refs_df.set_index('song_id'), how='left', rsuffix='_spotify')

not_found_sp.loc[:, 'artists_spotify'] = not_found_sp['artists.1'].fillna('[]').apply(lambda x: [f['name'] for f in eval(x)])

not_found_dict = not_found_sp.sort_index(ascending=False).to_dict(orient='index')
# -

not_found_sp.head()


def check_match(song_obj, name, artist):
    
    if song_obj \
        and (fuzz.partial_ratio(song_adj, str(song_lyric.title)) > 50) \
        and (fuzz.partial_ratio(artist, str(song_lyric.artist)) > 50):
        return True
    else:
        return False


for key, infos in not_found_dict.items():
    try:
#         key = not_found_songs.pop()
#         infos = song_info_dict[key]

        year = infos['year']
        Path(lyrics_path/'genius/per_year/{}'.format(year)).mkdir(parents=True, exist_ok=True)
        year_folder = lyrics_path/'genius/per_year/{}'.format(year)

        os.chdir(year_folder)

        print(key)
        genius = lyrics_info.setup()

        song_id = key
        song_adj = infos['song_adj']
        artist = infos['artist_adj']

        song_lyric = genius.search_song(song_adj, artist)

        if song_lyric:
            song_lyric.to_json(filename='{}.json'.format(song_id))

        else:
            # If artist and song didn't match, use their spotify match
            if fuzz.partial_ratio(infos['name_spotify'], infos['song_adj']) > 50:
                song_adj = infos['name_spotify']

            artist = infos['artist_adj']
            if fuzz.partial_ratio(infos['artists_spotify'][0], infos['artist_adj']) > 50:
                artist = infos['artists_spotify'][0]

            song_lyric = genius.search_song(song_adj, artist)
            if check_match(song_lyric, song_adj, artist):
                song_lyric.to_json(filename='{}.json'.format(song_id))
            else:
                print('Não encontrou')
                
    except:
        print('Algum problema')

# +
# while len(not_found_songs) > 0:
#     try:
#         key = not_found_songs.pop()
#         infos = song_info_dict[key]

#         year = infos['year']
#         Path(lyrics_path/'genius/per_year/{}'.format(year)).mkdir(parents=True, exist_ok=True)
#         year_folder = lyrics_path/'genius/per_year/{}'.format(year)

#         print(key)
#         not_found = lyrics_info.download_song_lyrics(year_folder, infos['name'], infos['artist_adj'], key)

# #         if not_found:
# #             not_found_songs.insert(1, key)
# #         else:
# #             time.sleep(1)
            
#     except:
#         print('erro')
# #         not_found_songs.insert(1, key)
        
# # all_songs

# +
key = '1997_11'
infos = song_info_dict[key]

genius = lyrics_info.setup()
genius.skip_non_songs=True

song_adj = infos['song_adj']
song_id = key
artist = infos['artist_adj']
song_lyric = genius.search_song(song_adj, artist)

# +
# song_lyric.artist

# +
key = '2002_14'
song_info_dict[key]

genius = lyrics_info.setup()
genius.skip_non_songs=True

song_adj = 'Perfect'
song_id = key
artist = 'Pink'
song_lyric = genius.search_song(song_adj, artist)
