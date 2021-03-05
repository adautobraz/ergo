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
import json

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
            .rename(columns={'index':'order'})\
            .sort_values(by=['year', 'order'], ascending=False)


charts_df.loc[:, 'song_id'] = charts_df['year'].astype(str) + '_' + charts_df['order'].astype(str)

charts_df.loc[:, 'song_sorter'] = 100 - charts_df['order'] 

charts_df.set_index('song_id', inplace=True)
charts_df.rename(columns={'song_title':'name', 'artists':'artist'}, inplace=True)

# charts_df.loc[:, 'artist_adj'] = charts_df['artist'].apply(lambda x: str(x).split(',')[0].split('feat')[0].split('and')[0].strip())
charts_df.loc[:, 'artist_adj'] = charts_df['artist'].apply(lambda x: re.split('and|feat|[,/(//&/[]', str(x))[0].strip())

charts_df.loc[:, 'song_adj'] = charts_df['name'].apply(lambda x: re.split('[/(///[]', re.sub(r'\([^)]*\)', '', str(x)))[0].strip())

charts_df.to_csv(data_path/'billboard_charts_1946_2020.csv', index=True)

charts_df.head()

# +
all_charts_sp = [f for f in os.listdir(data_path/'billboard_charts/spotify_ref/') if '.csv' in f]
spotify_info = pd.concat([pd.read_csv(data_path/f'billboard_charts/spotify_ref/{f}').reset_index() for f in all_charts_sp], axis=0)

spotify_info.loc[:, 'song_id'] = spotify_info['year'].astype(str) + '_' + spotify_info['index'].astype(str)

spotify_info.loc[:, 'artist_sp'] = spotify_info['artists.1'].fillna('[]').apply(lambda x: ','.join([f['name'] for f in eval(x)])).fillna('')

spotify_info.head()

# +
charts_sptfy = charts_df\
                .join(spotify_info.set_index('song_id').loc[:, ['name', 'artist_sp']], how='left', rsuffix='_sp')\
                .fillna({'name_sp':'', 'artist_sp':''})

charts_sptfy.loc[:, 'name_unmatch'] = charts_sptfy.apply(lambda x: fuzz.ratio(x['name'], x['name_sp']) < 70, axis=1)
charts_sptfy.loc[:, 'artist_unmatch'] = charts_sptfy.apply(lambda x: fuzz.ratio(x['artist'], x['artist_sp']) < 70, axis=1)

charts_sptfy.loc[:, 'unmatch'] = charts_sptfy['name_unmatch'] | charts_sptfy['artist_unmatch']

charts_sptfy.head()

# +
# Finding which lyrics weren't saved
lyrics_path = data_path/'lyrics/raw/'

all_songs = charts_sptfy\
                .sort_values(by=['year', 'song_sorter'], ascending=True)\
                .loc[charts_df['year'] >= 1960]\
                .loc[charts_df['order'] <= 100]

all_years = os.listdir(lyrics_path/'genius/per_year')
found_songs = []
for y in all_years:
    found_songs.extend([f.split('.json')[0] for f in os.listdir(lyrics_path/'genius/per_year/{}'.format(y)) if '.json' in f])
     
not_found_songs = [f for f in all_songs.index.tolist() if f not in found_songs]

song_info_dict = all_songs.to_dict(orient='index')

not_found_df = charts_sptfy.loc[charts_sptfy.index.isin(not_found_songs)]

# not_found_dict = not_found_df.sort_index(ascending=False).to_dict(orient='index')
# not_found_df.to_csv(data_path/'lyrics/raw/not_found.csv', index=False)

# +
# # After not finding, and having to manually make alterations
# corrected_df = pd.read_csv(data_path/'lyrics/raw/not_found_corr.csv', sep=';')
# corrected_df.loc[:, 'song_id'] = corrected_df['year'].astype(str) + '_' + corrected_df['order'].astype(str)

# corrected_df.loc[corrected_df['name_'].isnull(), 'name_'] = corrected_df['name']
# corrected_df.loc[corrected_df['artist_'].isnull(), 'artist_'] = corrected_df['artist']

# corrected_df = corrected_df.set_index('song_id').loc[:, ['name_', 'artist_']]

# corrected_df.head()


# not_found_df = not_found_df.join(corrected_df, how='inner')

# not_found_df.loc[ :, 'name'] = not_found_df['name_']
# not_found_df.loc[ :, 'artist'] = not_found_df['artist_']

# not_found_dict = not_found_df.sort_index(ascending=False).to_dict(orient='index')
# -

def check_match(song_obj, name, artist):
    
    if song_obj \
        and (fuzz.partial_ratio(name, str(song_lyric.title)) > 50) \
        and (fuzz.partial_ratio(artist, str(song_lyric.artist)) > 50):
        return True
    else:
        return False


# +
for key, infos in not_found_dict.items():
#     try:
#         key = not_found_songs.pop()
#         infos = song_info_dict[key]

    year = infos['year']
    Path(lyrics_path/'genius/per_year/{}'.format(year)).mkdir(parents=True, exist_ok=True)
    year_folder = lyrics_path/'genius/per_year/{}'.format(year)

    os.chdir(year_folder)

    print(key)
    genius = lyrics_info.setup()
    genius.skip_non_songs = False
    genius.skip_non_songs = False

    song_id = key
    song = infos['name']
    artist = infos['artist']

    tries=[(infos['name'], infos['artist']), (infos['name'], infos['artist_adj']), 
           (infos['song_adj'], infos['artist_adj'])]
    
    if not infos['unmatch']:
        tries.extend([(infos['name_sp'], infos['artist_sp'])])

    if not infos['artist_unmatch']:
        tries.extend([(infos['name'], infos['artist_sp']), ((infos['song_adj'], infos['artist_sp']))])

    if not infos['name_unmatch']:
        tries.extend([(infos['name_sp'], infos['artist_adj']), (infos['name_sp'], infos['artist'])])

    for tup in tries:
        s = tup[0]
        a = tup[1]

        song_lyric = genius.search_song(s, a)

        if song_lyric and check_match(song_lyric, song, artist):
            song_lyric.to_json(filename='{}.json'.format(song_id))
            print('Achou e salvou')
            break

#     except:
#         print('Algum problema')
# -

not_found_df.loc[not_found_df['artist'] == 'Hanson']

# +
key = '1998_49'
infos = song_info_dict[key]

genius = lyrics_info.setup()
genius.skip_non_songs=True

song_adj = infos['name']
song_id = key
artist = infos['artist_adj']
song_lyric = genius.search_song(song_adj, artist)

# +
genius = lyrics_info.setup()
genius.skip_non_songs=False

song_adj = 'MMMBop'
artist = 'Hanson'
song_lyric = genius.search_song(song_adj, artist)
# -

print(str(song_lyric.artist), str(song_lyric.title))

# +
# Check for wrong or unmatch lyrics if necessary
genius_path = lyrics_path/'genius/per_year'
year_folders = os.listdir(genius_path)

all_lyrics = [] 

for y in year_folders:
    all_files = os.listdir(genius_path/y)
    for f in all_files:
        if '.json' in f:
            with open(genius_path/y/f, 'r', encoding='windows-1252') as r:
                lyric = json.load(r)
                
            lyric['song_id'] = f.split('.json')[0]
            lyric['artist'] = lyric['primary_artist']['name']
                
            all_lyrics.append(lyric)

# Find wrong lyrics and delete them
found_lyrics_df = pd.DataFrame.from_dict(all_lyrics).set_index('song_id')

lyrics_match = found_lyrics_df.join(charts_sptfy.loc[:, ['name', 'artist']], how='left', rsuffix='_raw')

lyrics_match.loc[:, 'name_unmatch'] = lyrics_match.apply(lambda x: fuzz.ratio(x['name'].lower(), x['title'].lower()) < 70, axis=1)
lyrics_match.loc[:, 'artist_unmatch'] = lyrics_match.apply(lambda x: fuzz.partial_ratio(x['artist_raw'].lower(), x['artist'].lower()) < 70, axis=1)

lyrics_match.loc[:, 'unmatch'] = lyrics_match['name_unmatch'] & lyrics_match['artist_unmatch']

unmatch_lyrics = lyrics_match.loc[lyrics_match['unmatch']].index.tolist()

# delete_files = False
# for f in unmatch_lyrics:
#     year = f.split('_')[0]
#     if os.path.exists(genius_path/f'{year}/{f}.json'):
#         print(delete)
#         os.remove(genius_path/f'{year}/{f}.json')
        
unmatch_lyrics
