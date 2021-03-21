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

# # Setup

# +
import pandas as pd
from pathlib import Path
import os
from pymediainfo import MediaInfo


data_path = Path('./data')
movies_prep_path = data_path/'movies_prep'
# -

# # Data Prep

# ## Silences + Subs

movie_ids = [f for f in os.listdir(movies_prep_path) if not f.startswith('.') if os.path.exists(movies_prep_path/f'{f}/audio/silences.csv')]

# +
array = []
for movie_id in movie_ids:
    s = movies_prep_path/f'{movie_id}/audio/silences.csv'
    movie_id = str(s).split('/')[-3]
    df = pd.read_csv(s)
    df.loc[:, 'imdb_id'] = movie_id
#     df.loc[:, 'position'] = df['silence_start'].rank(method='first').astype(int)
#     df.loc[:, 'silence_id'] = df.apply(lambda x: f"{x['imdb_id']}_{x['position']}", axis=1)
    array.append(df)
    
silences_raw_df = pd.concat(array)
silences_raw_df.head()
# -

subs_raw_df = pd.read_csv(data_path/'prep/all_subtitles.csv')
subs_raw_df.loc[:, 'start'] = subs_raw_df.apply(lambda x: x['start_h']*3600 + x['start_min']*60 +  x['start_s'] + x['start_ms']/1000, axis=1)
subs_raw_df.loc[:, 'end'] = subs_raw_df.apply(lambda x: x['end_h']*3600 + x['end_min']*60 +  x['end_s'] + x['end_ms']/1000, axis=1)
subs_df = subs_raw_df.loc[:, ['imdb_id', 'text', 'start', 'end', 'position']]
subs_df.head()


def format_timer(x):
     return f"{x//3600:02.0f}:{(x%3600)//60:02.0f}:{(x%3600)%60:02.0f}"


# +
cols = ['imdb_id', 'start', 'end']

df_subs = subs_df.loc[:, ['imdb_id', 'start', 'end']]
df_subs.columns = cols
df_subs.loc[:, 'type'] = 'subtitle'

df_sil = silences_raw_df.loc[:, ['imdb_id', 'silence_start', 'silence_end']]
df_sil.columns = cols
df_sil.loc[:, 'type'] = 'silence'

df = pd.concat([df_subs, df_sil], axis=0).sort_values(by=['imdb_id', 'start'])
df.loc[:, 'before'] = df.groupby(['imdb_id'])['type'].shift(1).fillna('start')
df.loc[:, 'after'] = df.groupby(['imdb_id'])['type'].shift(-1).fillna('end')
df.loc[:, 'prev_end'] = df.groupby(['imdb_id'])['end'].shift(1).fillna(0)
df.loc[:, 'next_start'] = df.groupby(['imdb_id'])['end'].shift(-1).fillna(df['end'])

silences_df = df.loc[df['type'] == 'silence'].drop(columns=['type'])
# silences_df.loc[:, 'mid_sub'] = (silences_df['start'] < silences_df['last_end']) | (silences_df['end'] > silences_df['next_start'])

silences_df.loc[:, 'rank'] = silences_df\
                                .sort_values(by=['imdb_id','start'])\
                                .groupby(['imdb_id'])['start'].rank(method='first').astype(int)

silences_df.loc[:, 'silence_id'] = silences_df.apply(lambda x: f"{x['imdb_id']}_{x['rank']}", axis=1)
silences_df.loc[:, 'duration'] = silences_df['end'] - silences_df['start']

silences_df.head(1)
# -

silences_df.to_csv(data_path/'prep/silences_info.csv', index=False)

# ## Duration dict

movie_duration_dict = {}
for m in movie_ids:
    file_path = movies_prep_path/f'{m}/audio/movie_audio.mp3'
    media_info = MediaInfo.parse(file_path)
    #duration in seconds
    movie_duration = media_info.tracks[0].duration/1000 
    movie_duration_dict[m] = movie_duration

# ## Movie info 

imdb_df = pd.read_csv(data_path/'imdb_top250_movies.csv')
movie_info_df = imdb_df.loc[:, ['imdb_id', 'title', 'year', 'rating', 'genres', 'top_250_rank','color_info']]
movie_info_df.loc[:, 'genres'] = movie_info_df['genres'].apply(lambda x: eval(x))
movie_info_df.loc[:, 'color_info'] = movie_info_df['color_info'].apply(lambda x: eval(x))
movie_info_df.head(1)

# +
silences_df = pd.read_csv(data_path/'prep/silences_info.csv')

silences_df = pd.merge(left=silences_df, right=movie_info_df, on='imdb_id', how='inner')

silences_df.loc[:, 'total_duration'] = silences_df['imdb_id'].apply(lambda x: movie_duration_dict[x])
silences_df.loc[:, 'pos_rel'] = 100*silences_df['start']/silences_df['total_duration']
silences_df.loc[:, 'dur_rel'] = 100*silences_df['duration']/silences_df['total_duration']
silences_df.head()
# -


