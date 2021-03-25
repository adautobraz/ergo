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
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path 
import time
import pandas as pd
from fuzzywuzzy import fuzz
import re
import pickle
import time
import shutil
from mutagen.mp3 import MP3
import numpy as np
import json
from IPython.display import display

from sources.common_functions import *


data_path = Path('./data')
# -

imdb_df = pd.read_csv(data_path/'imdb_top250_movies.csv')

movies_raw_path = data_path/'movies_raw'
movies_prep_path = data_path/'movies_prep'
movies_folders = [f for f in os.listdir(movies_raw_path) if '.' not in f]

# Find all downloaded movies
downloaded = find_downloaded_movies(movies_raw_path)

torrent_names = {}
for d in downloaded:
    movie_id = str(d).split('/')[-1]
    torrent_names[movie_id] = [f for f in os.listdir(d) if not f.startswith('.')][0]
    aux_dict = {movie_id: torrent_names[movie_id]}
    
    Path(movies_prep_path/movie_id).mkdir(parents=True, exist_ok=True)
    
    with open(movies_prep_path/movie_id/'torrent_info.json', 'w') as w:
        json.dump(aux_dict, w)

# +
# if torrent_names:  
match_df = imdb_df.loc[imdb_df['imdb_id'].isin(torrent_names.keys()), ['imdb_id', 'title', 'year', 'top_250_rank']]
match_df.loc[:, 'file_downloaded'] = match_df['imdb_id'].apply(lambda x: torrent_names[x])
match_df.loc[:, 'file_adj'] = match_df['file_downloaded'].apply(lambda x: re.sub(r"(\[.*\])|(\(.*\))", r"", x))
match_df.loc[:, 'year_file'] = match_df['file_downloaded'].apply(lambda x: re.findall(r"[\. \(]*([0-9]{4})[\. \)]*", x)[0]).astype(int)

match_df.loc[:, 'match'] = match_df.apply(lambda x: fuzz.partial_ratio(x['file_adj'], x['title']), axis=1)
match_df.head(20)


# +
# # # Check movies that don't match 

wrong_movies = match_df.loc[match_df['year_file'] != match_df['year']]['imdb_id'].tolist()
print(wrong_movies)

display(match_df.loc[match_df['imdb_id'].isin(wrong_movies)])
# match_df.sort_values(by=['match'])

# if wrong_movies:

#     for w in wrong_movies:
#         shutil.rmtree(movies_raw_path/w)
#         shutil.rmtree(movies_prep_path/w)

#     match_df\
#         .loc[match_df['imdb_id'].isin(wrong_movies)]\
#         .to_csv(data_path/f'download_again/{int(time.time())}.csv', index=False)

# +
# Check if movies have all files needed
movie_status = get_movies_status(movies_raw_path, movies_prep_path)

done_ids = [k for k,v in movie_status.items() if len(v) == 0]
not_done = {k:v for k,v in movie_status.items() if len(v) > 0}
done_ids

# +
# For done files, remove their raw files

for d in done_ids:
    folder = movies_raw_path/f'{d}'
    print(folder)
    shutil.rmtree(folder)
# -
movies_prep_path = data_path/'movies_prep'
downloaded = [f for f in os.listdir(movies_prep_path) if not f.startswith('.')]

top_df = imdb_df.loc[imdb_df['top_250_rank'] <= 150]\
            .loc[~imdb_df['imdb_id'].isin(downloaded)]
top_df.head()


