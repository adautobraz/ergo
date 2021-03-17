# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Setup

# +
import pandas as pd
import os 
from pathlib import Path
from pyYify import yify
import json
import csv
import time
import shutil


data_path = Path('./data')

# + [markdown] heading_collapsed=true
# # Find magnet links

# + hidden=true
imdb_top_250_df = pd.read_csv(data_path/'imdb_top250_movies.csv')
movie_ids = imdb_top_250_df['imdb_id'].unique().tolist()

# + hidden=true
written_files = [f.split('.json')[0] for f in os.listdir(data_path/'torrent_files/')]

# + hidden=true
for i in range(0, len(movie_ids)):
    
    if movie_ids[i] not in written_files:
        
        try:
    
            search_results = yify.search_movies('{}'.format(movie_ids[i]))

            torr_obj = search_results[0]
            torr_obj.getinfo()

            movie_torrents = torr_obj.torrents

            all_torrents = [[i, movie_torrents[i].magnet, movie_torrents[i].size] for i in range(0, len(movie_torrents))]
            torrent_df = pd.DataFrame(all_torrents, columns=['index','magnet', 'size'])
            torrent_df.loc[:, 'scale'] = torrent_df['size'].str[-3:].str.strip()
            torrent_df.loc[:, 'value'] = torrent_df['size'].str[:-3].astype(float)
            torrent_df.loc[:, 'value_adj'] = torrent_df['value']
            torrent_df.loc[torrent_df['scale'] == 'GB', 'value_adj'] = 1000*torrent_df['value']
            smallest = torrent_df.sort_values(by='value_adj').iloc[0]
            magnet = smallest['magnet']
            size = smallest['size']
            name = smallest['name']

            info_dict = {'magnet':magnet, 'size':size, 'imdb_id':movie_ids[i]}
            print(info_dict)
            
            with open(data_path/'torrent_files/{}.json'.format(movie_ids[i]), 'w') as fp:
                json.dump(info_dict, fp)
            time.sleep(1)
            
        except:
            print(i)
            continue

        if i%10 == 0:
            print(i)

# + hidden=true
all_torrents_infos = []
written_files = [f for f in os.listdir(data_path/'torrent_files/')]
for w in written_files:
    with open(data_path/'torrent_files/{}'.format(w), 'r') as r:
        info = json.load(r)
    all_torrents_infos.append(info)

torrent_info_df = pd.DataFrame(all_torrents_infos)

torrent_info_df.head()

# + hidden=true
# torrent_info_df.to_csv(data_path/'torrent_infos.csv', index=False)
# -

# # Batch Torrents

torrent_info_df = pd.read_csv(data_path/'torrent_infos.csv').sort_values(by='imdb_id')
torrent_info_df.head(1)

all_movies

batch_folder = data_path/'movie_batches'

# Create folder with movie segmentation, to run download in batches
all_movies = torrent_info_df['imdb_id'].tolist()
step = 10
for i in range(0, len(all_movies), step):
    df = torrent_info_df.set_index('imdb_id')
    # Get only batch size of movies
    if i + step > len(all_movies):
        df = df.iloc[i:]
    else:
        df = df.iloc[i:(i+step)]
        
    # Create folder, save magnets of movies to download
    torrents_batch = df['magnet'].to_dict()
    batch_code = "{}_{}".format(i, i+10)
    Path(batch_folder/f'{batch_code}').mkdir(parents=True, exist_ok=True)
    file = batch_folder/'{0}/magnets.json'.format(batch_code)
    with open(file, 'w') as w:
        json.dump(torrents_batch, w)
        
    ids_list = df.index.tolist()
    # Create folders for all movies with imdb_id as folder name
    for imdb_id in ids_list:
        Path(batch_folder/f"{batch_code}/movies_raw/{imdb_id}").mkdir(parents=True, exist_ok=True)
        
        
    # Check folders with complete files
    movies_path = batch_folder/f"{batch_code}/movies_raw"
    file_ids = [f for f in os.listdir(movies_path) if '.' not in f]
    downloaded = []
    for m in file_ids:
        files_in_folder = os.listdir(movies_path/m)
        if len(files_in_folder) > 0:
            complete = True
            for f in files_in_folder:
                if '.aria2' in f:
                    complete = False
            if complete:
                downloaded.append(m)
                            
    # Save file to allow aria to multidownload
    csv_df = df.loc[~df.index.isin(downloaded), ['magnet']].reset_index()
    csv_df.columns = ['id', 'magnet']
    
    outuput_folder = f'/Users/adautobrazdasilvaneto/Documents/ergo/silence_in_movies/data/movie_batches/{batch_code}/movies_raw'

    csv_df.loc[:, 'comand'] = csv_df.apply(lambda x: '{}\n dir={}/{}'.format(x['magnet'], outuput_folder, x['id']), axis=1)
    csv_df.loc[:, ['comand']]\
        .to_csv(batch_folder/f'{batch_code}/aria_comand.txt', 
                header=None, index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")


# +
# Remove empty folders
# batches = [f for f in os.listdir(batch_folder) if '.' not in f]

# for b in batches:
#     print(b)
#     folder = batch_folder/f'{b}/movies_raw'
#     ids = [f for f in os.listdir(folder) if '.' not in f]
    
#     for i in ids:
#         if len(os.listdir(folder/i)) == 0:
#             shutil.rmtree(folder/i) # If so, delete it
