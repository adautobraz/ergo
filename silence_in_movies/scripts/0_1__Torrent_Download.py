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
from fuzzywuzzy import fuzz
from tpblite import TPB, CATEGORIES, ORDERS
from unidecode import unidecode

data_path = Path('./data')

# + [markdown] heading_collapsed=true
# # Find magnet links

# + hidden=true
imdb_top_250_df = pd.read_csv(data_path/'imdb_top250_movies.csv').set_index('imdb_id')
top_df = imdb_top_250_df.copy()
# top_df = imdb_top_250_df.loc[imdb_top_250_df['top_250_rank'] <= 100]

# + hidden=true
movie_dict = top_df.loc[:, ['title', 'year']].to_dict(orient='index')

# + hidden=true
written_files = [f.split('.json')[0] for f in os.listdir(data_path/'torrent_files/')]

# + [markdown] heading_collapsed=true hidden=true
# ## Yify

# + code_folding=[1] hidden=true
i = 0
for movie_id, movie_infos in movie_dict.items():
    
    if movie_id not in written_files:
        
        try:
    
            search_results = yify.search_movies('{}'.format(movie_id))

            torr_obj = search_results[0]
            torr_obj.getinfo()

            movie_torrents = torr_obj.torrents

            all_torrents = [[i, movie_torrents[i].magnet, movie_torrents[i].size, movie_torrents[i].name, movie_torrents[i].url] for i in range(0, len(movie_torrents))]
            torrent_df = pd.DataFrame(all_torrents, columns=['index','magnet', 'size', 'name', 'url'])

            torrent_df.loc[:, 'name_match'] = torrent_df['name'].apply(lambda x: fuzz.ratio(x.lower(), movie_infos['title'].lower()))

            torrent_df = torrent_df.loc[torrent_df['name_match'] > 80]

            if not torrent_df.empty:

                torrent_df.loc[:, 'scale'] = torrent_df['size'].str[-3:].str.strip()
                torrent_df.loc[:, 'value'] = torrent_df['size'].str[:-3].astype(float)
                torrent_df.loc[:, 'value_adj'] = torrent_df['value']
                torrent_df.loc[torrent_df['scale'] == 'GB', 'value_adj'] = 1000*torrent_df['value']
                smallest = torrent_df.sort_values(by='value_adj').iloc[0]
                magnet = smallest['magnet']
                size = smallest['size']
                name = smallest['name']
                url = smallest['url']

                info_dict = {'magnet':magnet, 'size':size, 'imdb_id':movie_id, 'name':name, 'url':url}

                with open(data_path/'torrent_files/{}.json'.format(movie_id), 'w') as fp:
                    json.dump(info_dict, fp)
                time.sleep(1)

            else:
                print(f"Name match problem: {movie_infos['title']}")

        except:
            print(f"Error torrent: {movie_infos['title']}")
            continue

        if i%10 == 0:
            print(i)
        
        i+= 1

# + hidden=true
all_torrents_infos = []
written_files = [f for f in os.listdir(data_path/'torrent_files/') if '.json' in f]
for w in written_files:
    with open(data_path/'torrent_files/{}'.format(w), 'r') as r:
        info = json.load(r)
    all_torrents_infos.append(info)

torrent_info_df = pd.DataFrame(all_torrents_infos)

torrent_info_df.head(1)

# + hidden=true
torrent_info_df.to_csv(data_path/'yify_torrents.csv', index=False)

# + [markdown] hidden=true
# ## The Pirate Bay

# + hidden=true
tpb = TPB()

# + hidden=true
i = 0

chosen_torrents = []
options = []
for movie_id, movie_infos in movie_dict.items():
    if i%10 == 0:
        print(i)
    search = unidecode(f"{movie_infos['title']} {movie_infos['year']}")

    # Customize your search
    torrents = tpb.search(search, page=1, category=CATEGORIES.VIDEO.ALL, order=ORDERS.SEEDERS.DES)
                       
    if torrents:
                               
        infos = []
        top = 5
        for t in torrents[:top]:
            infos.append([movie_id, t.title, t.byte_size, t.seeds, t.magnetlink])

        df = pd.DataFrame(infos)
        options.append(df)

    else:
        print(f'Not found: {search}')
    
    i += 1

# + hidden=true
tpb_df = pd.concat(options)
tpb_df.columns = ['imdb_id', 'title', 'byte_size', 'seeds', 'magnet']

tpb_df.loc[:, 'has_no_seeds'] = tpb_df['seeds'] < 2
tpb_df.loc[:, 'byte_neg'] = -tpb_df['byte_size']

tpb_df = tpb_df\
            .loc[tpb_df['byte_size']/(10**9) < 3]\
            .sort_values(by=['seeds', 'byte_neg'], ascending=False)\
            .drop_duplicates(subset=['imdb_id'], keep='first')

tpb_df.head()

# + hidden=true
tpb_df.to_csv(data_path/'the_pirate_bay_torrents.csv', index=False)
# -

# # Batch Torrents

yify_raw_df = pd.read_csv(data_path/'yify_torrents.csv').sort_values(by='imdb_id')
tpb_raw_df = pd.read_csv(data_path/'the_pirate_bay_torrents.csv').sort_values(by='imdb_id')
imdb_df = pd.read_csv(data_path/'imdb_top250_movies.csv')

# +
# Find files with problems on yify, replace them by magnets of The Pirate Bay
array = []
for f in os.listdir(data_path/'download_again'):
    df = pd.read_csv(data_path/f'download_again/{f}')
    array.append(df)

yify_problem = pd.concat(array)['imdb_id'].unique().tolist()

yify_df = yify_raw_df.loc[~yify_raw_df['imdb_id'].isin(yify_problem)]
tpb_df = tpb_raw_df.loc[tpb_raw_df['imdb_id'].isin(yify_problem)]

# torrents_df = pd.concat([yify_df, tpb_df], axis=0).loc[:, ['magnet', 'imdb_id']]
torrents_df = tpb_raw_df
# torrents_df = yify_df

# Join information
df = imdb_df.loc[:, ['imdb_id', 'batch', 'year', 'top_250_rank']]

torrent_info_df = pd.merge(left=torrents_df, right=df, on='imdb_id', how='left').sort_values(by='year', ascending=False)

torrent_info_df.head(1)

# +
batch_folder = data_path/'movie_batches'
movies_path = data_path/'movies_raw'
movies_prep = data_path/'movies_prep'

all_movies = torrent_info_df.set_index('imdb_id').to_dict(orient='index')
# -

# Create folders for all movie_ids
for movie_id, infos in all_movies.items():
    new_folder = movies_path/f'{movie_id}'
    Path(new_folder).mkdir(parents=True, exist_ok=True)

# Check done movies
prep_movies = []
for d in os.listdir(movies_prep):
    if 'tt' in d:
        if len([f for f in os.listdir(movies_prep/d) if not f.startswith('.')]) == 4:
            prep_movies.append(d)

# Check folders with complete files
file_ids = [f for f in os.listdir(movies_path) if '.' not in f]
downloaded = []
for m in file_ids:
    files_in_folder = os.listdir(movies_path/m)
    if len(files_in_folder) > 0:
        complete = True
        for f in files_in_folder:
            if '.aria2' in f:
                complete = False
                print(m)
                print(imdb_df.loc[imdb_df['imdb_id'] == m, ['title', 'year', 'imdb_id']])
        if complete:
            downloaded.append(m)

do_not_download = downloaded + prep_movies
len(do_not_download)

# +
not_downloaded_df = torrent_info_df\
                        .loc[torrent_info_df['top_250_rank'] <= 150]\
                        .loc[~torrent_info_df['imdb_id'].isin(do_not_download)].sample(frac=1.0)

not_downloaded_df.head()

# + code_folding=[]
shutil.rmtree(str(batch_folder))
Path(batch_folder).mkdir(parents=True, exist_ok=True)

# Create folder with movie segmentation, to run download in batches
step = 5
for i in range(0, not_downloaded_df.shape[0], step):
    if i + step > len(not_downloaded_df):
        batch_df = not_downloaded_df.iloc[i:,:]
    else:
        batch_df = not_downloaded_df.iloc[i:(i+step), :]
        
    # Save file to allow aria to multidownload
    csv_df = batch_df.loc[:, ['imdb_id', 'magnet']]
        
    output_folder = f'/Users/adautobrazdasilvaneto/Documents/ergo/silence_in_movies/data/movies_raw'

    csv_df.loc[:, 'comand'] = csv_df.apply(lambda x: '{}\n dir={}/{}'.format(x['magnet'], output_folder, x['imdb_id']), axis=1)
    csv_df.loc[:, ['comand']]\
        .to_csv(batch_folder/f'{i//5}.txt', 
                header=None, index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")


# +
# # Remove empty folders
# batches = [f for f in os.listdir(batch_folder) if '.' not in f]

# for b in batches:
#     print(b)
#     folder = batch_folder/f'{b}/movies_raw'
#     ids = [f for f in os.listdir(folder) if '.' not in f]
    
#     for i in ids:
#         if len(os.listdir(folder/i)) == 0:
#             shutil.rmtree(folder/i) # If so, delete it