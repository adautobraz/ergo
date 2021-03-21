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
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path 
from pythonopensubtitles.opensubtitles import OpenSubtitles
from pythonopensubtitles.utils import File
import json
import time
import pysrt
import pandas as pd



from sources.common_functions import *

data_path = Path('./data')
with open('../.config', 'r') as r:
    creds = json.load(r)['opensubtitles']
# -

ost = OpenSubtitles() 
ost.login(creds['user'], creds['key'])

# # Download Subtitles

# +
movies_raw_path = data_path/'movies_raw'
movies_prep_path = data_path/'movies_prep'

movie_file_dict = get_movie_file_dict(movies_raw_path)
movies_features_to_extract = get_movies_status(movies_raw_path, movies_prep_path)
movies_to_extract = [k for k, v in movies_features_to_extract.items() if 'subtitle' in v]
movies_to_extract
# -

# Loop through movie files, save their audio on folder prep
for movie_id in movies_to_extract:
    
    try:
        movie_file = movie_file_dict[movie_id]
        print(movie_id)

        f = File(str(movie_file))
        data = ost.search_subtitles([{'sublanguageid': 'eng', 'imdbid': movie_id[2:], 'moviehash': f.get_hash(), 
                                      'moviebytesize': f.size}])
        id_subtitle_file = data[0].get('IDSubtitleFile')

        movie_prep_folder = movies_prep_path/f"{movie_id}"
        Path(movie_prep_folder).mkdir(parents=True, exist_ok=True)

        movie_subtitle_folder = movie_prep_folder/'subtitle/'
        Path(movie_subtitle_folder).mkdir(parents=True, exist_ok=True)

        ost.download_subtitles([id_subtitle_file], output_directory=movie_subtitle_folder, extension='srt')
        time.sleep(1)
    
    except:
        print('\tErro')


# # Convert Subtitle

movies_with_subtitle = [f for f in os.listdir(movies_prep_path) if not f.startswith('.') if os.path.exists(movies_prep_path/f'{f}/subtitle/')]

# +
all_subs = []
for m in movies_with_subtitle:
    folder_subtitle = movies_prep_path/f'{m}/subtitle/'
    file_subtitle = [f for f in os.listdir(folder_subtitle) if not f.startswith('.')][0]
    if 'srt' in file_subtitle:
        subs = pysrt.open(f'{folder_subtitle/file_subtitle}')

        episode_subtitles = []
        for i in range(0, len(subs)):
            line = subs[i]
            ep_dict = {
                'text':line.text,
                'start_h':line.start.hours,
                'start_min':line.start.minutes,
                'start_s':line.start.seconds,
                'start_ms':line.start.milliseconds,
                'end_h':line.end.hours,
                'end_min':line.end.minutes,
                'end_s':line.end.seconds,
                'end_ms':line.end.milliseconds,
                'position':i
            }
            episode_subtitles.append(ep_dict)

        episode_df = pd.DataFrame(episode_subtitles)
        episode_df.loc[:, 'imdb_id'] = m
        all_subs.append(episode_df)

            
all_subs_df = pd.concat(all_subs, axis=0).set_index('imdb_id')
# -

all_subs_df.to_csv(data_path/'prep/all_subtitles.csv')


