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
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path
from moviepy.editor import *
from sources.common_functions import *
import pandas as pd
import autoscrub
import json

data_path = Path('./data')
# -

# # Extract movie audio as MP3

# +
movies_raw_path = data_path/'movies_raw'
movies_prep_path = data_path/'movies_prep'

movie_file_dict = get_movie_file_dict(movies_raw_path)
movies_features_to_extract = get_movies_status(movies_raw_path, movies_prep_path)
movies_to_extract = [movies_raw_path/k for k, v in movies_features_to_extract.items() if 'audio' in v]
# -

movies_to_extract

# Loop through movie files, save their audio on folder prep
for m in movies_to_extract:

    movie_id = str(m).split('/')[-1]
    movie_file = movie_file_dict[movie_id]
    print(movie_id)

    movie_prep_folder = movies_prep_path/f"{movie_id}"
    Path(movie_prep_folder).mkdir(parents=True, exist_ok=True)

    movie_audio_folder = movie_prep_folder/'audio'
    Path(movie_audio_folder).mkdir(parents=True, exist_ok=True)

    output_movie = movie_audio_folder/"movie_audio.mp3"
    
    video = VideoFileClip(str(movie_file))

    audio = video.audio
    audio.write_audiofile(str(output_movie))

    audio.close()
    audio.close()

# # Extract silence from audios

movies = [m for m in os.listdir(movies_prep_path) if not m.startswith('.')]

# m = movies[0]
for m in movies:
    silences_file = movies_prep_path/f'{m}/audio/silences.csv'

    if not os.path.exists(silences_file):
        print(m)
        audiofile = movies_prep_path/f'{m}/audio/movie_audio.mp3'

        autoscrub.suppress_ffmpeg_output(True)
        results = autoscrub.getSilences('./{}'.format(audiofile), 
                              input_threshold_dB=-40.0, 
                              silence_duration=1.0, 
                              save_silences=False)

        df = pd.DataFrame(results)
        df.to_csv(silences_file, index=False)
