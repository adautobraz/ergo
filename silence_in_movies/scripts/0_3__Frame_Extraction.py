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
import cv2
from sources.common_functions import *

data_path = Path('./data')

# +
movies_raw_path = data_path/'movies_raw'
movies_prep_path = data_path/'movies_prep'

movie_file_dict = get_movie_file_dict(movies_raw_path)
movies_features_to_extract = get_movies_status(movies_raw_path, movies_prep_path)
movies_to_extract = [movies_raw_path/k for k, v in movies_features_to_extract.items() if 'images' in v]
# -

movies_to_extract

# Loop through movie files, save their audio on folder prep
for m in movies_to_extract:

    movie_id = str(m).split('/')[-1]
    movie_file = movie_file_dict[movie_id]
    print(movie_id)

    movie_prep_folder = movies_prep_path/f"{movie_id}"
    Path(movie_prep_folder).mkdir(parents=True, exist_ok=True)

    movie_images_folder = movie_prep_folder/'images/'
    Path(movie_images_folder).mkdir(parents=True, exist_ok=True)

    pathIn = str(movie_file)
    pathOut = str(movie_images_folder)
    
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    seconds = 10
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*seconds*1000))    # added this line 
        success,image = vidcap.read()
        timer = int(count*seconds)
        if count % 30 == 0:
            print(count)
        if success:
            cv2.imwrite( pathOut + "/frame_%d.jpg" % timer, image)     # save frame as JPEG file
            count = count + 1


