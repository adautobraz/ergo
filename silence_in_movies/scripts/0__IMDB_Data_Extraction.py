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
# import json
import imdb
import pandas as pd
# import os 
# import time
# import numpy as np
# from pyYify import yify
from pathlib import Path
# import aria2p
# from ast import literal_eval
# import plotly.express as px

data_path = Path('./data')
# -

# # IMDB top 250 list

ia = imdb.IMDb()
top = ia.get_top250_movies()

# +
movies_info = []

for i in range(0, len(top)):
    if i % 50 == 0:
        print(i)
        
    movie_obj = top[i]
    ia.update(movie_obj, info = ['main', 'critic reviews']) 

    code = top[i].movieID
    
    keys = ['title', 'year', 'rating', 'runtime', 'genres', 'cover url', 'directors', 'top 250 rank',
            'votes', 'cast', 'color info', 'original air date', 'plot outline', 'box office', 'metascore'
           ]

    info_dict = {k.replace(' ', '_'): movie_obj[k] for k in keys if k in movie_obj}

    info_dict['imdb_code'] = code
    
    info_dict['directors'] = [d['name'] for d in info_dict['directors']]
    info_dict['cast'] = [d['name'] for d in info_dict['cast']]    

    movies_info.append(info_dict)
    
    
imdb_top_250_df = pd.DataFrame(movies_info)

imdb_top_250_df.loc[:, 'imdb_id'] = 'tt' + imdb_top_250_df['imdb_code']

imdb_top_250_df.head(3)

# +
# imdb_top_250_df.to_csv(data_path/'imdb_top250_movies.csv', index=False)
