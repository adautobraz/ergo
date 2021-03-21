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
import imdb
import pandas as pd
from pathlib import Path
import plotly.express as px
import numpy as np

data_path = Path('./data')

# + [markdown] heading_collapsed=true
# # IMDB top 250 list

# + hidden=true
ia = imdb.IMDb()
top = ia.get_top250_movies()

# + hidden=true
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

# + hidden=true
imdb_top_250_df.to_csv(data_path/'imdb_top250_movies.csv', index=False)
# -

# # Analysis

imdb_top_250_df = pd.read_csv(data_path/'imdb_top250_movies.csv')
imdb_top_250_df.head(1)

px.histogram(imdb_top_250_df, x='year')

# +
df = imdb_top_250_df.copy()
df.loc[:, 'batch'] = np.floor(df['year']/5)*5

df = df.groupby('batch', as_index=False).agg({'title':'count'})
px.bar(df, x='batch', y='title')
# -

import math

df = imdb_top_250_df.copy()
min_year = int(df['year'].min())
df.loc[:, 'batch'] = pd.qcut(df['year'], 25, duplicates='raise')
df.loc[:, 'batch'] = df['batch'].apply(lambda x: f"{math.floor(x.left) + 1:.0f}_{math.floor(x.right):.0f}" if x.left != min_year else f"{min_year:.0f}_{x.right:.0f}")
df['batch'].value_counts().sort_index()

df.to_csv(data_path/'imdb_top250_movies.csv', index=False)


