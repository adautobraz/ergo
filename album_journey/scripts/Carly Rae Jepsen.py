# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# # Setup

# +
import pandas as pd
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('..')

from ergo_utilities import songs_info, lyrics_info

from general_functions import *

path = "/Users/adautobrazdasilvaneto/Documents/ergo/album_journey/"

infos_folder = Path(path)/'data/infos'
Path(infos_folder).mkdir(parents=True, exist_ok=True)

analysis_folder = Path(path)/'analysis'
Path(analysis_folder).mkdir(parents=True, exist_ok=True)

os.chdir(path)

setup(analysis_folder, infos_folder)
# -
pd.set_option('max_columns', None)
pd.set_option('max_rows', 20)

# # Carly Rae Jepsen

artist = 'Carly Rae Jepsen'

# ## Data Prep

# ### Setup

# +
# Download data from Spotify
artist_discography_df = songs_info.get_artist_full_discography_df(artist)
artist_albums = songs_info.get_valid_albums(artist_discography_df)
Path(infos_folder/artist).mkdir(parents=True, exist_ok=True)

artist_albums.to_csv(infos_folder/'{}/spotify_infos.csv'.format(artist), index=True)

# +
# Download data from Genius

not_found = lyrics_info.download_artist_lyrics(infos_folder, artist_albums, artist)

# In case of error!
# if not_found:
#     songs_not_found = artist_albums.loc[artist_albums.index.isin(not_found)]
#     display(songs_not_found)
#     lyrics_info.correct_lyrics(infos_folder, songs_not_found.index.tolist(), ['e mo tion'], artist)
# -

# Gather data from Genius, run sentiment analysis
lyrics_df = lyrics_info.get_lyrics_df(infos_folder, artist)

# +
discography_df = artist_albums.join(lyrics_df.set_index('spotify_id').drop(columns=['artist']), how='left')
discography_df.to_csv(infos_folder/'{}/discography_info.csv'.format(artist), index=True)

discography_df.head()
# -

# ### Load Data

discography_df = pd.read_csv(infos_folder/'{}/discography_info.csv'.format(artist))
display(discography_df.head(3))
discography_df['album_name'].unique()

# ## Emotion

# ### Setup

album = 'Emotion'

# +
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)
# -

album_colors = get_album_color(discography_df, album, artist)

color = album_colors[3].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# +
# Cabin
# Fira Sans
# Inter
# -

# ### Album context graphs

graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

# + code_folding=[]
#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15,0.5], ['down', 'up', 'down', 'left', 'up', 'down'])
write(fig, graph_type + feature, album, artist)
# -

#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.13,0.6], ['down', 'down', 'up', 'right', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['left', 'down', 'up', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.13,0.5], ['up', 'down', 'up', 'left', 'up', 'up'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['down', 'down', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['down', 'left', 'up', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'down', 'up', 'up', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + code_folding=[]
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.14, 0.5], ['up', 'down', 'up', 'down', 'left', 'down'])
write(fig, graph_type + feature, album, artist)
# -

# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.4], ['down', 'down', 'up', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['left', 'up', 'down', 'up', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# ### Discography context graphs

feature_params = get_features_params('en', 'Discography', 'per album')

graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

## Check highlights
get_album_highlights(discography_df, feature_columns, 'Emotion')

graph_type = 'bar_'
for f in ['sentiment_score']:
    fig = plot_album_feature_comparison(discography_df, feature_columns, f, album, rgb, feature_params)
    write(fig, graph_type + 'discography_highlight_{}'.format(f), album, artist)

# ### Format images

pad_all_graphs(artist, album, rgb)

# ## Dedicated

# ### Setup

album = 'Dedicated'

# +
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)
# -

album_colors = get_album_color(discography_df, album, artist)

color = album_colors[2].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# ### Album context graphs

graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.12,0.5], ['up', 'down', 'up', 'up', 'left', 'down'])
write(fig, graph_type + feature, album, artist)
# plot(fig)

#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.13,0.6], ['down', 'left', 'up', 'down', 'right', 'up'])
write(fig, graph_type + feature, album, artist)


#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['down', 'up', 'down', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['down', 'up', 'left', 'up', 'down', 'down'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#), ['down', 'down', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)


# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11,0.5], ['up', 'left', 'down', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)


# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['up', 'down', 'up', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)


# + code_folding=[]
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5])# ['up', 'down', 'down', 'right', 'left', 'up'])
write(fig, graph_type + feature, album, artist)
# -


# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['up', 'up', 'down', 'up', 'down', 'right'])
write(fig, graph_type + feature, album, artist)


# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5], ['down', 'up','down', 'right', 'down', 'down'])
write(fig, graph_type + feature, album, artist)


# ### Discography context graphs

feature_params = get_features_params('en', 'Discography', 'per album')

graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

## Check highlights
get_album_highlights(discography_df, feature_columns, album)

# +
# graph_type = 'bar_'
# fig = plot_album_feature_comparison(discography_df, feature_columns, 'key', album, rgb)
# write(fig, graph_type + 'discography_highlight', album, artist)
# -

# ### Prepare image

# + code_folding=[]
pad_all_graphs(artist, album, rgb)
# -

# ## Dedicated Side B

# ### Setup

album = 'Dedicated Side B'

# +
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)
# -

album_colors = get_album_color(discography_df, album, artist)

color = album_colors[0].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# ### Album context graphs

graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.12,0.5])#, ['up', 'down', 'up', 'up', 'left', 'down'])
write(fig, graph_type + feature, album, artist)
# plot(fig)

#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.13,0.6])#, ['down', 'left', 'up', 'down', 'right', 'up'])
write(fig, graph_type + feature, album, artist)


#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['down', 'up', 'down', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.12,0.5], ['down', 'up', 'up', 'left', 'down', 'up'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.5], ['up', 'down', 'up', 'down', 'right', 'down'])
write(fig, graph_type + feature, album, artist)

# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11,0.5])#), ['up', 'left', 'down', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['down', 'up', 'down', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)


# + code_folding=[]
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5])# ['up', 'down', 'down', 'right', 'left', 'up'])
write(fig, graph_type + feature, album, artist)
# -

# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.5], ['left', 'up', 'down', 'up', 'down', 'down'])
write(fig, graph_type + feature, album, artist)

# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5], ['left', 'down','up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# ### Discography context graphs

feature_params = get_features_params('en', 'Discography', 'per album')

graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

# +
## Check highlights
highlights = get_album_highlights(discography_df, feature_columns, album)
display(highlights.head())

graph_type = 'bar_'
for f in highlights['variable'].tolist():
    fig = plot_album_feature_comparison(discography_df, feature_columns, f, album, rgb, feature_params)
    if fig:
        write(fig, graph_type + 'discography_highlight_{}'.format(f), album, artist)
# -

# ### Prepare image

# + code_folding=[]
pad_all_graphs(artist, album, rgb)
