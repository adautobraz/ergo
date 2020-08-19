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

# infos_folder = Path(path)/'data/infos'
# Path(infos_folder).mkdir(parents=True, exist_ok=True)

# analysis_folder = Path(path)/'analysis'
# Path(analysis_folder).mkdir(parents=True, exist_ok=True)

os.chdir(path)
# -
pd.set_option('max_columns', None)
pd.set_option('max_rows', 20)

# # HAIM

artist = 'HAIM'

# ## Data Prep

# + [markdown] heading_collapsed=true
# ### Setup

# + hidden=true
# Download data from Spotify
artist_discography_df = songs_info.get_artist_full_discography_df(artist)
artist_albums = songs_info.get_valid_albums(artist_discography_df)
Path(infos_folder/artist).mkdir(parents=True, exist_ok=True)

artist_albums.to_csv(infos_folder/'{}/spotify_infos.csv'.format(artist), index=True)

# + hidden=true
# Download data from Genius

not_found = lyrics_info.download_artist_lyrics(infos_folder, artist_albums, artist)

# In case of error!
# if not_found:
#     songs_not_found = artist_albums.loc[artist_albums.index.isin(not_found)]
#     display(songs_not_found)
#     correct_lyrics(songs_not_found.index.tolist(), ['3 am'], artist)

# + hidden=true
# Gather data from Genius, run sentiment analysis
lyrics_df = lyrics_info.get_lyrics_df(infos_folder, artist)

# + hidden=true
discography_df = artist_albums.join(lyrics_df.set_index('spotify_id').drop(columns=['artist']), how='left')
discography_df.to_csv(infos_folder/'{}/discography_info.csv'.format(artist), index=True)

discography_df.head()
# -

# ### Load Data

discography_df = pd.read_csv(infos_folder/'{}/discography_info.csv'.format(artist))
display(discography_df.head(3))
discography_df['album_name'].unique()

# + [markdown] heading_collapsed=true
# ## Days Are Gone

# + [markdown] hidden=true
# ### Setup

# + hidden=true
album = 'Days Are Gone'

# + hidden=true
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)

# + hidden=true
album_colors = get_album_color(discography_df, album, artist)

# + hidden=true
color = album_colors[0].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# + hidden=true
# Cabin
# Fira Sans
# Inter

# + [markdown] hidden=true
# ### Album context graphs

# + hidden=true
graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

# + hidden=true
#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5])#), ['up', 'up', 'down', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + code_folding=[] hidden=true
#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'up', 'down', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5])#, ['down', 'up', 'right', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['left', 'down', 'up', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.4], ['down', 'up', 'up', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# + hidden=true
# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['down', 'down', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'left', 'down', 'down', 'up', 'right'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'down', 'up', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + code_folding=[] hidden=true
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5])#, ['up', 'down', 'down', 'right', 'left', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5], ['up', 'down', 'up', 'right', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.4], ['up', 'down', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + [markdown] hidden=true
# ### Discography context graphs

# + hidden=true
feature_params = get_features_params('en', 'Discography', 'per album')

# + hidden=true
graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

# + hidden=true
## Check highlights
get_album_highlights(discography_df, feature_columns, album)

# + hidden=true
graph_type = 'bar_'
features = 'tempo'
for f in ['danceability', 'energy', 'sentiment_score']:
    fig = plot_album_feature_comparison(discography_df, feature_columns, f, album, rgb, feature_params)
    write(fig, graph_type + 'discography_highlight_{}'.format(f), album, artist)

# + [markdown] hidden=true
# ### Format images

# + hidden=true
pad_all_graphs(artist, album, rgb)
# -

# ## Something To Tell You

# ### Setup

album = 'Something To Tell You'

# +
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)
# -

album_colors = get_album_color(discography_df, album, artist)

color = album_colors[2].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# +
# Cabin
# Fira Sans
# Inter

# + [markdown] heading_collapsed=true
# ### Album context graphs

# + hidden=true
graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

# + hidden=true
#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['down', 'up', 'up', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'up', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['left', 'down', 'up', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.4], ['up', 'up', 'down', 'up', 'right', 'down'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# + hidden=true
# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.3], ['up', 'right', 'down', 'up', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['down', 'up', 'down', 'right', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'down', 'up', 'down', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + code_folding=[] hidden=true
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.6], ['up', 'up', 'down', 'up', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'up', 'right', 'down', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'down', 'up', 'down', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + [markdown] heading_collapsed=true
# ### Discography context graphs

# + hidden=true
feature_params = get_features_params('en', 'Discography', 'per album')

# + hidden=true
graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

# + hidden=true
## Check highlights
highlights = get_album_highlights(discography_df, feature_columns, album)

# + hidden=true
graph_type = 'bar_'
for f in highlights['variable'].tolist():
    fig = plot_album_feature_comparison(discography_df, feature_columns, f, album, rgb, feature_params)
    if fig:
        write(fig, graph_type + 'discography_highlight_{}'.format(f), album, artist)
# -

# ### Format images

pad_all_graphs(artist, album, rgb)
generate_tracklist(35, discography_df, artist, album, color)

# ## Women In Music, pt.III

# ### Setup

discography_df['album_name'].unique()

album = 'Women In Music Pt. III'

# +
feature_columns = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'loudness', 'mode', 'sentiment_score']

std_df =  prep_album_data(discography_df, album, feature_columns)
# -

album_colors = get_album_color(discography_df, album, artist)

color = album_colors[0].rgb
rgb = 'rgb({},{},{})'.format(color.r, color.g, color.b) 

# + [markdown] heading_collapsed=true
# ### Album context graphs

# + hidden=true
graph_type = 'line_'
feature_params = get_features_params('en', 'Album', 'track by track')

# + hidden=true
#Danceability
feature = 'danceability'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'up', 'up', 'down', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Energy
feature = 'energy'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'down', 'up', 'down', 'down', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Uniqueness
feature = 'uniqueness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5], ['up', 'down', 'up', 'down', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
#Valence
feature = 'valence'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.12,0.5], ['down', 'up', 'up', 'right', 'left', 'down'])
write(fig, graph_type + feature, album, artist)
#plot(fig)

# + hidden=true
# Speechiness
feature = 'speechiness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1, 0.5])#, ['up', 'right', 'down', 'up', 'up', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Acousticness
feature = 'acousticness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.1,0.5], ['up', 'down', 'down', 'up', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Instrumentalness
feature = 'instrumentalness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.5], ['down', 'up', 'right', 'down', 'down', 'up'])
write(fig, graph_type + feature, album, artist)

# + code_folding=[] hidden=true
# Tempo
feature = 'tempo'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.5], ['up', 'down', 'down', 'right', 'up', 'up'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Loudness
feature = 'loudness'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.11, 0.5], ['down', 'up', 'up', 'down', 'down', 'down'])
write(fig, graph_type + feature, album, artist)

# + hidden=true
# Lyrics emotion
feature = 'sentiment_score'
params = feature_params[feature]
fig = plot_feature_line(std_df, params, feature, 1, rgb, [0.15, 0.5], ['up', 'down', 'up', 'up', 'left', 'up'])
write(fig, graph_type + feature, album, artist)

# + [markdown] heading_collapsed=true
# ### Discography context graphs

# + hidden=true
feature_params = get_features_params('en', 'Discography', 'per album')

# + hidden=true
graph_type = 'line_'
fig = plot_discography_evolution(discography_df, artist, feature_columns, rgb, 'all')
fig = update_fig(fig)
fig.update_xaxes(tickfont_size=16)
write(fig, graph_type + 'discography_journey', album, artist, True)

# + hidden=true
## Check highlights
highlights = get_album_highlights(discography_df, feature_columns, album)
display(highlights.head())

# + hidden=true
graph_type = 'bar_'
for f in highlights['variable'].tolist():
    fig = plot_album_feature_comparison(discography_df, feature_columns, f, album, rgb, feature_params)
    if fig:
        write(fig, graph_type + 'discography_highlight_{}'.format(f), album, artist)
# -

# ### Format images

pad_all_graphs(artist, album, rgb)
generate_tracklist(25, discography_df, artist, album, color)
