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

# +
import pandas as pd
import numpy as np
import plotly.express as px
import ast
import umap
import umap.plot
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.graph_objects as go


pd.set_option('max_columns', None)
pd.set_option('max_rows', 100)

# +
import chart_studio

username = 'adautobraz.neto' # your username
api_key = 'mNl7JNqWOFCJAwZ8pEmV' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)


# -

# ## Layout Definition

# +
# Font: Lato, Karla, Asap, Poppins
# Colors: 
# Font size: title, legend

# +
def plot(fig):
    fig.update_layout(
        font_family='Asap', 
        template='plotly_white', 
        font_size=14)
    fig.show()
    
def write(fig, name):
    fig.write_image("images_en/{}.png".format(name), scale=1)
    

def write_square(fig, name):
    fig.update_layout(width=800, height=800)
    fig.write_image("images_en/{}.png".format(name), scale=0.8)


# -

# ## Data Prep

# +
# Load full dataset, raw
pop_divas_df_raw = pd.read_csv('./data/pop_divas_df_valid_albums.csv')
pop_divas_df_raw.head()

# Create columns
pop_divas_df_raw.loc[:, 'feat'] = (pop_divas_df_raw['artists_on_track'] > 1).astype(int)
pop_divas_df_raw.loc[:, 'explicit'] = pop_divas_df_raw['explicit'].astype(int)
pop_divas_df_raw.loc[:, 'id'] = pop_divas_df_raw['uri'].apply(lambda x: x.split(':')[2])

pop_divas_df_raw.loc[:, 'year'] = pop_divas_df_raw['album_release_date'].str[:4].astype(int)

pop_divas_df_raw.loc[:, 'duration_min'] = pop_divas_df_raw['duration_ms']/(60000)
pop_divas_df_raw.loc[:, 'remix'] = pop_divas_df_raw['name'].str.lower().str.contains('remix', case=False)

pop_divas_df_raw.loc[:, 'album_adj'] = pop_divas_df_raw['album_name'].str.extract(r'([^\(\)\[\]]*)').iloc[:, 0].str.strip()
pop_divas_df_raw.loc[:, 'song_adj'] = pop_divas_df_raw['name'].str.split(' - ', expand=True).iloc[:, 0].str.strip().str.replace('ß', 'B')
pop_divas_df_raw.loc[:, 'detail'] = ~pop_divas_df_raw['name'].str.split(' - ', expand=True).iloc[:, 1].isnull()

pop_divas_df_raw.loc[:, 'usual_track'] = (~pop_divas_df_raw['remix'])


drop_columns = ['external_urls', 'uri', 'type', 'uri.1', 'track_href', 'analysis_url', 
                'album_type', 'paranthesis', 'special_ed', #'track_number',
                'album_release_date_precision', 'artists_on_track', 'album_release_date'
               ]

pop_divas_df = pop_divas_df_raw\
                    .sort_values(by=['artist','year', 'song_adj', 'detail'])\
                    .drop_duplicates(subset=['artist', 'song_adj'], keep='first')\
                    .loc[pop_divas_df_raw['usual_track']]\
                    .drop(columns=drop_columns)\
                    .dropna(axis=0)\
                    .set_index('id')

## Lyrics
# Load data
lyrics_df_raw = pd.read_csv('./data/lyrics_info.csv')

lyrics_df = lyrics_df_raw\
                .loc[lyrics_df_raw['id'].isin(pop_divas_df.index.tolist())]\
                .set_index('id')

# Data Prep
lyrics_df.loc[:, 'nrc_emotions_dict'] =  lyrics_df['nrc_emotions'].fillna('{}').apply(lambda x: ast.literal_eval(x))
lyrics_df.loc[:, 'nrc_emotions_relative'] =  lyrics_df.apply(lambda x: {k: v/x['nrc_emotions_total_words'] for k, v in x['nrc_emotions_dict'].items()}, axis=1)

lyrics_df.loc[:, 'duration_min'] =  pop_divas_df['duration_min']
lyrics_df.loc[:, 'words_per_min'] =  lyrics_df['total_word_count']/lyrics_df['duration_min']
lyrics_df.loc[:, 'sentiment_label'] =  lyrics_df['flair_sentiment'].str[1:-1].str.split('(', expand=True).iloc[:, 0].fillna('').str.strip()
lyrics_df.loc[:, 'sentiment_probability'] =  lyrics_df['flair_sentiment'].str[1:-1].str.split('(', expand=True).iloc[:, 1].str[:-1].fillna(0)

lyrics_df.loc[:, 'sentiment'] =  lyrics_df['sentiment_label'].apply(lambda x: 1 if x == 'POSITIVE' else -1 if x=='NEGATIVE' else 0)

lyrics_df.loc[:, 'sentiment_score'] = (lyrics_df['sentiment_probability'].astype(float) - 0.5)
lyrics_df.loc[:, 'sentiment_score'] = lyrics_df['sentiment_score']*lyrics_df['sentiment']
lyrics_df.head()

## Final dataset
pop_divas_df = pop_divas_df\
                .join(lyrics_df.loc[:, ['sentiment_score', 'total_word_count']], how='left').dropna(axis=0)

pop_divas_df.head()

# +
all_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_min', 'key', 'sentiment_score', 'feat', 'explicit']


feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_min', 'key', 'sentiment_score']


range_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_min', 'key']

# + code_folding=[]
# Album df
albums_mean = pop_divas_df.groupby(['album_name', 'artist', 'year'], as_index=False).agg({c:'mean' for c in feature_columns})

# Artist df, per song
artist_per_song = pop_divas_df.groupby(['artist'], as_index=False).agg({c:'mean' for c in feature_columns})

artist_per_album = albums_mean.groupby(['artist'], as_index=False).agg({c:'mean' for c in feature_columns})


# Artist df, per song, pondered by popularity
pondered_by_popularity = pop_divas_df.copy()

pondered_by_popularity.loc[:, 'total_popularity'] = pondered_by_popularity.groupby(['artist'])['song_popularity'].transform(sum)
pondered_by_popularity.loc[:, 'importance'] = pondered_by_popularity['song_popularity']/pondered_by_popularity['total_popularity']

for f in feature_columns:
    pondered_by_popularity.loc[:, f] = pondered_by_popularity[f]*pondered_by_popularity['importance']

artist_per_song_imp = pondered_by_popularity.groupby(['artist'], as_index=False).agg({c:'mean' for c in feature_columns})

# +
params = {}

params['variable'] = feature_columns
params['artist'] = pop_divas_df.sort_values(by='artist_popularity', ascending=False)['artist'].unique().tolist()

params

color_list = ['#4CE0B3', '#DA5552', '#EDAE49', '#006DA3', '#820263', '#FFADC6', '#8E7DBE', '#42CAFD', '#366D3E']

color_dict = {params['artist'][i]:color_list[i] for i in range(0, len(params['artist']))}
color_dict
# -

# ## Songs visualization

# + [markdown] heading_collapsed=true
# ### Features distribution

# + hidden=true
df = pop_divas_df.copy().rename(columns={'name':'song', 'album_name':'album'})
melt = pd.melt(df, id_vars=['song','artist', 'album', 'song_popularity'], value_vars=feature_columns)


fig = px.histogram(melt, histnorm='percent',
                   facet_col_wrap=3, facet_col='variable', x='value', nbins=2000, height=700,
                   category_orders=params
                  )

fig.update_yaxes(matches=None, showticklabels=True)
fig.update_xaxes(matches=None, showticklabels=True)

fig.show()

# + [markdown] heading_collapsed=true
# ### Scatter UMAP

# + hidden=true
reducer = umap.UMAP(n_neighbors = 15)

# + hidden=true
X = pop_divas_df.copy().dropna(axis=0).loc[:, feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# + hidden=true
X_plot = pd.DataFrame(reducer.fit_transform(X_scaled))

# + hidden=true
X_plot.columns = ['dim0', 'dim1']
X_plot.index = pop_divas_df.index
X_plot.loc[:, 'artist'] = pop_divas_df['artist']
X_plot.loc[:, 'song'] = pop_divas_df['name']
X_plot.loc[:, 'popularity'] = pop_divas_df['song_popularity']
X_plot.loc[:, 'album'] = pop_divas_df['album_name']
X_plot.loc[:, 'feat'] = pop_divas_df['feat']
X_plot.loc[:, 'explicit'] = pop_divas_df['explicit']

# + hidden=true
max_x = np.ceil(max([abs(X_plot['dim0'].max()), abs(X_plot['dim0'].min())])/10)*10
max_y = np.ceil(max([abs(X_plot['dim1'].max()), abs(X_plot['dim1'].min())])/10)*10
range_x = [-max_x, max_x]
range_y = [-max_y, max_y]

fig = px.scatter(X_plot, x='dim1', y='dim0', hover_name='song', color='artist',
        size='popularity', size_max=8, opacity=0.8)
fig.update_layout(
    xaxis_title='', 
    yaxis_title='',
    title='Músicas por artista (Proximidade = Semelhança)' 
)

plot(fig)

# + hidden=true
import chart_studio.plotly as py
py.plot(fig, filename = 'umap_all_songs', auto_open=True)

# + [markdown] heading_collapsed=true
# ### Scatter TSNE

# + hidden=true
X = pop_divas_df.copy().dropna(axis=0).loc[:, feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_plot = pd.DataFrame(TSNE(n_components=2).fit_transform(X_scaled))

X_plot.columns = ['dim{}'.format(c) for c in X_plot]
X_plot.index = pop_divas_df.index
X_plot.loc[:, 'artist'] = pop_divas_df['artist']
X_plot.loc[:, 'song'] = pop_divas_df['name']
X_plot.loc[:, 'popularity'] = pop_divas_df['song_popularity']
X_plot.loc[:, 'album'] = pop_divas_df['album_name']
X_plot.loc[:, 'feat'] = pop_divas_df['feat']
X_plot.loc[:, 'explicit'] = pop_divas_df['explicit']

X_plot.loc[:, 'pop'] = X_plot['popularity']//20
X_plot.head()

# + hidden=true
max_x = np.ceil(max([abs(X_plot['dim0'].max()), abs(X_plot['dim0'].min())])/10)*10
max_y = np.ceil(max([abs(X_plot['dim1'].max()), abs(X_plot['dim1'].min())])/10)*10
range_x = [-max_x, max_x]
range_y = [-max_y, max_y]

fig = px.scatter(X_plot, x='dim1', y='dim0', hover_name='song', color='artist',
        size='pop', size_max=8, opacity=0.8, 
        hover_data=['album', 'feat', 'explicit'])

fig.update_layout(
    xaxis_title='', 
    yaxis_title='',
    title='Músicas por artista (Proximidade = Semelhança)' 
)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)

plot(fig)

# + code_folding=[0] hidden=true
# ## Músicas com a média
# artist_songs_tsne = X_plot.copy().reset_index()
# artist_songs_tsne.loc[:, 'is_song'] = True

# artist_avg_tsne = X_plot.groupby(['artist'], as_index=False).agg({'dim0':'mean', 'dim1':'mean', 'song':'count'})
# artist_avg_tsne.loc[:, 'is_song'] = False
# artist_avg_tsne.loc[:, 'song'] = artist_avg_tsne['artist']


# columns = ['artist', 'dim0', 'dim1', 'is_song', 'song']

# all_points = pd.concat([artist_songs_tsne.loc[:, columns], 
#                         artist_avg_tsne.loc[:, columns]], axis=0).reset_index()
# all_points.head()

# all_points.loc[:, 'size'] = all_points['is_song'].apply(lambda x: 1 if x else 10)
# # all_points.loc[:, 'opacity'] = all_points['is_song'].apply(lambda x: 0.5 if x else 1)

# fig = px.scatter(all_points, x='dim0', y='dim1', color='artist', size='size', 
#                  hover_name='song',symbol='is_song')

# # fig.update_traces(marker=dict(
# #     size=all_points['size'].values, 
# #     opacity=all_points['opacity'].values
# # ))

# plot(fig)

# + code_folding=[] hidden=true
fig = px.scatter(X_plot, x='dim1', y='dim0', hover_name='song', animation_frame='artist', color='album',
           hover_data=['album'], size='pop', size_max=8, opacity=0.8)

fig.update_layout(
    xaxis_title='', 
    yaxis_title='',
    title='Músicas por artista (Proximidade = Semelhança)' 
)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)

plot(fig)

# + [markdown] heading_collapsed=true
# ### Top/Bottom song per feature

# + hidden=true
## Filter or ponder by popularity?

# + code_folding=[0] hidden=true
##All songs visualization

df = pop_divas_df.copy().rename(columns={'name':'song', 'album_name':'album'})

# Option only usual songs
df = df.loc[df['usual_track']]

for c in range_columns:
    df.loc[:, c] = (df[c]/df[c].mean()) - 1

df.head()

melt = pd.melt(df, id_vars=['song','artist', 'album', 'song_popularity'], value_vars=range_columns)


melt.loc[:, 'max_pos'] = melt.groupby('variable')['value'].transform(max)
melt.loc[:, 'max_neg'] = melt.groupby('variable')['value'].transform(min)

melt.loc[:, 'size_pos'] = melt['value']/melt['max_pos']
melt.loc[:, 'size_neg'] = melt['value']/melt['max_neg']

melt.loc[:, 'size'] = melt.apply(lambda x: max(x['size_pos'], x['size_neg']), axis=1)

# Filter for only max and min songs
melt = melt.loc[melt['size'] == 1].sort_values(by='song_popularity', ascending=False).drop_duplicates(subset=['variable', 'value'])


fig = px.bar(melt, x='value', y='variable', color='artist', category_orders=params,
            height=800, text='song', range_x=[-60,60])

plot(fig)

# + [markdown] heading_collapsed=true
# ### Top/Bottom song per feature, per artist

# + code_folding=[0] hidden=true
# All songs visualization
df = pop_divas_df.copy().rename(columns={'name':'song', 'album_name':'album'})

# Only usual tracks
df = df.loc[df['usual_track']]

melt = pd\
        .melt(df, id_vars=['song','artist', 'album', 'song_popularity'], value_vars=range_columns)\
        .sort_values(by=['artist', 'song_popularity'], ascending=False)

melt.loc[:, 'artist_mean'] = melt.groupby(['artist', 'variable'])['value'].transform('mean')
melt.loc[:, 'artist_median'] = melt.groupby(['artist', 'variable'])['value'].transform('median')
melt.loc[:, 'value_rel'] = melt['value']/melt['artist_mean'] - 1
melt.loc[:, 'value_rel_abs'] = np.abs(melt['value']/melt['artist_mean'] - 1)


melt.loc[:, 'pos'] = melt.groupby(['artist', 'variable'])['value'].rank('first', ascending=False)
melt.loc[:, 'total_songs'] = melt.groupby(['artist', 'variable'])['pos'].transform('max')

melt.loc[melt['pos'] <= melt['total_songs']//2, 'pos_rel'] = melt['pos']
melt.loc[melt['pos'] > melt['total_songs']//2, 'pos_rel'] = melt['pos'] - (melt['total_songs'] + 1)

melt.loc[:, 'top_bottom'] = 'top'
melt.loc[melt['pos_rel'] < 0, 'top_bottom'] = 'bottom'

# Filter for only max and min songs
melt = melt\
        .loc[melt['pos_rel'].abs() <= 1]\
        .sort_values(by=['song_popularity', 'variable'], ascending=False)\
        .drop_duplicates(subset=['artist','variable', 'value'])


melt.loc[melt['top_bottom'] == 'top', 'text'] =  melt['song'] + ' (+'
melt.loc[melt['top_bottom'] == 'bottom', 'text'] = melt['song'] + ' (-'

fig = px.bar(melt, y='variable', color='top_bottom',
            height=800, text='text', animation_frame='artist',
#              range_x=[-80,80], x='value_rel',
            x='value_rel_abs',  barmode='group', log_x=True, range_x=[0.1, 100],
             category_orders = params
            )

fig.update_layout(
    title='Músicas de destaque de cada artista, em relação a sua média',
    yaxis_title='Feature', 
    xaxis_title='Distância da média'
)

fig.update_traces(texttemplate='%{text}%{x:.1f}x)')

plot(fig)
# -

# ## Artist Visualization

# ### Artists Amount

# + code_folding=[]
## Full view
group = pop_divas_df.groupby(['artist'], as_index=False).agg({'name':'count', 'album_name':'nunique'})
group.columns = ['artist', 'Songs', 'Albums']
group = pd.melt(group, id_vars='artist', value_vars=['Songs', 'Albums']).sort_values(by=['variable', 'value'])


fig = px.bar(group, y='artist', color='artist', x='value', facet_col='variable', color_discrete_map=color_dict)
fig.update_xaxes(matches=None, showticklabels=False, title='')
fig.update_xaxes(row=1, col=2, range=[0, 220])
fig.for_each_annotation(lambda a: a.update(text = a.text.split('=')[1]))
fig.update_layout(title='Total albums and songs, per artist', showlegend=False, yaxis_title='Artist')
fig.update_traces(texttemplate= '%{x:0f}', textposition='inside')

plot(fig)
# -

pop_divas_df\
        .groupby(['artist', 'album_name'])\
        .agg({'name':'count', 'year':'min'})\
        .sort_values(by=['artist', 'year'])#pop_divas_df['artist'].contains.head()

# + code_folding=[0]
# Hard work
# group = pop_divas_df.groupby(['artist'], as_index=False).agg({'name':'count', 'album_name':'nunique', 'year':['min', 'max']})
# group.columns = ['artist', 'songs', 'albums', 'first', 'last']
# group.loc[:, 'delta'] = group['last'] - group['first']
# group.loc[:, 'work'] = group['albums']/group['delta'] 
# group
# -

write(fig, 'total_albums_songs_en')

py.plot(fig, filename = 'total_albums_songs_en', auto_open=True)

# + [markdown] heading_collapsed=true
# ### Tests

# + hidden=true
# ## Artist features mean value, through songs

# melt = pd.melt(artist_per_song, id_vars='artist', value_vars=feature_columns)

# fig = px.bar(melt, color='artist', y='artist', x='value', orientation='h',
#              facet_col='variable', facet_col_wrap=5)

# fig.update_xaxes(matches=None)
# fig.update_layout(legend_orientation='h')
# fig.show()


# # ## Artist features mean value, through albums

# artist_per_album = albums_mean.groupby(['artist'], as_index=False).agg({c: 'mean' for c in columns})

# melt = pd.melt(artist_mean, id_vars='artist', value_vars=columns)

# #artist_mean_melt.head()
# fig = px.bar(melt, color='artist', y='artist', x='value', orientation='h',
#              facet_col='variable', facet_col_wrap=5)

# fig.update_xaxes(matches=None)
# fig.update_layout(legend_orientation='h')
# fig.show()


# Scaling songs_df and then grouping
# Too packed for some features, best scale after groupby

# Scaling artist df
# df = artist_per_song
# scale_album = StandardScaler()
# divas_scaled = pd.DataFrame(scale_album.fit_transform(df.loc[:, feature_columns]))
# divas_scaled.columns = feature_columns
# divas_scaled.loc[:, 'artist'] = df['artist'].values

# fig = px.scatter(melt, x='value', y='variable', color='artist', text='text',
#                 size='size', range_x=[-4,4], height=800, size_max=15)
# fig.update_traces(textposition='top center')
# fig.update_layout(yaxis_title='Feature', xaxis_title='Distância do artista da média geral',
#                  legend_orientation='h'
#                  )
# #fig.update_xaxes(showticklabels=False)
# fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='Black')

# + [markdown] heading_collapsed=true
# ### Artist distribution

# + code_folding=[0] hidden=true
# Ridgeline, per artist
df = pop_divas_df.copy()

scaler = StandardScaler()
songs_scaled = pd.DataFrame(scaler.fit_transform(df.loc[:, feature_columns]))

songs_scaled.columns = feature_columns
songs_scaled.loc[:, 'artist'] = df['artist'].values
songs_scaled.loc[:, 'name'] = df['name'].values
songs_scaled.loc[:, 'album_name'] = df['album_name'].values

melt = pd.melt(songs_scaled, id_vars=['name','artist', 'album_name'], value_vars=feature_columns)
melt = melt.rename(columns={'name':'song', 'album_name':'album'})

fig = px.violin(melt, x='value', y='artist', animation_frame='variable', range_x=[-5, 5],
#                 facet_col_wrap=3, facet_col='variable', 
                color='artist', height=800, hover_name='song', hover_data=['album'])

fig.update_traces(side='positive', width=4)

plot(fig)

# + [markdown] heading_collapsed=true
# ### Artist Scatter

# + hidden=true
df = pop_divas_df.copy()

scaler = StandardScaler()
songs_scaled = pd.DataFrame(scaler.fit_transform(df.loc[:, feature_columns]))

songs_scaled.columns = feature_columns
songs_scaled.loc[:, 'artist'] = df['artist'].values

artist_per_song_scaled = songs_scaled.groupby(['artist']).agg({c:'mean' for c in feature_columns})
artist_per_song_scaled.head()

# + hidden=true
# df = artist_per_song_scaled

# X_plot = pd.DataFrame(TSNE(n_components=2).fit_transform(df))

# X_plot.columns = ['dim{}'.format(c) for c in X_plot]
# X_plot.index = df.index

# X_plot.loc[:, 'total_songs'] = pop_divas_df.groupby(['artist']).count().iloc[:,0]

# X_plot.reset_index(inplace=True)

# X_plot.head()

# fig = px.scatter(X_plot, x='dim1', y='dim0', color='artist', size='total_songs', 
#                  opacity=0.8, text='artist')

# fig.update_traces(textposition='top center')
# fig.update_layout

# plot(fig)

# + code_folding=[] hidden=true
# Variable per artist

# df = artist_per_song_scaled

# X_plot = pd.DataFrame(TSNE(n_components=2).fit_transform(df))

# X_plot.columns = ['dim{}'.format(c) for c in X_plot]
# X_plot.index = df.index

# X_plot.loc[:, 'total_songs'] = pop_divas_df.groupby(['artist']).count().iloc[:,0]

# X_all_features = pd.concat([artist_per_song.set_index('artist'), X_plot], axis=1)
# for c in feature_columns:
#     X_all_features.loc[:, c] = X_all_features[c]/(X_all_features[c].min())

# X_all_features.head()
# melt = pd.melt(X_all_features.reset_index(), id_vars = ['artist', 'dim0', 'dim1', 'total_songs'],
#                value_vars=feature_columns).dropna(axis=0)

# melt = melt.loc[melt['variable'] != 'explicit']

# # melt.loc[:, 'max'] = melt.groupby(['variable'])['value'].transform(max)
# melt.loc[:, 'value_log'] = np.log2(melt['value']) + 1
# px.histogram(melt, x='value_log')

# fig = px.scatter(melt, x='dim1', y='dim0', color='artist', size='value_log', opacity=0.8,
#                 animation_frame='variable')
# plot(fig)

# + [markdown] heading_collapsed=true
# ### Top/Bottom artist in each feature 

# + code_folding=[0] hidden=true
# Compare to the average of group (percentage)
df = artist_per_song.copy().set_index('artist')

for c in df.columns.tolist():
    df.loc[:, c] = df[c]/(df[c].mean()) - 1
    
df = df.reset_index()

melt = pd.melt(df, id_vars='artist', value_vars=range_columns)


melt.loc[:, 'max_pos'] = melt.groupby('variable')['value'].transform(max)
melt.loc[:, 'max_neg'] = melt.groupby('variable')['value'].transform(min)

melt.loc[:, 'size_pos'] = melt['value']/melt['max_pos']
melt.loc[:, 'size_neg'] = melt['value']/melt['max_neg']

melt.loc[:, 'size'] = melt.apply(lambda x: max(x['size_pos'], x['size_neg']), axis=1)

melt.loc[:, 'text'] = ""
melt.loc[melt['size'] == 1, 'text'] = melt['artist']

melt = melt.loc[melt['size'] == 1]


fig = px.bar(melt, x='value', y='variable', color='artist', text='text',
            height=800, range_x=[-3,6], color_discrete_map=color_dict)
fig.update_layout(
    title='Artistas destaques de cada categoria',
    yaxis_title='Feature', 
    xaxis_title='Distância da média'
)
fig.update_traces(textposition='outside')

fig.update_xaxes(tickvals=[-4, -2, 2, 4], ticksuffix="x")

plot(fig)
# -

# ### Representativeness

# + code_folding=[0]
# Data Prep 
from scipy.spatial import distance

all_artists = pop_divas_df['artist'].unique().tolist()

scale = StandardScaler()
songs_scaled = pd.DataFrame(scale.fit_transform(pop_divas_df.loc[:, feature_columns]))
songs_scaled.index = pop_divas_df.index
songs_scaled.columns = feature_columns
songs_scaled.loc[:, 'artist'] = pop_divas_df['artist']
songs_scaled.loc[:, 'song'] = pop_divas_df['name']

median_artist = songs_scaled.groupby(['artist']).agg({c:'mean' for c in feature_columns})
artist_array = median_artist.values
artist_songs = songs_scaled.loc[:, feature_columns].values

dist_song_artist = pd.DataFrame(distance.cdist(artist_songs, artist_array, 'euclidean'))
dist_song_artist.columns = median_artist.index.tolist()
dist_song_artist.loc[:, 'artist'] = pop_divas_df['artist'].values
dist_song_artist.loc[:, 'album'] = pop_divas_df['album_name'].values
dist_song_artist.loc[:, 'song'] = pop_divas_df['name'].values
dist_song_artist.loc[:, 'usual_track'] = pop_divas_df['usual_track'].values
dist_song_artist.loc[:, 'song_popularity'] = pop_divas_df['song_popularity'].values

dist_song_artist = dist_song_artist.loc[dist_song_artist['usual_track']]

dist_song_artist.head(3)

# + code_folding=[0]
# Song distance from artist average

all_closest_songs = []
for a in dist_song_artist['artist'].unique().tolist():
    df = dist_song_artist.loc[:, [a, 'song', 'artist', 'album', 'song_popularity']]
    closest_song = df.sort_values(by=a)   
    closest_song.loc[:, 'distance'] = closest_song[a]
    closest_song.loc[:, 'artist_ref'] = a
    
    all_closest_songs.append(closest_song.drop(columns=[a]))

all_closest_songs = pd.concat(all_closest_songs, axis=0).sort_values(by='song_popularity', ascending=False)

all_closest_songs.loc[:, 'own_artist'] = all_closest_songs['artist'] == all_closest_songs['artist_ref']
all_closest_songs.loc[:, 'all_songs_ranking'] = all_closest_songs.groupby(['artist_ref'])['distance'].rank('first')
all_closest_songs.loc[:, 'own_songs_ranking'] = all_closest_songs.groupby(['artist_ref', 'own_artist'])['distance'].rank('first')
all_closest_songs.loc[:, 'total_songs'] = all_closest_songs.groupby(['artist_ref', 'own_artist'])['distance'].transform('count')

all_closest_songs.head(3)
# -

# #### Artists most representative song

# + code_folding=[0]
# Songs that most represent the artist song features

top_own_songs = all_closest_songs.loc[all_closest_songs['own_artist']]

top_own_songs = top_own_songs.loc[(top_own_songs['own_songs_ranking'] == 1) 
                                  | (top_own_songs['own_songs_ranking'] == top_own_songs['total_songs'])]

top_own_songs.loc[:, 'max'] = top_own_songs.groupby('artist_ref')['distance'].transform(max)
top_own_songs.loc[:, 'min'] = top_own_songs.groupby('artist_ref')['distance'].transform(min)

top_own_songs.loc[top_own_songs['distance'] == top_own_songs['max'], 'order'] = 'max'
top_own_songs.loc[top_own_songs['distance'] == top_own_songs['min'], 'order'] = 'min'

top_own_songs.loc[:, 'color'] = top_own_songs['distance'].astype(str)

top_own_songs.head()

fig = px.bar(top_own_songs, y='artist_ref', x='distance', text='song', color='order',
             barmode='group', height=600, category_orders=params,
           hover_name='song', hover_data=['album'])

fig.update_layout(
    title='Músicas mais e menos representativas de cada artista'
)

plot(fig)

# + code_folding=[0]
# Songs that most represent the artist song features

closest_songs = all_closest_songs\
                    .sort_values(by=['artist_ref', 'own_artist', 'distance'])\
                    .drop_duplicates(subset=['artist_ref', 'own_artist'])


closest_songs.loc[:, 'text'] = closest_songs['song'] + ' - ' + closest_songs['artist']

closest_songs.sort_values(by='artist_ref').head()

fig = px.bar(closest_songs, y='artist_ref', x='distance', color='own_artist', 
             text='text', height=600, hover_name='song', hover_data=['album'], 
             barmode='group', category_orders=params)

plot(fig)

# + [markdown] heading_collapsed=true
# #### General metrics, per artist

# + code_folding=[0] hidden=true
# # Closest songs equal to size of artist discography, checking artist own song represantation (normalized)
# # Problematic due to difference in discography size

# total_songs = pop_divas_df.groupby('artist').agg({'name': 'count'}).reset_index()

# artist_closest_discography = pd.merge(left=all_closest_songs, right=total_songs, left_on='artist_ref', right_on='artist', suffixes=('', '_merge'))

# artist_closest_discography = artist_closest_discography.loc[artist_closest_discography['all_songs_ranking'] <= artist_closest_discography['name']]

# group = artist_closest_discography\
#             .groupby(['artist_ref', 'artist', 'name'], as_index=False)\
#             .agg({'song':'count'})

# group.loc[:, 'percentage'] = group['song']/group['name']

# group_own = group.loc[group['artist'] == group['artist_ref']]

# fig = px.bar(group_own.sort_values(by='percentage'), x='percentage', y='artist', orientation='h', color='artist')
# plot(fig)

# + code_folding=[0] hidden=true
## Artists with more songs simlitar to other artists 

amount = 50
top_amount = all_closest_songs.loc[all_closest_songs['all_songs_ranking'] <= amount]

grouped = top_amount.groupby(['artist_ref', 'artist'], as_index=False).agg({'song':'count'})

total_songs = pop_divas_df.groupby('artist').agg({'name': 'count'})
grouped = grouped.set_index('artist').join(total_songs, how='left').reset_index().rename(columns={'index':'artist_ref'})

grouped.loc[:, 'total'] = grouped.groupby('artist_ref')['song'].transform('sum')
grouped.loc[:, 'percentage'] = grouped['song']/grouped['total']

artist_order = grouped\
                    .groupby('artist', as_index=False)\
                    .agg({'percentage':'sum'})\
                    .sort_values(by='percentage', ascending=False)\
                    ['artist'].unique().tolist()


fig = px.area(grouped, x='artist_ref', y='percentage', color='artist', 
              height=600, line_shape='linear'
              ,category_orders={'artist':artist_order}
             )

plot(fig)

#
grouped.loc[:, 'percentage_norm'] = grouped['percentage']/grouped['name']
grouped_own = grouped.loc[grouped['artist_ref'] == grouped['artist']]

fig = px.bar(grouped_own.sort_values(by='percentage_norm'), x='percentage_norm', y='artist', 
             orientation='h', color='artist', title='Participação própria no top 50 músicas mais parecidas com a média da artista, normalizado')
plot(fig)

# + code_folding=[0] hidden=true
# Influence (top 50 closest songs, normalized) 

total_songs = pop_divas_df.groupby('artist').agg({'name': 'count'})

influence_df = top_amount\
                .groupby(['artist'])\
                .agg({'song':'count'})\
                .join(total_songs, how='left')\
                .reset_index()

influence_df.loc[:, 'influence'] = influence_df['song']/(amount*top_amount['artist_ref'].nunique()*influence_df['name'])

influence_df.head()

fig = px.bar(influence_df.sort_values(by='influence', ascending=False), y='artist', 
             color='artist', x='influence', title='Participação no top 50 de todas as artistas, normalizado')
plot(fig)

# + [markdown] hidden=true
# - Katy Perry: a com mais músicas que podem descrever as demais artistas (versatilidade)
# - Rihanna: a com menos músicas que podem descrever as demais artistas (unicidade)

# + code_folding=[0] hidden=true
# Distribuição músicas artista
top_own_songs = all_closest_songs.loc[all_closest_songs['own_artist']]

# top_own_songs = top_own_songs.loc[(top_own_songs['own_songs_ranking'] == 1) 
#                                   | (top_own_songs['own_songs_ranking'] == top_own_songs['total_songs'])]

fig = px.violin(top_own_songs, y='distance', color='artist_ref', points='all', facet_col='artist_ref',
                height=600, category_orders=params, hover_name='song', hover_data=['album'])

fig.update_traces(width=5, side='positive')
fig.update_layout(
    title='Distribuição das músicas do artista em relação à média do seu catálogo',
    legend_orientation='h'
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
# fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))


plot(fig)
# + code_folding=[0] hidden=true
# Average distance from own representation 
df = top_own_songs.groupby(['artist_ref'], as_index=False).agg({'distance':'mean'})
fig = px.bar(df.sort_values(by='distance'), y='artist_ref', x='distance', color='artist_ref', 
             color_discrete_map=color_dict)
fig.update_layout(title='Discography versatility, per artist', 
                  showlegend=False, yaxis_title='Artist', xaxis_title='')
fig.update_xaxes(tickmode='array', tickvals=[0, 3.5], ticktext =['More cohesive', 
                                                                 'More versatile'], 
                 title='Discography versatility')
# fig.update_yaxes(tickmode='array', ticktext=limits, 
#                  tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)

# + hidden=true
write(fig, 'discography_versatility_en')

# + [markdown] code_folding=[] hidden=true
# Beyoncé: Discografia mais versátil  
# Katy Perry: Discografia mais coesa

# + [markdown] code_folding=[] heading_collapsed=true
# ### Feature cross

# + hidden=true
# Data Prep
songs_scaled_disc = songs_scaled.copy()
songs_scaled_disc.loc[:, 'upbeat_sound'] = (songs_scaled_disc['valence'] > 0)
songs_scaled_disc.loc[:, 'positive_lyrics'] = (songs_scaled_disc['sentiment_score'] > 0)
songs_scaled_disc.loc[:, 'dancing_song'] = (songs_scaled_disc['danceability'] > 0)

songs_scaled_disc.head()


# + code_folding=[] hidden=true
def plot_axis_cross(feature1, feature2):
    # Intetion match and Valence x Sentiment
    group = songs_scaled_disc\
                .groupby(['artist', feature1, feature2], as_index=False)\
                .agg({'song':'count'})

    group.loc[:, 'intention_match'] = (group[feature2] == group[feature1]).astype(int)

    group.loc[:, 'total_songs'] = group.groupby(['artist'])['song'].transform('sum')
    group.loc[:, 'song_percentage'] = 100*group['song']/group['total_songs']

    group.head()

    fig = px.histogram(group, x='artist', y='song', histfunc='sum', histnorm='percent',barnorm='percent',
                      color='intention_match', barmode='group')
    fig.update_layout(title='Intention Match: {} x {}'.format(feature1, feature2))
    plot(fig)

    fig = px.bar(group, x='artist', y='song_percentage', facet_col=feature1,  facet_row=feature2,
                color='artist', color_discrete_map=color_dict)
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(title='{} x {}'.format(feature1, feature2), showlegend=False)
    fig.update_yaxes(range=[0, 60])
    plot(fig)

    coherence = group.groupby(['artist'], as_index=False).agg({'song_percentage':'std'})
    fig = px.bar(coherence.sort_values(by='song_percentage'), x='artist', y='song_percentage', color='artist', color_discrete_map=color_dict)
    fig.update_layout(title='Variance among {} x {}'.format(feature1, feature2))
    plot(fig)


# + hidden=true
plot_axis_cross('upbeat_sound', 'positive_lyrics')

# + hidden=true
plot_axis_cross('dancing_song', 'positive_lyrics')

# + hidden=true
feature1 = 'upbeat_sound'
feature2 = 'positive_lyrics'

group = songs_scaled_disc\
            .groupby(['artist', feature1, feature2], as_index=False)\
            .agg({'song':'count'})

group.loc[:, 'intention_match'] = (group[feature2] == group[feature1]).astype(int)

group.loc[:, 'total_songs'] = group.groupby(['artist'])['song'].transform('sum')
group.loc[:, 'song_percentage'] = 100*group['song']/group['total_songs']

df = group.loc[(group[feature1])
                 &(~group[feature2])]

df.head()

fig = px.bar(df.sort_values(by='song_percentage'), y='artist', x='song_percentage',
            color='artist', color_discrete_map=color_dict)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_layout(title='Happy songs with negative lyrics, per artist', 
                  showlegend=False,
                  yaxis_title='Artist',
                  xaxis_title='Percentage of artist discography (%)'
                 )
fig.update_xaxes(range=[0, 30])
plot(fig)
write(fig, 'fake_happy_songs_en')

# + hidden=true
feature1 = 'dancing_song'
feature2 = 'positive_lyrics'

group = songs_scaled_disc\
            .groupby(['artist', feature1, feature2], as_index=False)\
            .agg({'song':'count'})

group.loc[:, 'intention_match'] = (group[feature2] == group[feature1]).astype(int)

group.loc[:, 'total_songs'] = group.groupby(['artist'])['song'].transform('sum')
group.loc[:, 'song_percentage'] = 100*group['song']/group['total_songs']

df = group.loc[(group[feature1])
                 &(~group[feature2])]

df.head()

fig = px.bar(df.sort_values(by='song_percentage'), y='artist', x='song_percentage',
            color='artist', color_discrete_map=color_dict)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_layout(title='Dancing songs with negative lyrics, per artist', 
                  showlegend=False,
                  yaxis_title='Artist',
                  xaxis_title='Percentage of artist discography (%)'
                 )
fig.update_xaxes(range=[0, 40])
plot(fig)
write(fig, 'dance_crying_songs_en')

# + hidden=true
feature1 = 'dancing_song'
feature2 = 'positive_lyrics'

group = songs_scaled_disc\
            .groupby(['artist', feature1, feature2], as_index=False)\
            .agg({'song':'count'})

group.loc[:, 'intention_match'] = (group[feature2] == group[feature1]).astype(int)

group.loc[:, 'total_songs'] = group.groupby(['artist'])['song'].transform('sum')
group.loc[:, 'song_percentage'] = 100*group['song']/group['total_songs']

df = group.loc[(group[feature1])
                 &(group[feature2])]

df.head()

fig = px.bar(df.sort_values(by='song_percentage'), y='artist', x='song_percentage',
            color='artist', color_discrete_map=color_dict)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_layout(title='Dancing songs with positive lyrics, per artist', 
                  showlegend=False,
                  yaxis_title='Artist',
                  xaxis_title='Percentage of artist discography (%)'
                 )
fig.update_xaxes(range=[0, 65])
plot(fig)
write(fig, 'happy_dance_songs_en')
# -

# ## Album visualization

# ### Data Prep

# + code_folding=[]
# Data Prep
all_artists = artist_per_album['artist'].unique().tolist()

scale = StandardScaler()
songs_scaled = pd.DataFrame(scale.fit_transform(pop_divas_df.loc[:, all_features]))
songs_scaled.index = pop_divas_df.index
songs_scaled.columns = all_features
songs_scaled.loc[:, 'song'] = pop_divas_df['name']
songs_scaled.loc[:, 'album'] = pop_divas_df['album_adj']
songs_scaled.loc[:, 'artist'] = pop_divas_df['artist']
songs_scaled.loc[:, 'year'] = pop_divas_df['year']

albums_scaled = songs_scaled.groupby(['album', 'artist', 'year'], as_index=False).agg({c:'mean' for c in all_features})

albums_scaled.loc[:, 'album_order'] = albums_scaled.groupby(['artist'])['year'].rank().astype(int)
albums_scaled.loc[:, 'total_albums'] = albums_scaled.groupby(['artist'])['album'].transform('count')


albums_scaled.loc[:, 'album_order_norm'] =  (albums_scaled['album_order']-1)/(albums_scaled['total_albums']-1)
# albums_scaled.loc[albums_scaled['album_order'] == 1, 'album_order_norm'] = 0
# albums_scaled.loc[albums_scaled['album_order'] == albums_scaled['total_albums'], 'album_order_norm'] = 1


albums_popularity = pop_divas_df.groupby(['album_name', 'artist'], as_index=False).agg({'song_popularity':'mean'})

albums_scaled.head()#loc[albums_scaled['artist'] == 'Taylor Swift'].head(10)

# + [markdown] heading_collapsed=true
# ### Scatter

# + code_folding=[] hidden=true
# Albums scattered 
import umap

mapper = umap.UMAP(n_neighbors=10)
X_plot = pd.DataFrame(mapper.fit_transform(albums_scaled.loc[:, feature_columns]))
X_plot.columns=['dim0', 'dim1']

X_plot = pd.concat([albums_scaled.loc[:, ['artist', 'album', 'year']], X_plot], axis=1)
X_plot.loc[:, 'album_popularity'] = albums_popularity['song_popularity']
X_plot.loc[:, 'album_order'] = X_plot.groupby(['artist'])['year'].rank().astype(int)
X_plot.loc[:, 'size'] = 1

fig = px.scatter(X_plot, x='dim0', y='dim1', color='artist', size='size', size_max=15,
                 hover_name='album')
plot(fig)
# -

# ### Mean Shift - Past discography

# + code_folding=[0]
# Code

artist_average = []

all_artists = albums_scaled['artist'].unique().tolist()

for a in all_artists:
    artist_df = albums_scaled.loc[albums_scaled['artist'] == a]
    total_albums = artist_df.shape[0]
    for i in range(1, total_albums + 1):
        if i> 1:
            current_work_df = artist_df.loc[artist_df['album_order'] < i]
            artist_rep = current_work_df.groupby(['artist']).agg({c:'mean' for c in feature_columns}).iloc[0].to_dict()
            artist_rep['artist'] = a
            artist_rep['till_album'] = i      
            artist_average.append(artist_rep)
        
                    
mean_till_album = pd.DataFrame(artist_average)

mean_till_album.loc[:, 'album'] = mean_till_album['artist'] + '-' + mean_till_album['till_album'].astype(str)

mean_till_album = mean_till_album.set_index('album').loc[:, feature_columns]

#Calculate distances from previous works

artist_avg_albums = mean_till_album.values
artist_albums_array = albums_scaled.loc[:, feature_columns].values

dist_album_progress = pd.DataFrame(distance.cdist(artist_albums_array, artist_avg_albums, 'euclidean'))
dist_album_progress.columns = mean_till_album.index.tolist()
dist_album_progress.loc[:, 'album_order'] = albums_scaled['album_order']
dist_album_progress.loc[:, 'artist'] = albums_scaled['artist']
dist_album_progress.loc[:, 'album'] = albums_scaled['album']


dist_album_progress = pd.melt(dist_album_progress, 
                              id_vars=['album', 'artist', 'album_order'],
                              value_vars = mean_till_album.index.tolist())

dist_album_progress.loc[:, 'artist_ref'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,0]
dist_album_progress.loc[:, 'album_ref_order'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,1 ].astype(int)

dist_album_progress = dist_album_progress\
                        .loc[
                            (dist_album_progress['artist'] == dist_album_progress['artist_ref'])
                            & (dist_album_progress['album_order'] == dist_album_progress['album_ref_order'])
                            ]


all_albums_info = albums_scaled.set_index(['album', 'artist']).loc[:, ['year', 'album_order']]
dist_album_progress = dist_album_progress.set_index(['album', 'artist']).loc[:, ['value']]

album_diff_prev = pd.concat([all_albums_info,dist_album_progress], axis=1).fillna(0).reset_index()

album_diff_prev.sort_values(by=['artist', 'album_order'], inplace=True)


# Data Prep
album_diff_prev.loc[:, 'max_value'] = album_diff_prev.groupby(['artist'])['value'].transform('max')
album_diff_prev.loc[:, 'cum_dist'] = album_diff_prev.groupby(['artist'])['value'].cumsum()
album_diff_prev.loc[:, 'order_norm'] = album_diff_prev.groupby(['artist'])['album_order'].apply(lambda x: x/x.max())
album_diff_prev.loc[:, 'dist_norm'] = album_diff_prev.groupby(['artist'])['cum_dist'].apply(lambda x: x/x.max())

album_diff_prev.loc[:,'text'] = ''
album_diff_prev.loc[album_diff_prev['value'] == album_diff_prev['max_value'] ,'text'] = album_diff_prev['album']
album_diff_prev.loc[:,'text'] = album_diff_prev['text'].str.extract(r'([^\(\)]*)').iloc[:, 0]
album_diff_prev.head()

# + code_folding=[]
# Separate views
fig = px.line(album_diff_prev, x='album_order', y='value', color='artist', facet_col='artist',
              height=700, facet_col_wrap=3,
#                 facet_col_wrap=9, width=2000, height=400, 
              hover_name='album', color_discrete_map=color_dict,
                 line_shape='spline', 
#                 trendline='ols',                  
                 text='text', range_y=[0, 4], range_x=[-2,15])

fig.update_layout(showlegend=False)
# fig.update_yaxes(showticklabels=False)
fig.update_traces(textposition='top center', texttemplate='<i>%{text}<i>', textfont_size=11)

fig.for_each_annotation(lambda a: a.update(text='{}'.format(a.text.split('=')[1])))
fig.update_annotations(dict(font_size=14))
fig.update_yaxes(title='', showticklabels=False)
fig.update_xaxes(title='')

# Add x-axis title
fig.add_annotation(
        dict(
            x=0.5,
            y=-0.1,
            showarrow=False,
            text="Album order",
            xref="paper",
            yref="paper",
            font_size=18
        )
)
# Add y-axis title
fig.add_annotation(
        dict(
            x=-0.07,
            y=0.5,
            showarrow=False,
            text="Distance from previous discography",
            textangle=-90,
            xref="paper",
            yref="paper",
            font_size=18
        )
)

plot(fig)
# -

write(fig, 'discography_evolution_en')

# +
### Specific
artist = 'Madonna'
df = album_diff_prev.loc[album_diff_prev['artist'] == artist]

df.loc[:, 'rank'] = df.groupby(['artist'])['value'].rank(method='first', ascending=False)
df.loc[df['rank'] <= 3, 'text'] = df['album']

fig = px.line(df, x='album_order', y='value', color='artist',
#                 facet_col_wrap=9, width=2000, height=400, 
              hover_name='album', color_discrete_map=color_dict,
                 line_shape='spline', 
#                 trendline='ols',                  
                 text='text', range_y=[0, 4], range_x=[0,15])

fig.update_layout(showlegend=False,
                  title='Innovation across discography (Madonna)',
                  yaxis_title='Distance from past discography', 
                  xaxis_title='Album order')
# fig.update_yaxes(showticklabels=False)
fig.update_traces(textposition='top center', texttemplate='<i>%{text}<i>', textfont_size=11)

fig.for_each_annotation(lambda a: a.update(text='{}'.format(a.text.split('=')[1])))
fig.update_annotations(dict(font_size=14))
fig.update_yaxes(showticklabels=False)

plot(fig)
write(fig, 'discography_evolution_madonna_en')

# + code_folding=[0]
# fig = px.scatter(album_diff_prev.loc[album_diff_prev['album_order'] > 1], x='album_order', y='value', 
#                  color='artist', height=700, hover_name='album',
#                  trendline='ols', facet_col='artist', facet_col_wrap=3)
# fig.for_each_annotation(lambda a: a.update(text='<b>{}<b>'.format(a.text.split('=')[1])))
# # fig.update_xaxes(matches=None)
# plot(fig)

# + code_folding=[0]
# # Artist evolution 
# fig = px.scatter(album_diff_prev, x='cum_dist', y='artist', color='artist', 
#                  text='text', size_max=20,
# #                  size='value_adj',
#                  hover_name='album')
# fig.update_traces(textposition='top center')
# fig.update_layout(
#     showlegend=False,
#     title='Evolução das artistas, em comparação à discografia anterior',
#     xaxis_title='',
#     yaxis_title=''
# )

# plot(fig)

# + code_folding=[0]
# # Inovation distribution
# avg_evolution = album_diff_prev\
#                     .loc[album_diff_prev['value'] > 0]
#                     #.groupby(['artist']).agg({'value':['mean','median']})
    
# fig = px.violin(avg_evolution, x='value', y='artist', color='artist')

# fig.update_traces(side='positive', width=3)
# plot(fig)

# fig = px.box(avg_evolution, x='artist', y= 'value', color='artist')
# plot(fig)

# + code_folding=[0]
# Inovation 
# Average inovation
df = album_diff_prev\
        .loc[album_diff_prev['album_order'] > 1]\
        .groupby(['artist'], as_index=False)\
        .agg({'value':['mean', 'median']})

df.columns = ['artist', 'value_mean', 'value_median']

fig = px.bar(df.sort_values(by='value_mean'), y='artist', x='value_mean', color='artist', 
                   title='Difference from previous works, per artist', color_discrete_map=color_dict)

fig.update_yaxes(title='Artist')
fig.update_xaxes(title='Average distance from previous discography', tickmode='array', 
                 tickvals=[0, 2], ticktext =['Less innovation',  'More innovation'])
fig.update_layout(showlegend=False)
plot(fig)
# # Median inovation 
# fig = px.bar(df.sort_values(by='value_median'), y='artist', x='value_median', color='artist', title='Inovação em relação aos trabalhos anteriores (mediana)')

# plot(fig)
# -

write(fig, 'inovation_per_artist_en')

# + [markdown] heading_collapsed=true
# ### Mean Shift - Last Album

# + code_folding=[0] hidden=true
## Code 
## Genearate artist array of last album
artist_average = []

all_artists = albums_scaled['artist'].unique().tolist()
for a in all_artists:
    artist_df = albums_scaled.loc[albums_scaled['artist'] == a]
#     display(artist_df)
    total_albums = artist_df.shape[0]
    for i in range(1, total_albums + 1):
        if i> 1:
            current_work_df = artist_df.loc[artist_df['album_order'] == i - 1]
#             display(current_work_df)
            artist_rep = current_work_df.groupby(['artist']).agg({c:'mean' for c in feature_columns}).iloc[0].to_dict()
            artist_rep['artist'] = a
            artist_rep['till_album'] = i      
            artist_average.append(artist_rep)
        
                    
mean_till_album = pd.DataFrame(artist_average)

mean_till_album.loc[:, 'album'] = mean_till_album['artist'] + '-' + mean_till_album['till_album'].astype(str)

mean_till_album = mean_till_album.set_index('album').loc[:,feature_columns]


# Get distance from follow_up

artist_avg_albums = mean_till_album.values
artist_albums_array = albums_scaled.loc[:, feature_columns].values

dist_album_progress = pd.DataFrame(distance.cdist(artist_albums_array, artist_avg_albums, 'euclidean'))
dist_album_progress.columns = mean_till_album.index.tolist()
dist_album_progress.loc[:, 'album_order'] = albums_scaled['album_order']
dist_album_progress.loc[:, 'artist'] = albums_scaled['artist']
dist_album_progress.loc[:, 'album'] = albums_scaled['album']


dist_album_progress = pd.melt(dist_album_progress, 
                              id_vars=['album', 'artist', 'album_order'],
                              value_vars = mean_till_album.index.tolist())

dist_album_progress.loc[:, 'artist_ref'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,0]
dist_album_progress.loc[:, 'album_ref_order'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,1 ].astype(int)

# Get only distance from current to past
dist_album_progress = dist_album_progress\
                        .loc[
                            (dist_album_progress['artist'] == dist_album_progress['artist_ref'])
                            & (dist_album_progress['album_order'] == dist_album_progress['album_ref_order'])
                            ]


# Join info of first album
all_albums_info = albums_scaled.set_index(['album', 'artist']).loc[:, ['year', 'album_order']]
dist_album_progress = dist_album_progress.set_index(['album', 'artist']).loc[:, ['value']]

album_diff_past = pd.concat([all_albums_info,dist_album_progress], axis=1).fillna(0).reset_index()

album_diff_past.sort_values(by=['artist', 'album_order'], inplace=True)


# Data Prep
album_diff_past.loc[:, 'max_value'] = album_diff_past.groupby(['artist'])['value'].transform('max')
album_diff_past.loc[:, 'cum_dist'] = album_diff_past.groupby(['artist'])['value'].cumsum()
album_diff_past.loc[:, 'order_norm'] = album_diff_past.groupby(['artist'])['album_order'].apply(lambda x: x/x.max())
album_diff_past.loc[:, 'dist_norm'] = album_diff_past.groupby(['artist'])['cum_dist'].apply(lambda x: x/x.max())

album_diff_past.loc[:,'text'] = ''
album_diff_past.loc[album_diff_past['value'] == album_diff_past['max_value'] ,'text'] = album_diff_past['album']
album_diff_past.loc[:,'text'] = album_diff_past['text'].str.extract(r'([^\(\)]*)').iloc[:, 0]
album_diff_past.head()

# + code_folding=[0] hidden=true
# Views

# Dist from last album, points
fig = px.scatter(album_diff_past, x='cum_dist', y='artist', color='artist', 
                 text='text', hover_name='album')
fig.update_traces(textposition='top center')
fig.update_layout(
    showlegend=False,
    
)
fig.update_layout(
    showlegend=False,
    xaxis_showgrid=False
)

plot(fig)


# Dist from last album, points
fig = px.line(album_diff_past, x='cum_dist', y='artist', color='artist', 
                 text='text', hover_name='album')
fig.update_traces(textposition='top center')
fig.update_layout(
    showlegend=False,
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

plot(fig)

# Diff from last album
fig = px.line(album_diff_past, x='album_order', y='value', color='artist', 
                text='text', hover_name='album', 
                line_shape='spline',
                facet_col_wrap=3, facet_col='artist'
             )
fig.update_traces(textposition='top center')
fig.update_layout(
    showlegend=False
)

plot(fig)

# + code_folding=[0] hidden=true
# Average inovation 
df = album_diff_past.loc[album_diff_past['album_order'] > 1]
fig = px.histogram(df, x='artist', y='value', color='artist', 
                   histfunc='avg', title='Inovação em relação ao álbum anterior')

plot(fig)

# + [markdown] heading_collapsed=true
# ### Most different albums

# + code_folding=[0] hidden=true
#Data Prep
artist_per_song_scaled = songs_scaled.groupby(['artist']).agg({c: 'mean' for c in feature_columns})

dist_album_artist_avg = pd.DataFrame(distance.cdist(albums_scaled.loc[:, feature_columns], artist_per_song_scaled, 'euclidean'))

dist_album_artist_avg.columns = artist_per_song_scaled.index.tolist()

dist_album_artist_avg.loc[:, 'artist'] = albums_scaled['artist'].values
dist_album_artist_avg.loc[:, 'album'] = albums_scaled['album'].values
dist_album_artist_avg.loc[:, 'album_order'] = albums_scaled['album_order'].values

dist_album_artist_avg = pd.melt(dist_album_artist_avg, 
                                id_vars=['artist', 'album', 'album_order'],
                                value_vars=artist_per_song_scaled.index.tolist(),
                                var_name='artist_ref',
                                value_name='distance'
                               )

dist_album_artist_avg.head()

# + code_folding=[0] hidden=true
# Album distance from Artist average
own_artist_dist = dist_album_artist_avg.loc[dist_album_artist_avg['artist'] == dist_album_artist_avg['artist_ref']].copy()

own_artist_dist.loc[:, 'max'] = own_artist_dist.groupby(['artist'])['distance'].transform(max)
own_artist_dist.loc[:, 'min'] = own_artist_dist.groupby(['artist'])['distance'].transform(min)

own_artist_dist.loc[:, 'text'] = own_artist_dist.apply(lambda x: x['album'] if (x['distance'] == x['max']) | (x['distance'] == x['min']) else '', axis=1)

fig = px.scatter(own_artist_dist, y='artist', x='distance', color='artist', 
                 hover_name='album', height=600, text='text', range_x=[-0, 4])

fig.update_traces(textposition='top center')

plot(fig)


fig = px.violin(own_artist_dist, y='artist', x='distance', color='artist',
                 hover_name='album', height=600, range_x=[-0, 4])

#fig.update_traces(textposition='down center')
fig.update_traces(side='positive', width=3, points=False)
plot(fig)

# + code_folding=[0] hidden=true
# Average innovation

df = own_artist_dist.groupby(['artist_ref'], as_index=False).agg({'distance':'mean'})
fig = px.bar(df.sort_values(by='distance'), x='artist_ref', y='distance',color='artist_ref', color_discrete_map=color_dict)
fig.update_layout(title='Inovação média, por artista, por álbum')
plot(fig)

# + code_folding=[0] hidden=true
# # Albums from other artists (closeness) 
# other_artist_dist = dist_album_artist_avg#.loc[dist_album_artist_avg['artist'] != dist_album_artist_avg['artist_ref']]

# other_artist_dist.loc[:, 'max'] = other_artist_dist.groupby(['artist_ref'])['distance'].transform(max)
# other_artist_dist.loc[:, 'min'] = other_artist_dist.groupby(['artist_ref'])['distance'].transform(min)

# other_artist_dist.loc[:, 'text'] = other_artist_dist.apply(lambda x: x['album'] if (x['distance'] == x['max']) | (x['distance'] == x['min']) else '', axis=1)
# other_artist_dist.head()

# fig = px.scatter(other_artist_dist, y='artist_ref', x='distance', color='artist', 
#                  hover_name='album', height=600, text='text', range_x=[-0.1,5])

# fig.update_traces(textposition='top center')

# plot(fig)#.show()

# + hidden=true
# main_features = ['danceability', 'energy', 'sentiment_score', 'valence']

# df = melt.sort_values(by=['variable', 'album_order']).loc[melt['variable'].isin(main_features)]
# fig = px.line(df, x='album_order', y='value', color='artist', animation_frame='variable', line_shape='spline',
#              facet_col='artist', facet_col_wrap=3, height=700, hover_name='album')

# fig.update_xaxes(matches=None)
# plot(fig)

# + [markdown] heading_collapsed=true
# ### Album cohesion

# + code_folding=[0] hidden=true
#Data Prep
album_per_song_scaled = songs_scaled.groupby(['album', 'artist']).agg({c: 'mean' for c in feature_columns})

dist_songs_album_avg = pd.DataFrame(distance.cdist(songs_scaled.loc[:, feature_columns], album_per_song_scaled.values, 'euclidean'))

dist_songs_album_avg.columns = album_per_song_scaled.reset_index()['album'].tolist()

dist_songs_album_avg.loc[:, 'song'] = songs_scaled['song'].values
dist_songs_album_avg.loc[:, 'album'] = songs_scaled['album'].values
dist_songs_album_avg.loc[:, 'artist'] = songs_scaled['artist'].values
dist_songs_album_avg.loc[:, 'year'] = songs_scaled['year'].values

dist_songs_album_avg = pd.melt(dist_songs_album_avg, 
                                id_vars=['song', 'album', 'artist', 'year'],
                                value_vars=album_per_song_scaled.reset_index()['album'].tolist(),
                                var_name='album__ref',
                                value_name='distance'
                               )

dist_songs_album_avg = pd.merge(left=dist_songs_album_avg, left_on='album__ref',
                                right=album_per_song_scaled.reset_index().loc[:, ['artist', 'album']], 
                                right_on='album', suffixes=('', '_ref')).drop(columns=['album__ref'])

dist_songs_album_avg = dist_songs_album_avg.loc[dist_songs_album_avg['album'] == dist_songs_album_avg['album_ref']]

dist_songs_album_avg.head()

album_cohesion = dist_songs_album_avg.groupby(['artist', 'album', 'year'], as_index=False).agg({'distance':'mean'})

album_cohesion.loc[:, 'album_order'] = album_cohesion.groupby(['artist'])['year'].rank(method='first').astype(int)

album_cohesion.loc[:, 'total_albums'] = album_cohesion.groupby(['artist'])['album'].transform('count')

album_cohesion.loc[:, 'album_order_norm'] = (album_cohesion['album_order']-1)/(album_cohesion['total_albums']-1)

album_cohesion.head()

# + code_folding=[0] hidden=true
# Plot average cohesion per album
fig = px.histogram(album_cohesion, x='artist', y='distance', histfunc='avg', color='artist')
plot(fig)

fig = px.line(album_cohesion.sort_values(by='album_order_norm'), 
              x='album_order_norm', y='distance', color='artist', 
             line_shape='spline', hover_name='album', facet_col='artist', facet_col_wrap=3
             )
plot(fig)

# + [markdown] heading_collapsed=true
# ### Albums highlights

# + code_folding=[] hidden=true
## Albums per characteristic 

melt = pd.melt(albums_scaled, id_vars=['artist', 'album', 'year', 'album_order'], value_vars=feature_columns)

melt.loc[:, 'max'] = melt.groupby(['variable'])['value'].transform(max)
melt.loc[:, 'min'] = melt.groupby(['variable'])['value'].transform(min)

melt.loc[:, 'text'] = melt.apply(lambda x: x['album'] if x['value'] in [x['max'], x['min']] else '', axis=1)

# fig = px.scatter(melt, y='variable', x='value', color='artist', hover_name='album', text='text')
# fig.update_traces(textposition='top center')
# fig.update_layout(showlegend=False)

# plot(fig)

# + hidden=true
main_features = ['danceability', 'energy', 'sentiment_score', 'valence', 'speechiness', 'tempo', 'duration_min',
                'instrumentalness']

df = melt.sort_values(by=['variable', 'album_order']).loc[melt['variable'].isin(main_features)]
fig = px.line(df, x='album_order', y='value', color='artist', animation_frame='variable', line_shape='spline',
             facet_col='artist', facet_col_wrap=3, height=700, hover_name='album')

fig.update_xaxes(matches=None)
plot(fig)


# + [markdown] code_folding=[]
# ### Focus view per feature

# + code_folding=[0]
def plot_feature_focus(artists, feature, position):
    
    x='album_order_norm'
    grey_color='#cccccc'
    
    df_albums = albums_scaled.copy()
    
    df_albums.loc[:, 'focus_artist'] = df_albums['artist'].isin(artists)

    df_albums = df_albums.sort_values(by=[x, 'focus_artist'], ascending=True)
    
    df_albums.loc[:, feature] = np.round(df_albums[feature], 4)
    
    # Get max and min album
    min_max_artist = df_albums\
                        .loc[df_albums['focus_artist']]\
                        .groupby(['artist']).agg({feature:['min', 'max']})
    
    min_max_artist.columns = ['min', 'max']
    
    min_max_albums = df_albums.set_index('artist').join(min_max_artist, how='inner').reset_index()
    min_max_albums = min_max_albums\
                        .loc[(min_max_albums[feature] == min_max_albums['min'])
                            | (min_max_albums[feature] == min_max_albums['max'])
                        ]\
                        .sort_values(by='album_order', ascending=True)\
                        .drop_duplicates(subset=['artist', feature], keep='last')\
                        .loc[:, 'album'].tolist()
    
    
    df_albums.loc[:, 'text'] = ''
    df_albums.loc[df_albums['album'].isin(min_max_albums), 'text'] = df_albums['album']

    
    df_albums.loc[:, 'size'] = df_albums['text'].apply(lambda x: 10 if x != '' else 7)
  
    fig = go.Figure()    
    
    i = df_albums['artist'].nunique() - 1
    for a in df_albums['artist'].unique().tolist():
        df = df_albums.loc[df_albums['artist'] == a]
        if a in artists:
            fig.add_trace(go.Scatter(x=df[x], y=df[feature], mode='lines+markers+text', 
                name=a, marker=dict(color=color_dict[a], symbol=i, size=df['size']), 
                text=df['text'], textposition=position, hovertext=df['album'], texttemplate='<i>%{text}<i>',
                textfont=dict(size=12),
                line=dict(color=color_dict[a], shape='spline'),
            ))
            
        
        else:
            fig.add_trace(go.Scatter(x=df[x], y=df[feature], mode='markers', hovertext=df['album'],
                name=a, marker=dict(color=grey_color, symbol=i, size=df['size']), line=dict(color=grey_color, shape='spline')
            ))

        i -= 1

    fig.update_xaxes(tickmode='array', ticktext=['First album', 'Latest album'], 
                     tickvals=[0,1], range=[-0.1,1.2])
    
    delta_y = df_albums[feature].max()
    
    x_pos = -0.04
    fig.add_annotation(
            ax=x_pos,
            x=x_pos,
            ay=0.2,
            y=delta_y - 0.2,
            axref='pixel',
            xref='paper',
            ayref='y',
            yref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=1
    )
    
    fig.add_annotation(
            ax=x_pos,
            x=x_pos,
            ay=-0.2,
            y=-delta_y + 0.2,
            axref='pixel',
            xref='paper',
            ayref='y',
            yref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=1
    )
    
    return fig
# -

features_mean = songs_scaled.groupby(['artist'], as_index=False).agg({f:'mean' for f in all_features})

# #### Tempo

# +
feature = 'tempo'
limits = ['Slower beats', 'Average of artists', 'Faster beats']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Tempo, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'tempo_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Katy Perry'], feature, 'top right')
fig.update_layout(title='Tempo evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'tempo_highlights_evolution_en_top')

abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Ariana Grande'], feature, 'top center')
fig.update_layout(title='Tempo evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'tempo_highlights_evolution_en_bottom')
# -

# #### Danceability

# +
feature = 'danceability'
limits = ['Less danceable', 'Average of artists', 'More danceable']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Danceability, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'danceability_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Britney Spears'], feature, 'top right')
fig.update_layout(title='Danceability evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'danceability_highlights_evolution_en')
# -

# #### Energy

# +
feature = 'energy'
limits = ['More calm', 'Average of artists', 'More energy']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Energy, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'energy_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Lady Gaga'], feature, 'bottom center')
fig.update_layout(title='Energy evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'energy_highlights_evolution_en')

# + [markdown] heading_collapsed=true
# #### Valence

# + hidden=true
feature = 'valence'
limits = ['Sadder/Angrier<br>sound', 'Average of artists', 'Happier/Cheerful<br>sound']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Valence, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
# write(fig, 'valence_per_artist_en')

abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Britney Spears'], feature, 'middle right')
fig.update_layout(title='Valence evolution, per album', legend_orientation='v')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.1, abs_range+0.1])

plot(fig)
write(fig, 'valence_highlights_evolution_en')

# + [markdown] heading_collapsed=true
# #### Speechiness

# + code_folding=[5] hidden=true
feature = 'speechiness'
limits = ['More singing', 'Average of artists', 'More rapping']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Speechiness, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'speechiness_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Beyoncé'], feature, 'bottom center')
fig.update_layout(title='Spechiness evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'speechiness_highlights_evolution_en')
# -

# #### Duration

# +
feature = 'duration_min'
limits = ['Shorter songs', 'Average of artists', 'Longer songs']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Song duration, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'duration_per_artist_en')

abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Ariana Grande'], feature, 'bottom center')
fig.update_layout(title='Song duration evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
# -

# #### Feat

# +
feature = 'feat'
limits = ['Less feats', 'Average  ', 'More feats']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Features, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Ariana Grande'], feature, 'top right')
fig.update_layout(title='Features evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)

# + [markdown] heading_collapsed=true
# #### Explicit

# + hidden=true
feature = 'explicit'
limits = ['Less explicit', 'Average of artists', 'More explicit']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Explicitness, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'explicitness_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Rihanna'], feature, 'middle right')
fig.update_layout(title='Explicitness evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.4, abs_range+0.4])

plot(fig)
write(fig, 'explicitness_highlights_evolution_en')

# + hidden=true
albums_scaled.loc[albums_scaled['artist'] == 'Rihanna']
# -

# #### Sentiment 

# +
feature = 'sentiment_score'
limits = ['Sadder lyrics', 'Average of artists', 'Happier lyrics']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Lyrics sentiment, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'sentiment_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Taylor Swift'], feature, 'middle right')
fig.update_layout(title='Lyrics sentiment evolution, per album', legend_orientation='h')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.3, abs_range+0.3])

plot(fig)
write(fig, 'sentiment_highlights_evolution_en')
# -

# #### Acousticness

# +
feature = 'acousticness'
limits = ['Less acoustic', 'Average of artists', 'More acoustic']

fig = px.bar(features_mean.sort_values(by=feature), y='artist', x=feature, color='artist', 
             color_discrete_map=color_dict)
fig.update_layout(yaxis_title='', title='Acousticness, per artist', xaxis_title='', showlegend=False)
abs_range = features_mean[feature].abs().max()
fig.update_xaxes(ticktext=limits,
                 range=[-abs_range-0.1,abs_range+0.1], tickvals=[-abs_range, 0, abs_range])
plot(fig)
write(fig, 'acousticness_per_artist_en')


abs_range = albums_scaled[feature].abs().max()
fig = plot_feature_focus(['Mariah Carey'], feature, 'middle left')
fig.update_layout(title='Acousticness evolution, per album')
fig.update_yaxes(tickmode='array', ticktext=limits, 
                 tickvals=[-abs_range,0, abs_range], range=[-abs_range-0.2, abs_range+0.2])

plot(fig)
write(fig, 'acousticness_highlights_evolution_en')
# -

py.plot(fig, filename = 'sentiment_highlights_evol_en', auto_open=True)

# + [markdown] heading_collapsed=true
# ## Lyrics exploring

# + [markdown] heading_collapsed=true hidden=true
# ### Sentiment

# + hidden=true
df = lyrics_df.loc[lyrics_df['sentiment_label'] != '']

# + code_folding=[] hidden=true
#Average sentiment score
fig = px.histogram(df, x='artist', y='sentiment_score', histfunc='avg', color='artist')

plot(fig)


fig = px.histogram(df, y='artist', histfunc='avg', color='sentiment_label', 
                   barnorm='percent', orientation='h')
plot(fig)


# + code_folding=[0] hidden=true
# Sentiment score distribution 
fig = px.scatter(df, y='artist', x='sentiment_score', title='Sentimento por música',
                 color='artist', hover_name='name')

plot(fig)


# + [markdown] heading_collapsed=true hidden=true
# ### Emotions

# + hidden=true
lyrics_df.head(3)

# + hidden=true
emotions_df = lyrics_df.loc[:, ['album_name', 'artist', 'name', 'nrc_emotions_relative', 'lyrics']]
emotions_df.columns = ['album', 'artist', 'song', 'emotions', 'lyrics']

emotions_df.loc[:, 'equal_dist'] = emotions_df['emotions'].apply(lambda x: len(np.unique(list(x.values()))) > 1)

# emotions_df.loc[:, 'zero_dist'] = emotions_df['emotions'].apply(lambda x: np.sum(list(x.values())) == 0)

emotions_df.loc[:, 'has_main_emotion'] = emotions_df['emotions'].apply(lambda x: len(np.unique(list(x.values()))) > 1)

emotions_df.loc[:, 'emotion_only'] = emotions_df['emotions'].apply(lambda x: len(np.unique(list(x.values()))) > 1)

emotions_df = emotions_df.loc[(~emotions_df['lyrics'].isnull()) & (emotions_df['has_main_emotion'])]

emotions_df.loc[:, 'main_emotion'] = emotions_df['emotions'].apply(lambda x:max(x, key=x.get))
emotions_df.loc[~emotions_df['equal_dist'], 'main_emotion'] = 'inconclusive'

aux = lyrics_df['nrc_emotions_relative'].apply(pd.Series)
emotions = aux.columns.tolist()
emotions.remove('positive')
emotions.remove('negative')

emotions_df = emotions_df.join(aux, how='left')
emotions_df.head()

# + hidden=true
melt = pd.melt(emotions_df, id_vars=['album', 'artist', 'song', 'main_emotion'], value_vars=emotions).reset_index()
melt.head()

# + hidden=true
melt.loc[melt['artist'] == 'Rihanna'].sort_values(by='value')

# + hidden=true
fig = px.histogram(melt, y='artist', x='value', facet_col='variable', facet_col_wrap=4, 
                color='artist', height=700, hover_name='song', orientation='h', histfunc='avg')
# fig.update_traces(side='positive', width=2)

plot(fig)

# + hidden=true
emotions_avg = emotions_df.groupby(['artist']).agg({c:'mean' for c in emotions})
#emotions_med = emotions_df.groupby(['artist']).agg({c:'median' for c in emotions})
emotions_avg
px.histogram(melt, x='artist', color='main_emotion')

# + [markdown] heading_collapsed=true hidden=true
# ### Songwriting and Producing Credits

# + code_folding=[] hidden=true
# Data Prep 
credits_df = lyrics_df\
                .groupby(['artist'], as_index=False)\
                .agg({'artist_wrote_song':'sum', 'artist_produced_song':'sum', 'name':'count'})

credits_df.loc[:, 'writing_credits'] = 100*credits_df['artist_wrote_song']/credits_df['name']
credits_df.loc[:, 'producing_credits'] = 100*credits_df['artist_produced_song']/credits_df['name']
credits_df.head()

# + code_folding=[] hidden=true
# Producing and songwriting credits, per artist
fig = px.bar(credits_df.sort_values(by='writing_credits'), y='artist', 
             x='writing_credits', orientation='h', color='artist', color_discrete_map=color_dict)

fig.update_traces(texttemplate='%{x:.0f}%')
fig.update_layout(
    title='Writing credits, per artist',
    xaxis_title='Percentage of songs with artist as writer',
    yaxis_title='Artist',
    showlegend=False,
)
plot(fig)
write(fig, 'writing_credits_en')

fig = px.bar(credits_df.sort_values(by='producing_credits'), y='artist', 
             x='producing_credits', orientation='h', color='artist', color_discrete_map=color_dict)

fig.update_traces(texttemplate='%{x:.0f}%', textposition='outside')
fig.update_xaxes(range=[0, 110])
fig.update_layout(
    title='Producing credits, per artist',
    xaxis_title='Percentage of songs with artist as producer',
    yaxis_title='Artist',
    showlegend=False,
)
plot(fig)
write(fig, 'producing_credits_en')
py.plot(fig, filename = 'producing_credits_en', auto_open=True)

# + [markdown] hidden=true
# ### Lyrics and Production collab

# + code_folding=[0] hidden=true
# General
colabs_df = lyrics_df.copy()
colabs_df = lyrics_df\
                .groupby(['artist'], as_index=False)\
                .agg({'writers': lambda x: ','.join(x), 'album_name':'nunique', 
                      'song':'count', 'producers': lambda x: ','.join(x),})
colabs_df.head()

colabs_df.loc[:, 'writers'] = colabs_df['writers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'producers'] = colabs_df['producers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'writers_num'] = colabs_df['writers'].apply(lambda x: len(set(x)))
colabs_df.loc[:, 'producers_num'] = colabs_df['producers'].apply(lambda x: len(set(x)))

colabs_df.loc[:, 'Writers'] = colabs_df['writers_num']/colabs_df['song']
colabs_df.loc[:, 'Producers'] = colabs_df['producers_num']/colabs_df['song']


group = pd\
        .melt(colabs_df, id_vars='artist', value_vars=['Writers', 'Producers'])\
        .sort_values(by=['variable', 'value'])

fig = px.bar(group, y='artist', x='value', color='artist', facet_col='variable',
             color_discrete_map=color_dict)
fig.update_layout(title='Number of collaborators/Discography Size, per artist', 
                  showlegend=False, yaxis_title='Artist')
fig.update_traces(texttemplate='%{x:.1f}')
fig.update_xaxes(matches=None, title='')
fig.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1]))
plot(fig)
write(fig, 'collaborations_en')


# fig = px.bar(colabs_df.sort_values(by='producers_per_song'), y='artist', x='producers_per_song', color='artist', color_discrete_map=color_dict)
# plot(fig)

# + code_folding=[0] hidden=true
# per Song
colabs_df = lyrics_df.copy()
# colabs_df = lyrics_df\
#                 .groupby(['artist'], as_index=False)\
#                 .agg({'writers': lambda x: ','.join(x), 'album_name':'nunique', 
#                       'song':'count', 'producers': lambda x: ','.join(x),})
# colabs_df.head()

colabs_df.loc[:, 'writers'] = colabs_df['writers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'producers'] = colabs_df['producers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'writers_num'] = colabs_df['writers'].apply(lambda x: len(set(x)))
colabs_df.loc[:, 'producers_num'] = colabs_df['producers'].apply(lambda x: len(set(x)))

# colabs_df.loc[:, 'writers_per_song'] = colabs_df['writers_num']/colabs_df['song']
# colabs_df.loc[:, 'producers_per_song'] = colabs_df['producers_num']/colabs_df['song']


colabs_df.head()

fig = px.histogram(colabs_df.sort_values(by='writers_num'), x='artist', y='writers_num',
                   color='artist', color_discrete_map=color_dict, histfunc='avg')
plot(fig)

fig = px.histogram(colabs_df.sort_values(by='producers_num'), x='artist', y='producers_num', 
             color='artist', color_discrete_map=color_dict, histfunc='avg')
plot(fig)

# + code_folding=[0] hidden=true
# per Album
colabs_df = lyrics_df.copy()
colabs_df = lyrics_df\
                .groupby(['artist', 'album_name'], as_index=False)\
                .agg({'writers': lambda x: ','.join(x),
                      'song':'count', 'producers': lambda x: ','.join(x),})
colabs_df.head()

colabs_df.loc[:, 'writers'] = colabs_df['writers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'producers'] = colabs_df['producers']\
                                .str.replace('[', '')\
                                .str.replace(']', '')\
                                .str.replace(' ', '')\
                                .str.strip()\
                                .str.split(',')

colabs_df.loc[:, 'writers_num'] = colabs_df['writers'].apply(lambda x: len(set(x)))
colabs_df.loc[:, 'producers_num'] = colabs_df['producers'].apply(lambda x: len(set(x)))

# colabs_df.loc[:, 'writers_per_song'] = colabs_df['writers_num']/colabs_df['song']
# colabs_df.loc[:, 'producers_per_song'] = colabs_df['producers_num']/colabs_df['song']


colabs_df.head()

fig = px.histogram(colabs_df.sort_values(by='writers_num'), x='artist', y='writers_num',
                   color='artist', color_discrete_map=color_dict, histfunc='avg')
plot(fig)

fig = px.histogram(colabs_df.sort_values(by='producers_num'), x='artist', y='producers_num', 
             color='artist', color_discrete_map=color_dict, histfunc='avg')
plot(fig)

# + [markdown] hidden=true
# ### Lemmas and vocabulary

# + hidden=true
from collections import Counter

# + code_folding=[0] hidden=true
# Data Prep 
vocab_dfs = []

for artist in lyrics_df['artist'].unique().tolist():
    vocab_explore = {}
    df = lyrics_df.loc[lyrics_df['artist'] == artist].copy()
    df.loc[:, 'lyrics_lemmatized'] = df['lyrics_lemmatized'].fillna('').str.lower()
    all_lemmas = []
    all_lemmas = ' '.join(df['lyrics_lemmatized'].tolist())
    all_lemmas_list = all_lemmas.split(' ')
    vocab_freq = dict(Counter(all_lemmas_list))
    vocab_dict = {'artist':artist, 'lemma_frequency':vocab_freq}
    
    vocab_df = pd.DataFrame.from_dict(vocab_dict).reset_index()
    vocab_df.columns = ['lemma', 'artist', 'lemma_frequency']
    
    vocab_dfs.append(vocab_df)
    
vocab_df = pd.concat(vocab_dfs, axis=0)

vocab_df.loc[:, 'lemma_rank_to_artist'] = vocab_df.groupby(['artist'])['lemma_frequency'].rank(method='first', ascending=False)
vocab_df.loc[:, 'total_artist_lemmas'] = vocab_df.groupby(['artist'])['lemma_frequency'].transform('sum')#(method='first', ascending=False)
vocab_df.loc[:, 'artist_lemma_percentage'] = 100*vocab_df['lemma_frequency']/vocab_df['total_artist_lemmas']

vocab_df.loc[:, 'lemma_artist_rank'] = vocab_df.groupby(['lemma'])['artist_lemma_percentage'].rank(method='first', ascending=False)

vocab_df.loc[:, 'lemma_total_ocurrence'] = vocab_df.groupby(['lemma'])['lemma_frequency'].transform('sum')
vocab_df.loc[:, 'lemma_rank_overall'] = vocab_df['lemma_total_ocurrence'].rank(method='dense', ascending=False)
vocab_df.loc[:, 'total_lemmas'] = vocab_df['lemma_frequency'].sum()
vocab_df.loc[:, 'lemma_percentage_overall'] = 100*vocab_df['lemma_total_ocurrence']/vocab_df['total_lemmas']


vocab_df.head()

# + hidden=true
lemma_df = vocab_df\
            .groupby(['lemma'], as_index=False)\
            .agg({'lemma_frequency':'sum'})\
            .sort_values(by='lemma_frequency', ascending=False)

lemma_df.loc[:, 'total_lemmas'] = lemma_df['lemma_frequency'].sum()
lemma_df.loc[:, 'percentage'] = 100*lemma_df['lemma_frequency']/lemma_df['total_lemmas']
lemma_df.loc[:, 'rank'] = lemma_df['lemma_frequency'].rank(method='first', ascending=False)

lemma_df.head(10)

# + code_folding=[] hidden=true
# Main words

df = vocab_df.loc[(vocab_df['lemma_rank_overall'] <= 15) & (vocab_df['lemma_artist_rank'] == 1)]

fig = px.bar(df, x='lemma_rank_overall', text='lemma', y='lemma_percentage_overall', color='artist',
            title='Top 15 most used words, and artist that uses it the most', color_discrete_map=color_dict
            )
fig.update_layout(yaxis_title='Percentage from all words', xaxis_title='')
fig.update_traces(textposition='outside')
fig.update_xaxes(tickmode='linear')
plot(fig)
write(fig, 'top_used_words_en')

# + code_folding=[] hidden=true
# # Words comparison 
# main_expressions = ['love']
# df = vocab_df.loc[vocab_df['lemma'].isin(main_expressions)].sort_values(by=['lemma_rank_overall', 'artist_lemma_percentage'])

# fig = px.bar(df, y='artist', x='artist_lemma_percentage', color='artist', facet_col='lemma',
#             facet_col_wrap=3, title='Love', color_discrete_map=color_dict)
# plot(fig)


# # main_expressions = ['love', 'know', 'like']
# # df = vocab_df.loc[vocab_df['lemma'].isin(main_expressions)].sort_values(by=['lemma_rank_overall'])
# # fig = px.bar(df, y='artist', x='artist_lemma_percentage', color='artist', facet_col='lemma',
# #             facet_col_wrap=3, title='Love x Know x Like', color_discrete_map=color_dict)
# # plot(fig)

# + code_folding=[] hidden=true
# Quantidade de palavras 
song_count = lyrics_df.groupby(['artist']).agg({'song':'count'})
unique_lemmas = vocab_df.groupby(['artist']).agg({'lemma':'nunique'}).join(song_count, how='left').reset_index()

unique_lemmas.loc[:, 'unique_lemmas_per_song'] =  unique_lemmas['lemma']/unique_lemmas['song']

# # Visão geral
# fig = px.bar(unique_lemmas.sort_values(by='lemma'), y='artist', x='lemma', color='artist', title='Palavras únicas na discografia')
# fig.show()

fig = px.bar(unique_lemmas.sort_values(by='unique_lemmas_per_song'), y='artist', x='unique_lemmas_per_song', 
             color='artist', title='', color_discrete_map=color_dict)
fig.update_traces(texttemplate='%{x:.0f}')

fig.update_layout(yaxis_title='Artist', xaxis_title='', showlegend=False,
                  title='Unique words/Discography size, per artist')

plot(fig)
write(fig, 'unique_words_per_song_en')

# + [markdown] hidden=true
# Katy Pery: Mais palavras diferentes!

# + code_folding=[0] hidden=true
# # Top 10 lemmas 
# df = vocab_df.loc[vocab_df['lemma_rank_to_artist'] <= 10]#.head(100)

# # fig = px.scatter(df, x='lemma_rank_to_artist', y='artist', color='artist', text='lemma', size='artist_lemma_percentage')
# # fig.update_traces(textposition='top center')
# # plot(fig)

# fig = px.bar(df, x='lemma_frequency', y='lemma_rank_to_artist', orientation='h', range_y=[11, 0],
#              facet_col='artist', color='artist', text='lemma')
# fig.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1]))
# fig.update_traces(textposition='outside')

# plot(fig)

# + code_folding=[] hidden=true
# Words per minute and per song 
df = lyrics_df.copy()

df_group = df.groupby(['artist'], as_index=False).agg({'total_word_count':'median', 'words_per_min':'median'})

fig = px.bar(df_group.sort_values(by='total_word_count'), y='artist', x='total_word_count', color='artist', 
            orientation='h', color_discrete_map=color_dict)

fig.update_traces(texttemplate='%{x:.0f}')
fig.update_layout(yaxis_title='Artist', xaxis_title='', showlegend=False,
                  title='Words per song, per artist')

plot(fig)
write(fig, 'words_per_song_en')


fig = px.bar(df_group.sort_values(by='words_per_min'), y='artist', x='words_per_min', color='artist', 
            orientation='h',  color_discrete_map=color_dict)

fig.update_traces(texttemplate='%{x:.0f}')
fig.update_layout(yaxis_title='Artist', xaxis_title='', showlegend=False,
                  title='Words per minute, per artist')

plot(fig)
write(fig, 'words_per_minute_en')
# fig = px.box(df, x='total_word_count', y='artist', hover_name='name', color='artist', orientation='h')
# plot(fig)

# fig = px.box(df, x='words_per_min', y='artist', hover_name='name', color='artist', orientation='h')
# plot(fig)

# + hidden=true
words = vocab_df.loc[vocab_df['lemma_rank_overall'] < 10, 'lemma'].unique()
#words = ['fuck']
check_word_in_lyrics = lyrics_df.copy()

for word in words:
    check_word_in_lyrics.loc[:, word] = check_word_in_lyrics['lyrics_clean'].str.contains(word, case=False, na=False).astype(int)

df = check_word_in_lyrics.groupby(['artist']).agg({w:'mean' for w in words})
df.head(20)


# + code_folding=[] hidden=true
def plot_word_appeareance(words):
    main_expressions = words
    check_word_in_lyrics = lyrics_df.copy()

    check_word_in_lyrics.loc[:, 'has_words'] = check_word_in_lyrics['lyrics_lemmatized'].fillna('').apply(lambda x: max([1 if e in x else 0 for e in main_expressions]))

    df = check_word_in_lyrics.groupby(['artist'], as_index=False).agg({'has_words':'mean'})
    df.loc[:, 'value'] = 100*df['has_words']
    fig = px.bar(df.sort_values(by='value'), y='artist', x='value', color='artist', 
                 color_discrete_map=color_dict)
    fig.update_layout(showlegend=False, 
                      yaxis_title='Artist', 
                      xaxis_title='Word presence in songs (%)'
                     )
    fig.update_traces(texttemplate='%{x:.1f}%')
    return fig


# + hidden=true
fig = plot_word_appeareance([' love '])
fig.update_layout(title='Use of \'love\' in songs')
plot(fig)
write(fig, 'love_on_songs_en')

# + hidden=true
fig = plot_word_appeareance([' yeah '])
fig.update_layout(title='Use of \'yeah\' in songs')
plot(fig)
write(fig, 'yeah_on_songs_en')

# + hidden=true
fig = plot_word_appeareance([' fuck '])
fig.update_layout(title='Use of \'fuck\' in songs')
plot(fig)
write(fig, 'fuck_on_songs_en')

# + hidden=true
fig = plot_word_appeareance([' oh '])
fig.update_layout(title='Use of \'Oh\' in songs')
plot(fig)
write(fig, 'oh_on_songs_en')

# + [markdown] heading_collapsed=true hidden=true
# ### Keywords

# + hidden=true
keywords_df = lyrics_df.copy()

# + [markdown] heading_collapsed=true hidden=true
# #### Tf-idf

# + hidden=true
from sklearn.feature_extraction.text import TfidfVectorizer

# + hidden=true
all_dfs = []
for a in lyrics_df['artist'].unique().tolist():

    data = lyrics_df.loc[(lyrics_df['artist'] == a) &
                        (~lyrics_df['lyrics_lemmatized'].isnull()), 'lyrics_lemmatized'].tolist()

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(data)
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["score"])
    df.loc[:, 'artist'] = a
    all_dfs.append(df)

tf_idf = pd.concat(all_dfs, axis=0)
tf_idf.loc[:, 'artist_rank'] = tf_idf.groupby(['artist'])['score'].rank(method='first', ascending=False)

tf_idf.sort_values(by='score')

# + [markdown] heading_collapsed=true hidden=true
# #### Yake - per song, album, artist

# + hidden=true
import yake

lyrics_data = keywords_df['lyrics_clean'].fillna('').tolist()

keywords = [] 
for l in lyrics_data:
    if l and l != '':
        kw_extractor = yake.KeywordExtractor()
        keyword = kw_extractor.extract_keywords(l)[0][0]
        keywords.append(keyword)
    else:
        keywords.append('')
        
keywords_df.loc[:, 'keyord_yake'] = keywords

# + hidden=true
artists_keywords = []
for a in lyrics_df['artist'].unique().tolist():

    artist_lyrics = '.'.join(lyrics_df.loc[lyrics_df['artist'] == a, 'lyrics_clean'].fillna('').tolist())

    keywords = [] 
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(artist_lyrics)
    df = pd.DataFrame(keywords)
    df.columns = ['word', 'score']
    df.loc[:, 'artist'] = a
    artists_keywords.append(df)
    
artists_kw = pd.concat(artists_keywords)
artists_kw.loc[:, 'rank'] = artists_kw.groupby('artist')['score'].rank(method='first', ascending=True)
artists_kw.head()

# + hidden=true
fig = px.bar(artists_kw.loc[artists_kw['rank'] <= 5], color='artist', x='word', y='score', barmode='group')
plot(fig)

# + hidden=true
album_keywords = []
for a in lyrics_df['album_name'].unique().tolist():

    artist_lyrics = '.'.join(lyrics_df.loc[lyrics_df['album_name'] == a, 'lyrics_clean'].fillna('').tolist())

    keywords = [] 
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(artist_lyrics)
    df = pd.DataFrame(keywords)
    df.columns = ['word', 'score']
    df.loc[:, 'album'] = a
    album_keywords.append(df)
    
albums_kw = pd.concat(album_keywords)
albums_kw.head()

# + [markdown] hidden=true
# #### Rake
# -

# ## Finder


pop_divas_df.head()


def filter_word_appeareance(word, artist):
    main_expressions = [word]
    check_word_in_lyrics = lyrics_df.loc[lyrics_df['artist'] == artist]
    check_word_in_lyrics.loc[:, 'has_words'] = check_word_in_lyrics['lyrics_lemmatized'].fillna('').apply(lambda x: max([1 if e in x else 0 for e in main_expressions]))
    
    check_word_in_lyrics = check_word_in_lyrics.loc[check_word_in_lyrics['has_words'] == 1]
    df = pop_divas_df.join(check_word_in_lyrics, how='inner', rsuffix='_lyrics')
    
    return df


# + [markdown] heading_collapsed=true
# ### Beyoncé

# + hidden=true
df = filter_word_appeareance(' love ', 'Beyoncé')
df.sort_values(by=['speechiness', 'total_word_count'], ascending=False).iloc[3]

# + hidden=true
all_closest_songs.loc[all_closest_songs['own_artist']].groupby(['artist']).agg({'own_songs_ranking':'max'})

# + hidden=true
all_closest_songs.loc[(all_closest_songs['artist'] == 'Beyoncé') 
                      & (all_closest_songs['song'] == 'Hold Up')                    
                      & (all_closest_songs['own_artist'])]

# + [markdown] heading_collapsed=true
# ### Taylor Swift

# + hidden=true
df = songs_scaled.loc[
    (songs_scaled['artist'] == 'Taylor Swift')
    & (songs_scaled['valence'] < 0)
    & (songs_scaled['sentiment_score'] < 0)
]

df.loc[:, 'rank_sentiment'] = df.groupby(['artist'])['sentiment_score'].rank(method='first', ascending=True)
df.loc[:, 'rank_valence'] = df.groupby(['artist'])['valence'].rank(method='first', ascending=True)

# + hidden=true
px.scatter(df, x='valence', y='sentiment_score', hover_name='song', hover_data=['album', 'rank_sentiment', 'rank_valence'], 
           color='album', size_max=10)

# + hidden=true
px.scatter(df, x='artist', y='sentiment_score', hover_name='song', hover_data=['album'], 
           color='album', size_max=10)

# + [markdown] heading_collapsed=true
# ### Rihanna

# + hidden=true
df = filter_word_appeareance(' fuck ', 'Rihanna')
df.sort_values(by=['explicit', 'song_popularity'], ascending=False).head(10)
# -

# ### Katy Perry

# +
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Katy Perry']
fig = px.scatter(df, x='album_name', y='tempo', size='tempo', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
# -

df = filter_word_appeareance(' oh ', 'Katy Perry')
df.sort_values(by='tempo', ascending=False).head(20)
px.scatter(df, x='sentiment_score', y='valence', size='tempo', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)

# ### Britney Spears

# +
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Britney Spears']
fig = px.scatter(df, x='sentiment_score', y='danceability', size='song_popularity', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

# px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
#            color='album_name', size_max=10)
# -

# ### Lady Gaga

# +
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Lady Gaga']
fig = px.scatter(df, x='album_name', y='energy', size='song_popularity', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

# px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
#            color='album_name', size_max=10)
# -

px.scatter(df.loc[df['sentiment_score'] < 0], y='artist', x='energy', color='album_name', hover_name='name')

# +
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Lady Gaga']
fig = px.scatter(df, x='sentiment_score', y='danceability', size='energy', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

# px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
#            color='album_name', size_max=10)
# -

# ### Ariana Grande

df.sort_values(by='words_per_min', ascending=False)

# +
# df = filter_word_appeareance(' yeah ', 'Ariana Grande')
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Ariana Grande']

df.loc[:, 'words_per_min'] = df['total_word_count']/df['duration_min']
fig = px.scatter(df, x='words_per_min', y='tempo', size='duration_min', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

# px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
#            color='album_name', size_max=10)
# -

# ### Madonna

albums_mean.head()

# +
df = albums_mean.loc[albums_mean['artist'] == 'Madonna']
cols = df.iloc[:, 3:].columns.tolist()

df = pd.melt(df, id_vars=['album_name', 'year'], value_vars=cols)
df = df.loc[df['variable'].isin(['danceability', 'energy', 'sentiment_score', 'instrumentalness','acousticness'])]

# fig = px.bar(df.sort_values(by='year'), y='album_name', x='value', facet_col='variable', 
#              facet_col_wrap=4, color='album_name')
# fig.update_xaxes(matches=None)
# plot(fig)

# df.head()

fig = px.line(df.sort_values(by='year'), y='value', x='year', line_shape='spline', hover_name='album_name',
              color='variable', facet_col='variable',facet_col_wrap=2, height=800)
fig.update_layout(showlegend=False)
fig.update_yaxes(matches=None, showticklabels=False, title='')
fig.update_xaxes(showticklabels=True, range=[1980,2019])
fig.update_traces(texttemplate='%{x:}')
fig.for_each_annotation(lambda x: x.update(text=x.text.split('=')[1]))
plot(fig)
# -

df = top_own_songs.loc[top_own_songs['artist'] == 'Madonna']
px.scatter(df, x='distance', y='artist', color='album', hover_name='song')

# ## Mariah Carey

# +
# df = filter_word_appeareance(' yeah ', 'Ariana Grande')
df = pop_divas_df.loc[pop_divas_df['artist'] == 'Mariah Carey']

fig = px.scatter(df, x='song_popularity', y='acousticness', hover_name='name', hover_data=['album_name'], 
           color='album_name', size_max=10)
fig.show()

# px.scatter(df, x='valence', y='sentiment_score', size='tempo', hover_name='name', hover_data=['album_name'], 
#            color='album_name', size_max=10)

# + [markdown] heading_collapsed=true
# ## Clustering

# + hidden=true
cluster_columns = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min', 'time_signature', 
                   'track_position', 'feat', 'explicit'
                  ]

cluster_df = pop_divas_df.copy().dropna(axis=0)

X_cluster = cluster_df.loc[:, cluster_columns]
X_cluster.loc[:, 'feat'] = X_cluster['feat'].astype(int)
X_cluster.loc[:, 'explicit'] = X_cluster['explicit'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# + hidden=true
# Plot 2d
pca = PCA(n_components=2)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled))

print(np.sum(pca.explained_variance_ratio_))
X_pca.columns = ['dim{}'.format(c) for c in X_pca]
X_pca.index = pop_divas_df.index
X_pca.loc[:, 'artist'] = pop_divas_df['artist']
px.scatter(X_pca, x='dim1', y='dim0', color='artist').show()

# Plot 3d
pca = PCA(n_components=3)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled))

print(np.sum(pca.explained_variance_ratio_))
X_pca.columns = ['dim{}'.format(c) for c in X_pca]
X_pca.index = pop_divas_df.index
X_pca.loc[:, 'artist'] = pop_divas_df['artist']
px.scatter_3d(X_pca, x='dim1', y='dim0', z='dim2', color='artist').show()


# + hidden=true
def calculate_wcss(data, max_clusters):
    wcss = {}
    for n in range(1, max_clusters):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss[n] = kmeans.inertia_
        
        df = cluster_df.copy()
        
        df = 
            .groupby(['cluster'], as_index=False)\
            .agg({'artist':'nunique', 'name':'count'})\
            .sort_values(by='artist', ascending=False)\
            .head(10)

    return wcss


dict_num_clusters = calculate_wcss(X_cluster, 50)
metrics_df = pd.DataFrame.from_dict(dict_num_clusters, orient='index').reset_index()
metrics_df.columns = ['num_clusters', 'wcss']

px.line(metrics_df, x='num_clusters', y='wcss', line_shape='spline')

# + hidden=true
# n_clusters = (cluster_df['album_name'].nunique())
n_clusters = 50

# + hidden=true
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit_transform(X_cluster)

cluster_df.loc[:, 'cluster'] = kmeans.labels_

# + hidden=true
info_clusters = cluster_df\
                    .groupby(['cluster'], as_index=False)\
                    .agg({'artist':'nunique', 'name':'count'})\
                    .sort_values(by='artist', ascending=False)\

all_divas_clusters = info_clusters.loc[info_clusters['artist']  == cluster_df['artist'].nunique()]['cluster'].tolist()


clusters_all_divas = cluster_df\
                        .loc[cluster_df['cluster'].isin(all_divas_clusters)]\
                        .groupby(['cluster', 'artist'], as_index=False)\
                        .agg({'name': 'count'})

clusters_all_divas.loc[:, 'cluster_size'] = clusters_all_divas.groupby(['])

# + hidden=true
cluster_df.loc[cluster_df['cluster'] == 31]
