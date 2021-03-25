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

import pandas as pd
from pathlib import Path
import os
from pymediainfo import MediaInfo
from sources.visualization_functions import *
import numpy as np
from scipy.signal import find_peaks 
import re

data_path = Path('./data')
movies_prep_path = data_path/'movies_prep'
# -

# # Data Prep

# ## Silences + Subs

movie_ids = [f for f in os.listdir(movies_prep_path) if not f.startswith('.') if os.path.exists(movies_prep_path/f'{f}/audio/silences.csv')]

# +
array = []
for movie_id in movie_ids:
    s = movies_prep_path/f'{movie_id}/audio/silences.csv'
    movie_id = str(s).split('/')[-3]
    df = pd.read_csv(s)
    df.loc[:, 'imdb_id'] = movie_id
#     df.loc[:, 'position'] = df['silence_start'].rank(method='first').astype(int)
#     df.loc[:, 'silence_id'] = df.apply(lambda x: f"{x['imdb_id']}_{x['position']}", axis=1)
    array.append(df)
    
silences_raw_df = pd.concat(array)
silences_raw_df.head()
# -

subs_raw_df = pd.read_csv(data_path/'prep/all_subtitles.csv')
subs_raw_df.loc[:, 'start'] = subs_raw_df.apply(lambda x: x['start_h']*3600 + x['start_min']*60 +  x['start_s'] + x['start_ms']/1000, axis=1)
subs_raw_df.loc[:, 'end'] = subs_raw_df.apply(lambda x: x['end_h']*3600 + x['end_min']*60 +  x['end_s'] + x['end_ms']/1000, axis=1)
subs_df = subs_raw_df.loc[:, ['imdb_id', 'text', 'start', 'end', 'position']]
subs_df.loc[:, 'duration'] = subs_df['end'] - subs_df['start']
subs_df.head()


def format_timer(x):
     return f"{x//3600:02.0f}:{(x%3600)//60:02.0f}:{(x%3600)%60:02.0f}"


# +
cols = ['imdb_id', 'start', 'end']

df_subs = subs_df.loc[:, ['imdb_id', 'start', 'end']]
df_subs.columns = cols
df_subs.loc[:, 'type'] = 'subtitle'

df_sil = silences_raw_df.loc[:, ['imdb_id', 'silence_start', 'silence_end']]
df_sil.columns = cols
df_sil.loc[:, 'type'] = 'silence'

df = pd.concat([df_subs, df_sil], axis=0).sort_values(by=['imdb_id', 'start'])
df.loc[:, 'before'] = df.groupby(['imdb_id'])['type'].shift(1).fillna('start')
df.loc[:, 'after'] = df.groupby(['imdb_id'])['type'].shift(-1).fillna('end')
df.loc[:, 'prev_end'] = df.groupby(['imdb_id'])['end'].shift(1).fillna(0)
df.loc[:, 'next_start'] = df.groupby(['imdb_id'])['end'].shift(-1).fillna(df['end'])

silences_df = df.loc[df['type'] == 'silence'].drop(columns=['type'])
# silences_df.loc[:, 'mid_sub'] = (silences_df['start'] < silences_df['last_end']) | (silences_df['end'] > silences_df['next_start'])

silences_df.loc[:, 'rank'] = silences_df\
                                .sort_values(by=['imdb_id','start'])\
                                .groupby(['imdb_id'])['start'].rank(method='first').astype(int)

silences_df.loc[:, 'silence_id'] = silences_df.apply(lambda x: f"{x['imdb_id']}_{x['rank']}", axis=1)
silences_df.loc[:, 'duration'] = silences_df['end'] - silences_df['start']

silences_df.head(1)
# -

silences_df.to_csv(data_path/'prep/silences_info.csv', index=False)

# ## Duration dict

movie_duration_dict = {}
for m in movie_ids:
    file_path = movies_prep_path/f'{m}/audio/movie_audio.mp3'
#     print(file_path)
    media_info = MediaInfo.parse(file_path)
    #duration in seconds
    movie_duration = media_info.tracks[0].duration/1000 
    movie_duration_dict[m] = movie_duration

# ## Movie info 

# +
imdb_df = pd.read_csv(data_path/'imdb_top250_movies.csv')

imdb_df = imdb_df.loc[imdb_df['top_250_rank'] <= 150]

movie_info_df = imdb_df.loc[:, ['imdb_id', 'title', 'year', 'rating', 'genres', 'top_250_rank','color_info']]
# movie_info_df.loc[:, 'genres'] = movie_info_df['genres'].apply(lambda x: eval(x))
# movie_info_df.loc[:, 'color_info'] = movie_info_df['color_info'].apply(lambda x: eval(x))
movie_info_df.head(1)

# +
silences_df = pd.read_csv(data_path/'prep/silences_info.csv')

silences_df = pd.merge(left=silences_df, right=movie_info_df, on='imdb_id', how='inner')

silences_df.loc[:, 'total_duration'] = silences_df['imdb_id'].apply(lambda x: movie_duration_dict[x])
silences_df.loc[:, 'pos_rel'] = 100*silences_df['start']/silences_df['total_duration']
silences_df.loc[:, 'dur_rel'] = 100*silences_df['duration']/silences_df['total_duration']
silences_df.head()

# + code_folding=[]
# Dialogue + Silence + IMDB info
# Dialogue info

df = subs_df.copy()
df.loc[:, 'no_cc'] = df['text'].apply(lambda x: re.sub(r'(\[[^\[\]]*\])', '', str(x))).str.strip()
df = df.loc[df['no_cc'] != '']

df.loc[:, 'word_count'] = df['no_cc'].apply(lambda x: len([f for f in re.split(r'[ \n]', str(x)) if f]))
df.loc[:, 'line_count'] = df['no_cc'].apply(lambda x: len([f for f in re.split(r'[ \n]- ', str(x)) if f]))

df_dial = df\
            .groupby(['imdb_id'])\
            .agg({'duration':'sum', 'word_count':'sum', 'line_count':'sum'})\
            .rename(columns={'duration':'dialogue_duration'})

# Silences info
df_sil = silences_df\
            .groupby(['imdb_id'])\
            .agg({'duration':'sum', 'silence_id':'count'})\
            .rename(columns={'duration':'silence_duration', 'silence_id':'num_silences'})

# IMDB info
df_info = movie_info_df.set_index('imdb_id')

# Final DataFrame
movies_df = pd.concat([df_info, df_sil, df_dial], axis=1).reset_index().rename(columns={'index':'imdb_id'}).dropna()


movies_df.loc[:, 'total_duration'] = movies_df['imdb_id'].apply(lambda x: movie_duration_dict[x])
movies_df.loc[:, 'silence_dur'] = 100*movies_df['silence_duration']/movies_df['total_duration']
movies_df.loc[:, 'dialogue_dur'] = 100*movies_df['dialogue_duration']/movies_df['total_duration']
movies_df.loc[:, 'other_dur'] = 100 - movies_df['silence_dur'] - movies_df['dialogue_dur']
movies_df.loc[movies_df['other_dur'] < 0, 'other_dur'] = 0
movies_df.head()
# -

movies_df.to_csv(data_path/'prep/movies_infos.csv', index=False)

imdb_df.loc[~imdb_df['imdb_id'].isin(subs_df['imdb_id'])]

main_palette = px.colors.qualitative.Safe


def vectorize_column(raw_df, col):
    df = raw_df.copy()
    df.loc[:, col] = df[col].apply(lambda x: eval(x))
    return df


# # Analysis

# ## All Sounds

# +
df = movies_df.copy()
fig = px.violin(df, x = ['silence_dur', 'dialogue_dur', 'other_dur'], y='variable', 
                color='variable', color_discrete_sequence=main_palette)

fig.update_traces(orientation='h', side='positive', width=1,
                  points=False, spanmode='hard', meanline_visible=True)

fig.update_xaxes(ticksuffix='%', title='Percentage of movie', dtick=100, range=[-1,101])
fig.update_yaxes(showgrid=True)
fig.update_layout(title='What composes the sound of movies?', showlegend=False)

plot(fig)

# + code_folding=[0]
# Sound composition
df = movies_df.copy()

df.loc[:, 'decade'] = (np.floor(df['year']/20)*20).astype(int)
df.loc[:, 'decade'] = df['decade'].apply(lambda x: f"{x} - {x+19}")

fig = px.strip(df.sort_values(by='decade'), y=['silence_dur', 'dialogue_dur', 'other_dur'],
               facet_col='decade', color_discrete_sequence=main_palette, color='variable',
               hover_name='title', hover_data=['year', 'top_250_rank'])

fig.update_yaxes(dtick=50, range=[0,110], ticksuffix='%')
fig.update_xaxes(title='', showticklabels=False)

fig.update_layout(
    title = 'By which type of sound each movie is composed?', 
    showlegend=True, 
    margin_t=120,
    legend_title='Type of sound',
    yaxis_title='')

fig = facet_prettify(fig)
plot(fig)
# -
# ## Silences

# + code_folding=[]
# Duration of each silence
df = silences_df.copy()

top = df['duration'].quantile(0.99)
fig = px.histogram(df, x='duration', histnorm='percent',  color_discrete_sequence=main_palette)
fig.update_xaxes(matches=None, title='Seconds', range=[0.8, top])
fig.update_yaxes(ticksuffix='%', title='Percentage of occurrences')
fig.update_layout(title='How long are silence parts on movies?')
plot(fig)

df = silences_df.copy()

display(df.sort_values(by=['duration'], ascending=False).head(5))

# + code_folding=[0]
# Where do the silences occur?


# # Test 1: Violin

# df = silences_df.copy()

# # top = df['duration'].quantile(0.99)
# fig = px.violin(df, x='pos_rel',  color_discrete_sequence=main_palette)

# fig.update_traces(orientation='h', side='positive', width=1, 
#                   points=False, spanmode='hard')

# fig.update_xaxes(ticksuffix='%', title='Position on movie', dtick=25)
# fig.update_layout(title='Which part of a movie is the quietest?')

# plot(fig)


# Test 2: Line
smooth = 5
df = silences_df.copy()
df.loc[:, 'position_disc'] = np.ceil(df['pos_rel']/smooth)*smooth

df = df\
        .groupby(['imdb_id', 'position_disc'], as_index=False)\
        .agg({'dur_rel':'sum', 'silence_id':'count'})\
        .groupby(['position_disc'])\
        .agg({'dur_rel':['median', 'mean']}).droplevel(0, axis=1)\
        .reset_index()

df.loc[:, 'roll'] = df.sort_values(by='position_disc')['median'].rolling(10, center=True).mean()
df.loc[:, 'rank'] = df['median'].rank(ascending=False).astype(int)

# Use peaks on axis ticks
df.loc[:, 'not_peak'] = True

peaks = list(find_peaks(df['median'])[0])
df.loc[df.index.isin(peaks), 'not_peak'] = False

top_values = df.sort_values(by=['not_peak', 'rank']).iloc[:2]['position_disc'].tolist()
top_values.append(100)

fig = px.line(df, x='position_disc', y=['median'], line_shape='linear', 
              facet_col='variable', color_discrete_sequence=main_palette)
fig.update_xaxes(ticksuffix='%', title='Position on movie', tickvals=top_values, dtick=100, range=[0, 101])
fig.update_yaxes(ticksuffix='%', title='Level of Silence', rangemode='tozero', showgrid=False)
fig.for_each_annotation(lambda x: x.update(text=''))
fig.update_layout(
    title='Which part of a movie is the quietest?',
    showlegend=False
)
plot(fig)




# + code_folding=[0]
# How many silence events, and for how long?
df = silences_df\
        .groupby(['imdb_id'], as_index=False)\
        .agg({'silence_id':'count', 'dur_rel':'sum'})


fig = px.histogram(df, x='silence_id', histnorm='percent', color_discrete_sequence=main_palette)
fig.update_xaxes(matches=None, title='# Silences')
fig.update_yaxes(ticksuffix='%', title='Percentage of movies')
fig.update_layout(title='How many silence events does a film usually have?')
plot(fig)


fig = px.histogram(df, x='dur_rel', histnorm='percent',  color_discrete_sequence=main_palette)
fig.update_xaxes(matches=None, title='Silence duration (%)', ticksuffix='%')
fig.update_yaxes(ticksuffix='%', title='Percentage of movies')
fig.update_layout(title='How much of each movie is made of silence?')

plot(fig)

# + code_folding=[0]
# How many silence events, and for how long?
df = silences_df\
        .groupby(['imdb_id', 'genres', 'total_duration'], as_index=False)\
        .agg({'silence_id':'count', 'dur_rel':'sum'})


df.loc[:, 'genres'] = df['genres'].apply(lambda x: eval(x))

df = pd.DataFrame.explode(df, column='genres')

df.loc[: , 'total_movies_genre'] = df.groupby(['genres'])['imdb_id'].transform('nunique')
df.loc[: , 'rank_genre'] = df['total_movies_genre'].rank(ascending=False, method='dense')

df.loc[:, 'silences_per_hour'] = df['silence_id']/(df['total_duration']/3600)

# fig = px.violin(df, x='dur_rel', y='genres', color='genres')
# fig.update_traces(orientation='h', side='positive', width=2, 
#                   points=False, spanmode='soft', meanline_visible=True)
# plot(fig)


# Average per genre

df = df\
        .groupby(['genres'], as_index=False)\
        .agg({'silences_per_hour':'mean', 'dur_rel':'mean', 'imdb_id':'nunique'})


df.loc[:, 'y_axis'] = df.apply(lambda x: "<b>{}</b> ({})".format(x['genres'], x['imdb_id']), axis=1)


fig = px.bar(df, x=['dur_rel', 'silences_per_hour'], y='y_axis', facet_col='variable', 
             color_discrete_sequence=main_palette)
fig.update_xaxes(matches=None)
fig.update_yaxes(col=1, categoryorder='total ascending')

aux_dict = {'dur_rel': 'Average share of silence on movie genre<br> ', 'silences_per_hour': '# Silences/hour<br> '}

fig.for_each_annotation(lambda a: a.update(text = aux_dict[a.text.split('=')[1]]))
fig.update_xaxes(title='')
fig.update_xaxes(col=1, range=[-0.5, 31], tickvals=[0,30], ticksuffix='%')
fig.update_xaxes(col=2, range=[0, 301], tickvals=[0,300])
fig.update_traces(texttemplate='%{x:.3s}', textposition='outside')
fig.update_traces(texttemplate='%{x:.3s}%', col=1)


fig.update_layout(
    title='Which movie genre is the quietest?',
    showlegend=False,
    yaxis_title='',
    font_size=14,
    titlefont_size=20,
    margin_t=150,
    height=700
)

plot(fig)

# + code_folding=[0]
# Where do the silences occur?
h = 230
grey = f"rgb({h},{h},{h})"
# Test 2: Line
smooth = 5
df = silences_df.copy()
df.loc[:, 'position_disc'] = np.ceil(df['pos_rel']/smooth)*smooth

df = df\
        .groupby(['imdb_id', 'position_disc', 'genres'], as_index=False)\
        .agg({'dur_rel':'sum', 'silence_id':'count'})
        
df = vectorize_column(df, 'genres')

df = pd.DataFrame.explode(df, column='genres')

df = df.groupby(['position_disc', 'genres'])\
        .agg({'dur_rel':['median', 'mean'], 'imdb_id':'nunique'}).droplevel(0, axis=1)\
        .reset_index()

all_dfs = []
for g in df['genres'].unique().tolist():
    h_df = df.copy()
    h_df.loc[:, 'animation_group'] = g
    h_df.loc[:, 'is_highlight'] = h_df['genres'] == g
    all_dfs.append(h_df)
    
highlight_df = pd.concat(all_dfs, axis=0).sort_values(by=['animation_group', 'is_highlight'])
highlight_df.loc[:, 'Genre highlighted'] = highlight_df['animation_group'].apply(lambda x: f"<b>{x}</b>")
highlight_df.loc[:, 'Genre'] = highlight_df['genres']

fig = px.line(highlight_df, x='position_disc', y='mean', color='is_highlight', animation_frame='Genre highlighted',
              color_discrete_map={True:main_palette[1], False:grey},
              line_group='Genre')

fig.update_traces(patch={"line":{"width":3}}, 
                  selector={"name":"True"})

fig.update_layout(
    title='At which part the movie is the quietest? (View per genre)',
    yaxis_title='Silence relevance',
    yaxis_dtick=1,
    yaxis_ticksuffix='%',
    xaxis_title='Position of movie',
    xaxis_ticksuffix='%',    
    showlegend=False,
    xaxis_dtick=25
)

plot(fig)

# + code_folding=[0]
# Where do the silences happen?
smooth = 0.5
df = silences_df.copy()
df.loc[:, 'position_disc'] = np.ceil(df['pos_rel']/smooth)*smooth

df = df\
        .groupby(['imdb_id', 'position_disc', 'genres'], as_index=False)\
        .agg({'dur_rel':'sum', 'silence_id':'count'})
        
df = vectorize_column(df, 'genres')

df = pd.DataFrame.explode(df, column='genres')

df = df\
        .groupby(['position_disc', 'genres'], as_index=False).agg({'dur_rel':'mean'})

df.loc[:, 'x_start'] = df['position_disc']
df.loc[:, 'x_end'] = df['x_start'] + df['dur_rel']
df.loc[df['x_end'] > 100, 'x_end'] = 100

df.loc[: , 'total_dur'] = df.groupby(['genres'])['dur_rel'].transform('sum')
df.loc[:, 'rank'] = df['total_dur'].rank(ascending=True, method='dense')

df = df.sort_values(by='rank')

fig = px.timeline(df, y='genres', x_start='x_start', x_end='x_end', color_discrete_sequence=main_palette[1:])

fig.layout.xaxis.type = 'linear'
fig.data[0].x = df['dur_rel'].tolist()

fig.update_xaxes(range=[-1, 100], ticksuffix='%')
fig.update_yaxes(title='')

fig.update_layout(title='At what part the movie is the quietest?')

plot(fig)

# + code_folding=[0]
# # Where do the silences occur?


# # Test 2: Line
# smooth = 10
# df = silences_df.copy()
# df.loc[:, 'position_disc'] = np.ceil(df['pos_rel']/smooth)*smooth

# df = df\
#         .groupby(['imdb_id', 'position_disc', 'genres'], as_index=False)\
#         .agg({'dur_rel':'sum', 'silence_id':'count'})
        
# df = vectorize_column(df, 'genres')

# df = pd.DataFrame.explode(df, column='genres')

# df = df.groupby(['position_disc', 'genres'])\
#         .agg({'dur_rel':['median', 'mean'], 'imdb_id':'nunique'}).droplevel(0, axis=1)\
#         .reset_index()

# # df.loc[:, 'roll'] = df.sort_values(by='position_disc')['median'].rolling(10, center=True).mean()
# # df.loc[:, 'rank'] = df['median'].rank(ascending=False).astype(int)
# aux = df.groupby(['genres'], as_index=False).agg({'nunique':'max'})
# aux.loc[:, 'text'] = aux.apply(lambda x: f"<b>{x['genres']}</b> ({x['nunique']})", axis=1)
# aux_dict = aux.set_index('genres')['text'].to_dict()
                               
# category_orders = {'genres':aux.sort_values(by='nunique', ascending=False)['genres'].tolist()}
                            
                                                               
# fig = px.line(df, x='position_disc', y='mean', line_shape='linear', facet_col_spacing=0.05,
#               facet_col='genres', color_discrete_sequence=main_palette, category_orders = category_orders,
#               facet_col_wrap=5)
# fig.update_xaxes(ticksuffix='%', dtick=50, range=[0, 101], color='grey', title='')
# fig.update_xaxes(title='Position on movie', row=1, titlefont_color='black', col=3)

# fig.update_yaxes(rangemode='tozero', showgrid=False, title='', dtick=1, color='grey')
# fig.update_yaxes(ticksuffix='%')
# # fig.update_yaxes(col=1, ticksuffix='%')

# , title='Level of Silence', row=3
# fig.for_each_annotation(lambda x: x.update(text=aux_dict[x.text.split('=')[1]]))
# fig.update_layout(
#     title='Which part of a movie is the quietest?',
#     showlegend=False,
#     height=800
# )
# plot(fig)




# + code_folding=[0]
# # Scatter movie relationship year x rating x silence
# df = silences_df\
#         .groupby(['imdb_id', 'year', 'genres', 'rating', 'top_250_rank', 'title'], as_index=False)\
#         .agg({'dur_rel':'sum'})

# df = vectorize_column(df, 'genres')

# fig = px.scatter(df, x='year', y='dur_rel', hover_name='title')
# plot(fig)


# fig = px.box(df, x='rating', y='dur_rel', hover_name='title', points=False)
# plot(fig)


# fig = px.box(df, x='rating', y='dur_rel', hover_name='title', points=False)
# plot(fig)
# +
# Silence fup
# Most silent movies
# Black and White x Color
# -

# ## Dialogue

# +
# Speach follow up
# Overall time on dialogue
# Dialogue per genre

# + code_folding=[0]
# Dialogue view
df = movies_df.copy()

df.loc[:, 'words_per_minute'] = df['word_count']/(df['total_duration']/60)
df.loc[:, 'lines_per_minute'] = df['line_count']/(df['total_duration']/60)

df.loc[:, 'decade'] = (np.floor(df['year']/20)*20).astype(int)
df.loc[:, 'decade'] = df['decade'].apply(lambda x: f"{x} - {x+19}")

fig = px.strip(df.sort_values(by='decade'), x=['words_per_minute', 'lines_per_minute'], facet_col='variable',
               y='decade', color_discrete_sequence=px.colors.sequential.Agsunset, color='decade',
               hover_name='title', hover_data=['year', 'top_250_rank'])
fig.update_xaxes(matches=None, showticklabels=True, nticks=2, title='')
fig.update_layout(
    title = 'Which movies have the most dialogue?', 
    showlegend=False, 
    margin_t=120,
    yaxis_title='')

fig = facet_prettify(fig)
plot(fig)
# -


