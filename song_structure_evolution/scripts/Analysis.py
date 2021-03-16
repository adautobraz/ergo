# -*- coding: utf-8 -*-
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

# + [markdown] heading_collapsed=true
# # Setup

# + hidden=true
import os
import sys
import pandas as pd
import re
import json
from pathlib import Path
from fuzzywuzzy import fuzz
import plotly.express as px
import copy
import numpy as np
import umap
from scipy.spatial.distance import cdist
from scipy import stats
import string
from fuzzywuzzy import fuzz
from difflib import Differ
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots


pd.set_option('max_columns', None)

# + hidden=true
data_path = Path('./data/')

# + [markdown] heading_collapsed=true
# # Data Prep

# + hidden=true
raw_structure_df = pd.read_csv(data_path/'lyrics/prep/songs_paragraphs_raw.csv')
raw_structure_df.head(1)

# + hidden=true
# Clean structures that have too few

# + hidden=true
structure_df = raw_structure_df.loc[raw_structure_df['is_structure_tag']].sort_values(by=['song_id', 'order'])


structure_df.loc[:, 'structure_tags'] = structure_df['structure_tags'].apply(lambda x: eval(x))
structure_df.loc[:, 'structure_tags_map'] = structure_df['structure_tags_map'].apply(lambda x: eval(x))
structure_df.loc[:, 'all_tags_map'] = structure_df['all_tags_map'].apply(lambda x: eval(x))
structure_df.loc[:, 'line_repetition_dict'] = structure_df['line_repetition_dict'].apply(lambda x: eval(x))


structure_df.loc[:, 'total_paragraphs'] = structure_df['lyrics'].apply(lambda x: len(x.split('\n\n')))

structure_df.loc[:, 'lines_on_element'] = structure_df['section_lyrics'].fillna('').apply(lambda x: len([f for f in x.split('\n') if f.strip()]))
structure_df.loc[:, 'lines_on_lyrics'] = structure_df['lyrics'].apply(lambda x: len([f for f in x.split('\n') if f.strip()]))
# structure_df.loc[:, 'tags_on_lyrics'] = structure_df['structure_tags'].apply(lambda x: len(x))

# structure_df.loc[:, 'total_lines'] = structure_df['lines_on_lyrics'] - structure_df['tags_on_lyrics']

structure_df.loc[:, 'words_on_element'] = structure_df['section_lyrics'].apply(lambda x: len([w for w in re.split('[ \\n\']', str(x)) if w]))


# structure_df.loc[:, 'total_valid_tags'] = structure_df['structure_tags_map'].apply(lambda x: len([x for x if x['is_structure_tag']])))

structure_df.loc[:, 'rank'] = structure_df.groupby(['song_id'])['order'].rank()
structure_df.loc[:, 'max_rank'] = structure_df.groupby(['song_id'])['rank'].transform('max')

structure_df.loc[:, 'paragraph_pos_rel'] = 100*(structure_df['rank'] - 1)/(structure_df['max_rank'] - 1).fillna(0)

structure_df.loc[:, 'lines_previous_element'] = structure_df\
                                        .sort_values(['song_id', 'rank'])\
                                        .groupby(['song_id'])['lines_on_element'].shift(1).fillna(0)
structure_df.loc[:, 'line_pos_rel'] = 100*structure_df.groupby(['song_id'])['lines_previous_element'].cumsum()/structure_df['lines_on_lyrics']

structure_df.loc[:, 'year'] = structure_df['song_id'].apply(lambda x: x.split('_')[0]).astype(int)
structure_df.loc[:, 'decade'] = (np.floor(structure_df['year']/10)*10).astype(int)

structure_df.loc[:, 'chart_position'] = structure_df['song_id'].apply(lambda x: x.split('_')[1]).astype(int)

structure_df.loc[:, 'next_structure'] = structure_df.sort_values(by=['song_id', 'rank']).groupby(['song_id'])['tag_type'].shift(-1).fillna('end')
structure_df.loc[:, 'previous_structure'] = structure_df.sort_values(by=['song_id', 'rank']).groupby(['song_id'])['tag_type'].shift(1).fillna('start')

total_songs_per_year = structure_df.groupby(['year'])['song_id'].nunique().to_dict()

structure_df.head(1)


# + hidden=true
# structure_df.loc[structure_df['tag_type'] == 'other']

# + hidden=true
# structure_df.loc[ structure_df['max_rank']/structure_df['total_paragraphs'] < 0.3]

# + hidden=true
# print(structure_df.loc[structure_df['song_id'] == '2010_77'].iloc[0]['lyrics'])

# + hidden=true
def plot(fig):
    fig.update_layout(
        template = 'plotly_white',
        font_family='Fira Sans'
    )
    
    fig.show()
    
    
def generate_highlight_dict(color_dict, keys_highlight):
    new_dict = {}
    for k, v in color_dict.items():
        if k in keys_highlight:
            new_dict[k] = v
        else:
            new_dict[k] = px.colors.qualitative.Pastel[10]


# + hidden=true
def print_lyrics(song_id):
    print(structure_df.loc[structure_df['song_id'] == song_id].iloc[0]['lyrics'])


# + hidden=true
palette = px.colors.qualitative.Vivid
structure_elements = structure_df['tag_type'].value_counts().index.tolist()
# color_map_dict

main_elements = ['verse', 'chorus', 'bridge', 'outro', 'intro', 'pre-chorus']

elements = [f for f in structure_elements if 'chorus' not in f]
elements = ['intro', 'verse', 'bridge', 'outro', 'hook', 'instrumental', 'refrain', 'break', 'solo']

color_map_dict = {elements[i].title():palette[i]  for i in range(0, len(elements))}

palette_chorus = px.colors.sequential.Magenta
i=3
color_map_dict['Chorus'] = palette_chorus[i]
color_map_dict['Pre-Chorus'] = palette_chorus[i-2]
color_map_dict['Post-Chorus'] = palette_chorus[i+2]


color_map_dict['Start'] = 'rgb(204, 204, 204)'
color_map_dict['End'] = 'rgb(102, 102, 102)'
color_map_dict['Other'] = 'rgb(179, 179, 179)'

new_dict = copy.deepcopy(color_map_dict)
for k, v in color_map_dict.items():
    new_dict[k.lower()] = v
    
color_map_dict = new_dict
# -

# # Analysis

# ## General graphs

# + code_folding=[0, 2]
# How many songs do we have to analyze per year?

df = structure_df\
        .groupby(['year'], as_index=False)\
        .agg({'song_id':'nunique'})
              
fig = px.bar(df, x='year', y='song_id')
fig.update_layout(
    title='Songs with structure annotations, per year',
    yaxis_title='Total of songs',
    xaxis_title='Year'
)
plot(fig)

# + code_folding=[]
# Number of structures, per song

df = structure_df.drop_duplicates(subset=['song_id'])

fig = px.histogram(df, x='max_rank', histnorm='percent')
fig.update_layout(
    title='Distribution of number of elements, per song',
    yaxis_title='Percentage of songs (%)',
    xaxis_title='Number of elements'
)
plot(fig)

df.sort_values(by='chart_position')\
    .drop_duplicates(subset=['max_rank'])\
    .sort_values(by='year', ascending=False).head(1)


# + code_folding=[0]
# What are the most commont strutucture elements? How often they use to appear? 

df = structure_df\
    .groupby(['song_id', 'tag_type'], as_index=False)\
    .agg({'order':'count'})\
    .groupby(['tag_type'], as_index=False)\
    .agg({'song_id':'nunique', 'order':'mean'})\
    .sort_values(by=['song_id'], ascending=False)

df.columns = ['structure', 'song_count', 'average_song_appearance']

df.loc[:, 'total_songs'] = structure_df['song_id'].nunique()

df.loc[:, 'structure_type'] = df['structure'].str.title()


df.loc[:, 'presence_in_songs'] = 100*df['song_count']/df['total_songs']

fig = px.bar(df, y='structure_type', x=['presence_in_songs', 'average_song_appearance'], facet_col='variable')
fig.update_yaxes(categoryorder='total ascending', row=1, col=1)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

fig.update_xaxes(matches=None, showticklabels=False)
fig.update_xaxes(row=1, col=1, title='Percentage of songs<br>that use element', range=[0,110], dtick=100)
fig.update_xaxes(row=1, col=2, title='Elements per song', dtick=1)

fig.update_traces(texttemplate = '%{x:.1f}', textposition='outside', row=1, col=2)
fig.update_traces(texttemplate = '%{x:.1f}%', textposition='outside', row=1, col=1)

# fig.update_annotations(
#     annotations_font=12
# )

fig.update_layout(
    showlegend=False, 
    yaxis_title='',
    title = 'Most common song elements, and how much they appear per song',
    font_size=13,
    margin_t=100,
    margin_b=10
)

plot(fig)

# + code_folding=[0]
# Song element distribution

df = structure_df.copy()

df = df\
        .groupby(['song_id', 'tag_type'], as_index=False)\
        .agg({'order':'count'})\
        .groupby(['tag_type', 'order'], as_index=False)\
        .agg({'song_id':'nunique'})
        
        
df.loc[:, 'total_songs'] = structure_df['song_id'].nunique()
df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs']

df.loc[:, 'total_songs_with_element'] = df.groupby(['tag_type'])['song_id'].transform('sum')
df.loc[:, 'pct_distribution'] = 100*df['song_id']/df['total_songs_with_element']

df.loc[:, 'rank'] = df.groupby(['tag_type'])['pct_distribution'].rank(ascending=False)

df.loc[:, 'cumul_dist'] = df.sort_values(by='rank').groupby(['tag_type'])['pct_distribution'].cumsum()
df.loc[:, 'distinct'] = df.groupby(['tag_type'])['order'].transform('nunique')
df.loc[:, 'overall_rank'] = df['percentage'].rank(ascending=False)

grey = 160
color = 'rgb({0},{0},{0})'.format(grey)

# Top elements
top = df.loc[df['overall_rank'] <= 15].copy()
top.loc[:, 'text'] = top.apply(lambda x: "{} {}".format(x['order'], x['tag_type']), axis=1)
top.loc[(top['order'] > 1) & (top['tag_type'].str[-1] != 's'), 'text'] = top['text'] + 's'

fig = px.bar(top, y='overall_rank', x='percentage', color='tag_type', 
             orientation='h', color_discrete_map=color_map_dict,
             text='text'
            )

fig.update_yaxes(showticklabels=True, autorange='reversed', tickprefix='Top ', 
                 tickvals=[1, 5, 10, 15], title='Ranking')
fig.update_xaxes(ticksuffix='%', dtick=10, color=color, range=[0,65])
fig.update_xaxes(title='Percentage of songs with combination', titlefont_color='black')

fig.update_traces(textposition='outside')

fig.update_layout(title='Most common same-element combinations')

plot(fig)


# Distribution per element
fig = px.bar(df.sort_values(by='distinct'), x='order', y='pct_distribution', facet_col='tag_type', orientation='v',
             facet_col_wrap=4)

fig.update_yaxes(showticklabels=False, range=[0,120], dtick=100, title='', color=color)
fig.update_yaxes(showticklabels=True, col=1, ticksuffix='%')

fig.update_xaxes(showticklabels=True, range=[0.5,6.5], dtick=1, color=color)

fig.update_layout(
    showlegend=False, height=800,
    title='Song element quantity distribution',
    margin_t=120
)

fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
fig.for_each_annotation(lambda x: x.update(text="{}".format(x.text.split('=')[1].title()), font_color='black'))
plot(fig)


# -

# ## Structure time evolution

# + code_folding=[0]
# Song elements count, year evolution
df = structure_df\
        .groupby(['song_id', 'year'], as_index=False)\
        .agg({'words_on_element':'sum', 'lines_on_element':'sum',
              'lines_on_lyrics':'mean', 'total_paragraphs':'mean',
              'tag_type':'count'
             })\
        .groupby(['year'], as_index=False)\
        .agg({'words_on_element':'median', 
              'lines_on_lyrics':'median', 
              'total_paragraphs':'mean',
              'tag_type':'mean'
             })

fig = px.line(df, x='year', y=['words_on_element', 'lines_on_lyrics', 'total_paragraphs', 'tag_type'], 
              facet_col='variable', facet_col_spacing=0.1)
fig.update_yaxes(matches=None, showticklabels=True, rangemode='tozero')
fig.update_xaxes(title='Year', showticklabels=True)

# fig.update_xaxes(row=1, col=1, title='Year', showticklabels=True)


fig.update_layout(
    title='Song elements of Billboard Hot 100 songs, per year',
    showlegend=False,
    margin_t=100
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))
plot(fig)
# + code_folding=[0]
# How does the use of each element evolve over time? 

df = structure_df\
    .groupby(['song_id', 'tag_type', 'year'], as_index=False)\
    .agg({'order':'count', 'lines_on_element':'sum', 'lines_on_lyrics':'mean'})

df.loc[:, 'song_share'] = 100*df['lines_on_element']/df['lines_on_lyrics']

df = df\
        .groupby(['tag_type', 'year'], as_index=False)\
        .agg({'song_id':'nunique', 'order':'mean', 'song_share':'median'})\
        .sort_values(by=['year'], ascending=False)

df.columns = ['song_element', 'year', 'songs', 'average_song_appearance', 'share_in_song']

df.loc[:, 'total_songs'] = df['year'].apply(lambda x: total_songs_per_year[x])
df.loc[:, 'Element'] = df['song_element'].str.title()
df.loc[:, 'presence_in_songs'] = 100*df['songs']/df['total_songs']

df = df.loc[df['song_element'].isin(main_elements)]

fig = px.line(df, x='year', y='presence_in_songs', facet_col='Element', 
              color='Element', facet_col_spacing=0.1, facet_col_wrap=3, 
              color_discrete_map=color_map_dict
             )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

fig.update_yaxes(range=[-10,110])
fig.update_yaxes(col=1, title='Songs using', ticksuffix='%')
fig.update_xaxes(title='Year', row=1)

fig.update_layout(
    showlegend=False, 
    title = 'How do song elements usage change over time?',
    font_size=13,
    margin_t=100
)

plot(fig)


# df = df.loc[df['song_element'].isin(['verse'])]

fig = px.line(df, x='year', y='average_song_appearance', facet_col='Element', 
              color='Element', facet_col_spacing=0.1, facet_col_wrap=3, 
              color_discrete_map=color_map_dict
             )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

fig.update_yaxes(range=[0,4])
fig.update_yaxes(col=1, title='Presence in song')
fig.update_xaxes(title='Year', row=1)

fig.update_layout(
    showlegend=False, 
    title = 'How much of each song element is used in songs?',
    font_size=13,
    margin_t=100
)

plot(fig)


fig = px.line(df, x='year', y='share_in_song', facet_col='Element', 
              color='Element', facet_col_spacing=0.1, facet_col_wrap=3, 
              color_discrete_map=color_map_dict
             )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

# fig.update_yaxes(range=[0,4])
fig.update_yaxes(col=1, title='Presence in song', ticksuffix='%')
fig.update_xaxes(title='Year', row=1)

fig.update_layout(
    showlegend=False, 
    title = 'How much does each element represent of the song?',
    font_size=13,
    margin_t=100
)

plot(fig)
# -

# ## How does a song start?

# + code_folding=[0]
# How does a song start? General view of first element on song structure
aux = structure_df.loc[structure_df['rank'] == 1].copy()

# Line graph
df = aux\
        .groupby(['tag_type', 'rank'], as_index=False)\
        .agg({'song_id':'count'})\
        .sort_values(by=['rank'])

df.loc[:, 'Element'] = df['tag_type'].str.title()

df.loc[:, 'total_songs_on_year'] = df['song_id'].sum()
df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs_on_year']

fig = px.bar(df, y='Element', x='percentage')
fig.update_yaxes(categoryorder='total ascending', title='')
fig.update_xaxes(ticksuffix='%', title='% of songs starting with element', dtick=50)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_layout(
    title='How does a song start?<br>A view on the first element present on song structure',
    margin_t=120
)
plot(fig)

# + code_folding=[0]
# # How does the use of each element evolves on time? 

# df = structure_df\
#     .groupby(['song_id', 'tag_type', 'year'], as_index=False)\
#     .agg({'order':'count'})\
#     .groupby(['tag_type', 'year'], as_index=False)\
#     .agg({'song_id':'nunique', 'order':'mean'})\
#     .sort_values(by=['year'], ascending=False)

# df.columns = ['song_element', 'year', 'songs', 'average_song_appearance']

# df = df.loc[df['song_element'].isin(['chorus', 'verse', 'bridge', 'outro', 'intro', 'pre-chorus'])]

# df.loc[:, 'total_songs'] = df['year'].apply(lambda x: total_songs_per_year[x])

# df.loc[:, 'Element'] = df['song_element'].str.title()


# df.loc[:, 'presence_in_songs'] = 100*df['songs']/df['total_songs']


# fig = px.line(df, x='year', y=['presence_in_songs', 'average_song_appearance'], 
#               facet_col='variable', color='Element', facet_col_spacing=0.1)

# fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

# fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_yaxes(row=1, col=2, range=[-0.1, 4])
# fig.update_yaxes(row=1, col=1, ticksuffix='%')

# fig.update_xaxes(title='Year')

# fig.update_layout(
#     showlegend=True, 
#     yaxis_title='',
#     title = 'Evolution of most common song elements presence and appereance',
#     font_size=13,
#     margin_t=100
# )

# plot(fig)
# -

song_info_dict = structure_df\
                    .drop_duplicates(subset=['song_id'])\
                    .set_index(['song_id'])\
                    .loc[:, ['name', 'artist_chart', 'year', 'decade']]\
                    .to_dict(orient='index')


def get_example_song(raw_df):
    df = raw_df.copy()
    
    df.loc[:, 'song_example_id'] = df.apply(lambda x: '{}_{}'.format(x['year'], x['chart_position']), axis=1)
    df.loc[:, 'song_example'] = df['song_example_id'].apply(lambda x: song_info_dict[x]['name'])                                      
    df.loc[:, 'artist'] = df['song_example_id'].apply(lambda x: song_info_dict[x]['artist_chart'])                                      
    df.loc[:, 'chart_position'] = df['chart_position'] + 1                                   

    return df


# + code_folding=[0]
# How does a song start? View per year
aux = structure_df.loc[structure_df['rank'] <= 1].copy()

aux.loc[:, 'category'] = 'Other'
aux.loc[aux['tag_type'].isin(['intro', 'chorus', 'verse']), 'category'] = aux['tag_type'].str.title()


df = aux\
        .groupby(['category','year'], as_index=False)\
        .agg({'song_id':'count', 'chart_position':'min'})

df.loc[:, 'total_songs_on_year'] = df.groupby(['year'])['song_id'].transform('sum')

df = get_example_song(df)
df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs_on_year']

df.loc[:, 'rolling_percentage'] = df.loc[:, 'percentage'].rolling(window=2).mean()

df.loc[:, 'First Element'] = df['category']

fig = px.line(df, x='year', y='rolling_percentage', color='First Element', 
              color_discrete_map=color_map_dict, hover_name='song_example', hover_data=['artist', 'chart_position'])
fig.update_layout(
    title='How do songs start?<br>A yearly evolution on the type of first song element',
    yaxis_title='Songs starting with element',
    yaxis_ticksuffix='%',
    xaxis_title='Year'
)
plot(fig)
# -

# ## Pop overture

# + code_folding=[0]
# Does intro repeats?

df = structure_df\
        .sort_values(by=['song_id', 'rank'])\
        .drop_duplicates(subset=['song_id'])

df.loc[:, 'repetition_of_element_lines'] = df.apply(lambda x: [int(v) for k,v in x['line_repetition_dict'].items() if k <= x['lines_on_element']], axis=1)
df.loc[:, 'repeats_later'] = df['repetition_of_element_lines'].apply(lambda x: max(x) if x else 0).astype(bool)

df.loc[:, 'class'] = 'Other'
df.loc[df['tag_type'].isin(['verse', 'chorus', 'intro']), 'class'] = df['tag_type'].str.title()
df.loc[df['tag_type'] == 'intro', 'class'] = 'Intro' + df['repeats_later'].apply(lambda x: ' (Peek)' if x else '')

# df.loc[df['class'] == 'intro', 'class'] = 'intro' + df['next_structure'].apply(lambda x: ' - peek' if x not in ['verse', 'end'] else '')
df.loc[df['class'] == 'Intro', 'class'] = 'Intro' + df['next_structure'].apply(lambda x: ' + ' + x.title() if x in ['verse', 'chorus'] else '')

# Lyrical foreshadowing
df = df\
        .groupby(['year', 'class'], as_index=False)\
        .agg({'song_id':'nunique', 'chart_position':'min'})

df.loc[:, 'type'] = 'detail'
df.loc[:, 'pop_overture'] = False
df.loc[df['class'].isin(['Chorus', 'Intro (Peek)', 'Intro + Chorus']), 'pop_overture'] = True

df.loc[:, 'total_songs'] = df.groupby(['year'])['song_id'].transform('sum')
df.loc[:, 'rate'] = 100*df['song_id']/df['total_songs']

df.loc[:, 'smooth'] = df.sort_values(by='year').groupby(['class'])['rate'].apply(lambda x: x.rolling(window=2).mean())

tot_df = df.groupby(['year', 'pop_overture'], as_index=False).agg({'song_id':'sum', 'chart_position':'min'})

tot_df = get_example_song(tot_df)

tot_df.loc[:, 'total_songs'] = tot_df.groupby(['year'])['song_id'].transform('sum')
tot_df.loc[:, 'rate'] = 100*tot_df['song_id']/tot_df['total_songs']

tot_df = tot_df.loc[tot_df['pop_overture']].sort_values(by='year')

tot_df.loc[:, 'smooth'] = tot_df['rate'].rolling(window=2).mean()
fig = px.line(tot_df, x='year', y='smooth', hover_name='song_example', hover_data=['artist', 'chart_position'])
fig.update_yaxes(rangemode='tozero', ticksuffix='%', dtick=20, title='Songs with a Pop Overture')
fig.update_xaxes(title='Year', range=[1958, 2021])

fig.update_layout(title='Occurrence of Pop Overture on songs, per year')
fig.add_vline(x=2014, line_width=2, 
              line_dash="dash", line_color="grey",
              annotation_text="2014", 
              annotation_position="bottom")
plot(fig)

df.loc[:, 'pop_overture_txt'] = df['pop_overture'].apply(lambda x: 'w/ Pop Overture' if x else 'No inversion')
df.loc[:, 'Structure Type'] = df['class']
df = get_example_song(df)

fig = px.line(df, x='year', y='smooth', color='Structure Type', facet_col='pop_overture_txt', 
              facet_col_spacing=0.05, color_discrete_map=color_map_dict, hover_name='song_example', 
              hover_data=['artist', 'chart_position']
             )
fig.update_layout(title='Song beginnings evolution, per year', 
                  yaxis_title='Share of songs',
                  margin_t=120
                 )
fig.update_xaxes(title='Year', color='grey', range=[1956, 2020])
fig.update_xaxes(color='black', col=1)
fig.update_yaxes(rangemode='tozero', ticksuffix='%', dtick=20)
fig.for_each_annotation(lambda x: x.update(text=x.text.split('=')[1]))
plot(fig)


# -

# ## Where does each element go?

# + code_folding=[0]
# Where in each song does each element stay? Viewing element density distribution

# # Divide in how many groups?
# px.histogram(structure_df, x='paragraph_pos_rel', color='tag_type').show()

# # How many annotations
# df = structure_df\
#         .loc[~structure_df['tag_type'].isin(['other'])]\
#         .groupby(['song_id'], as_index=False).agg({'rank':'max'})

# px.histogram(df, x='rank')


df = structure_df.copy()

smooth = 10

df.loc[:, 'pos_disc'] = np.floor(df['paragraph_pos_rel']/smooth)*smooth
        
group_df = df.groupby(['tag_type','pos_disc'], as_index=False).agg({'song_id':'count'})

group_df.loc[:, 'total'] = group_df.groupby(['tag_type'])['song_id'].transform('sum')

group_df.loc[:, 'percentage'] = 100*group_df['song_id']/group_df['total']

group_df.loc[:, 'max_norm'] = group_df.groupby(['tag_type'])['percentage'].transform('max')
group_df.loc[:, 'percent_norm'] = group_df['percentage']/group_df['max_norm']


# View 1
# fig = px.line(group_df, x='pos_disc', y='percent_norm', color='tag_type', line_shape='spline',
#              facet_col='tag_type',  facet_col_wrap=7)
# fig.update_yaxes(matches=None, rangemode='tozero', showticklabels=False)
# fig.update_xaxes(matches=None, rangemode='tozero', showticklabels=False, title='')
# fig.update_xaxes(row=1, col=1, showticklabels=True)

# fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))
# fig.update_traces(fill='tozeroy')
# fig.update_layout(showlegend=False)

# plot(fig)


# View 2
# top = df.loc[df['tag_type'].isin(['verse', 'chorus', 'pre-chorus', 'post-chorus', 'bridge'])]
# fig = px.violin(top, x='paragraph_pos_rel', color='tag_type',
#                 points=False, violinmode='overlay')

# fig.update_traces(side='positive', spanmode='hard', width=2, opacity=0.5)

# # fig.update_yaxes(matches=None)#=[-1,1])
# fig.update_xaxes(showticklabels=False, title='')#=[-1,1])
# fig.update_xaxes(showticklabels=True, row=1, col=1)#=[-1,1])

# fig.update_layout(showlegend=True)
# fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

# plot(fig)


# View 3
# df.loc[:, 'scale'] = 'others'
# df.loc[df['tag_type'].isin(['intro', 'outro']), 'scale'] = 'intro/outro'

# # df = df.loc[df['scale'] == 'others']

# fig = px.violin(df, x='line_pos_rel', color='tag_type', facet_col='tag_type',
#                 points=False, violinmode='overlay', facet_col_wrap=7)

# fig.update_traces(side='positive', spanmode='hard', width=2)

# # fig.update_yaxes(matches=None)#=[-1,1])
# fig.update_xaxes(showticklabels=False, title='')#=[-1,1])
# fig.update_xaxes(showticklabels=True, row=1, col=1, ticksuffix='%')#=[-1,1])

# fig.update_layout(
#     title='Where does each element go?<br>Viewing each song structure type distribution',
#     margin_t=120,
#     showlegend=False)

# fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

# plot(fig)

# View 4
df = structure_df.copy()
df.loc[:, 'text'] = df['tag_type'].str.title()
df.loc[:, 'average_point'] = df.groupby(['tag_type'])['line_pos_rel'].transform('mean')

palette_match = {
    0:['intro', 'instrumental'],
    1:['verse', 'pre-chorus', 'chorus', 'post-chorus'],
    2:['solo', 'bridge','interlude', 'break', 'outro'],
    3:['hook', 'refrain']
}

color_dict = {}
palette = px.colors.qualitative.Safe
for i, tag_list in palette_match.items():
    for tag in tag_list:
        color_dict[tag] = palette[i]

category_orders = [i.title() for l in palette_match.values() for i in l]

fig = px.violin(df, y='text', x='line_pos_rel',
                 color='tag_type', color_discrete_map=color_dict,
#                  category_orders={'text':df.sort_values(by='average_point')['text'].unique().tolist()}
                 category_orders={'text':category_orders}
                
               )

fig.update_traces(orientation='h', side='positive', width=3, scalemode='width',
                  points=False, spanmode='hard')
fig.update_yaxes(showgrid=True, title='')
fig.update_xaxes(dtick=25, title='Position on song', ticksuffix='%', range=[-3, 101])

fig.update_layout(
    title='Where does each element go?<br>Viewing each song structure position distribution',
    showlegend=False,
    height=800
)

plot(fig)
# + code_folding=[]
# What comes after/before each element?
array = []
cols = ['tag_type', 'structure', 'counter', 'type']
next_df = structure_df\
            .groupby(['tag_type', 'next_structure'], as_index=False)\
            .agg({'song_id':'count'})

next_df.loc[:, 'type'] = '3-After'
next_df.columns = cols
array.append(next_df)

prev_df = structure_df\
            .groupby(['tag_type', 'previous_structure'], as_index=False)\
            .agg({'song_id':'count'})

prev_df.loc[:, 'type'] = '1-Before'
prev_df.columns = cols
array.append(prev_df)

mid_df = prev_df.copy()
mid_df.loc[:, 'type'] = '2-Middle'

array.append(mid_df)

df = pd.concat(array, axis=0)

df.loc[:, 'total_count'] = df.groupby(['tag_type', 'type'])['counter'].transform('sum')
df.loc[:, 'percentage'] = 100*df['counter']/df['total_count']

df.loc[:, 'rank'] = df.groupby(['tag_type', 'type'])['percentage'].rank(ascending=False)

df = df.loc[df['rank'] <= 1].sort_values(by=['type', 'rank']).reset_index()

df.loc[df['type'] == '2-Middle', 'percentage'] = 0
# df.loc[df['type'] == '1-Before', 'percentage'] = -df['percentage']

df.loc[:, 'Structure'] = df['structure'].str.title()
df.loc[:, 'y_axis'] = df['tag_type'].str.title()


# Visualization
fig = make_subplots(rows=1, cols=3, column_widths=[0.48, 0.02, 0.48], 
                    subplot_titles=("Before", "<b>Element</b>", "After")
                   )

types = sorted(df['type'].unique().tolist())

df = df.sort_values(by=['y_axis'])
color_list = df['Structure'].unique().tolist()
shown = []

# Add bars to viz
for i in range(0, len(types)):
    info = df.loc[df['type'] == types[i]].sort_values(by='y_axis')
    for e in info['Structure'].unique().tolist():
        bars = info.loc[info['Structure'] == e].sort_values(by='y_axis')
        
        if e in shown:
            fig.add_trace(go.Bar(y=bars['y_axis'], x=bars['percentage'],
                                orientation='h', name=e, legendgroup=e,
                                marker_color=color_map_dict[e],
                                 showlegend=False
                                ),
                          row=1, col=i + 1)
        else:
            fig.add_trace(go.Bar(y=bars['y_axis'], x=bars['percentage'],
                    orientation='h', name=e, legendgroup=e,
                    marker_color=color_map_dict[e],
                    showlegend=True
                    ),
              row=1, col=i + 1) 
            shown.append(e)

elements = sorted(df['y_axis'].unique().tolist())
fig.update_yaxes(categoryorder='category descending', row=1)

# Add elements of 
for e in elements:
    fig.add_annotation(
        xref="x",
        yref="y",
        showarrow=False,
        x=0,
        align="center",
        y=e,
        text=e,
        font_size=14,
        row=1,
        col=2
    )
    

fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside', col=1)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside', col=3)
    
fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showticklabels=True, ticksuffix='%', dtick=100, title='Occurrence')

fig.update_xaxes(showticklabels=False, col=2, zeroline=False, showgrid=False, title='')
fig.update_xaxes(col=1, range=[130, 0])
fig.update_xaxes(col=3, range=[0, 130])
# fig.update_xaxes(col=1, autorange='reversed', range=[120, 0])

fig.layout.annotations[0].update(y=1.03, font_size=14)
fig.layout.annotations[1].update(y=1.03, font_size=14)
fig.layout.annotations[2].update(y=1.03, font_size=14)
fig.update_layout(
                  title='Most common structures, before and after each element',
                  margin_t=120,
                 )

fig.update_layout(legend=dict(
    yanchor="top",
    y=1,
    orientation='v', 
    title_text='Element',
    xanchor="left",
    x=0.95
))

plot(fig)


# + code_folding=[0]
# # What comes after/before each element?

# cols = ['tag_type', 'structure', 'counter', 'type']
# next_df = structure_df\
#             .groupby(['tag_type', 'next_structure'], as_index=False)\
#             .agg({'song_id':'count'})

# next_df.loc[:, 'type'] = 'After'
# next_df.columns = cols

# prev_df = structure_df\
#             .groupby(['tag_type', 'previous_structure'], as_index=False)\
#             .agg({'song_id':'count'})

# prev_df.loc[:, 'type'] = 'Before'
# prev_df.columns = cols

# df = pd.concat([next_df, prev_df], axis=0)

# df.loc[:, 'total_count'] = df.groupby(['tag_type', 'type'])['counter'].transform('sum')
# df.loc[:, 'percentage'] = 100*df['counter']/df['total_count']

# df.loc[:, 'rank'] = df.groupby(['tag_type', 'type'])['percentage'].rank(ascending=False)

# df = df.loc[df['rank'] <= 1].sort_values(by='rank')

# df.loc[:, 'Structure'] = df['structure'].str.title()
# df.loc[:, 'y_axis'] = df['tag_type'].str.title()

# fig = px.bar(df, y='y_axis', color='Structure', x='percentage', 
#              facet_col='type', color_discrete_map=color_map_dict)

# # fig.update_yaxes(col=1, categoryorder='category ascending')
# fig.update_xaxes(title='Ocurrence', ticksuffix='%', range=[0,110], dtick=50)
# fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
# fig.for_each_annotation(lambda x: x.update(text=x.text.split('=')[1]))
# fig.update_layout(
#     title='What comes before and after each song element?',
#     yaxis_title=''
# )

# # fig.update_xaxes(autorange='reversed', row=1, col=1)
# # fig.update_xaxes(autorange=True, row=1, col=2)
# # fig.update_yaxes(showticklabels=False)

# plot(fig)

# + code_folding=[0]
# # Position investigation on song, per rank order
# df = structure_df\
#         .groupby(['rank', 'tag_type'], as_index=False)\
#         .agg({'song_id':'nunique'})

# df.loc[:, 'total_songs_rank'] = df.groupby(['rank'])['song_id'].transform('sum')

# df.loc[:, 'max_songs'] = df['total_songs_rank'].max()

# df.loc[:, 'percentage_of_songs'] = 100*df['total_songs_rank']/df['max_songs']

# df = df.loc[df['percentage_of_songs'] >= 10]

# df.loc[:, 'percentage_position'] = df['song_id']/df['total_songs_rank']

# fig = px.line(df, x='rank', y='percentage_position', color='tag_type')
# plot(fig)
# -

# ## Structure sequence

# + code_folding=[0, 1]
# Data prep
def check_diff(tags_replacer, back_dict, structure, ref):
    
    ref_seq = ''.join([tags_replacer[f] for f in ref.split('_')])
    structure_seq = ''.join([tags_replacer[f] for f in structure.split('_')])
    
    dif = Differ()
    diffs = list(dif.compare(ref_seq, structure_seq))
    
    diff_tags = []
    for d in diffs:
        diff_tags.append(d[:-1] + back_dict[d[-1]])
    
    return diff_tags


df = structure_df.copy()

seq_df = df.groupby(['song_id', 'decade', 'year', 'chart_position'])['tag_type'].apply(list).to_frame().reset_index()

seq_df.loc[:, 'structure'] = seq_df['tag_type'].apply(lambda x: '_'.join(x))
seq_df.loc[:, 'structure_size'] = seq_df['tag_type'].apply(lambda x: len(x))

seq_df.loc[:, 'repetition'] = seq_df.groupby(['decade', 'structure'])['song_id'].transform('count')
seq_df.loc[:, 'total_songs'] = seq_df.groupby(['decade'])['song_id'].transform('count')

seq_df.loc[:, 'rank'] = seq_df.groupby(['decade'])['repetition'].rank(ascending=False, method='dense')

most_common_dec_structure = seq_df\
                            .loc[seq_df['rank'] == 1]\
                            .drop_duplicates(subset=['decade', 'structure'])\
                            .set_index('decade')['structure'].to_dict()

seq_df.loc[:, 'top_structure'] = seq_df['decade'].apply(lambda x: most_common_structure[x])


tags = structure_df['tag_type'].unique().tolist()
tags_replacer = {tags[i]: string.ascii_uppercase[i] for i in range(0, len(tags))}
back_dict = {v:k for k,v in tags_replacer.items()}

seq_df.loc[:, 'diff_to_top'] = seq_df.apply(lambda x: check_diff(tags_replacer, back_dict, 
                                                              x['structure'], x['top_structure']), axis=1)

seq_df.loc[:, 'distance_to_top'] = seq_df['diff_to_top'].apply(lambda x: len([f for f in x if '+ ' in f or '- ' in f]))
seq_df.head(1)
# -

df_elem

# + code_folding=[0]
# What are the most common song structures of all time? 

# Data Prep
rank_df = seq_df\
            .groupby(['structure', 'structure_size'], as_index=False)\
            .agg({'song_id':'nunique'})\
            .sort_values(by='structure_size')

rank_df.loc[:, 'rank'] = rank_df['song_id'].rank(method='first', ascending=False)
rank_dict = rank_df.set_index('structure')['rank'].to_dict()

df = seq_df.copy()

df.loc[:, 'struct_count'] = df.groupby(['structure'])['song_id'].transform('count')
df.loc[:, 'struct_rank'] = df['structure'].apply(lambda x: rank_dict[x])

rank_cut = 5
top_df = df.loc[df['struct_rank'] <= rank_cut]
songs_top_structs = top_df['song_id'].unique().tolist()

df_elem = structure_df.loc[structure_df['song_id'].isin(songs_top_structs)]
df_elem = pd.merge(left=df_elem, right=top_df.loc[:, ['song_id','structure', 'struct_rank', 'struct_count']], on='song_id')

df_elem.loc[:, 'line_percentage'] = (100*df_elem['lines_on_element']/df_elem['lines_on_lyrics']).replace(0, 100)

df = df_elem\
        .groupby(['struct_rank', 'structure', 'tag_type', 'rank', 'struct_count'], as_index=False)\
        .agg({'line_percentage':'mean', 'lines_on_element':'mean'})\
        .sort_values(by=['struct_rank', 'rank'])


df.loc[:, 'total_size'] = df.groupby(['structure'])['line_percentage'].transform('sum')
df.loc[:, 'relative_size'] = 100*df['line_percentage']/df['total_size'] 
df.loc[:, 'text'] = df['tag_type'].apply(lambda x: '<br>'.join(x.split('-')).title())
df.loc[:, 'song_percentage'] = 100*df['struct_count']/structure_df['song_id'].nunique()
df.loc[:, 'y_text'] = df.apply(lambda x: "<b>Top {:.0f}</b> ({:.1f}%)".format(x['struct_rank'], x['song_percentage']), axis=1)

df.loc[:, 'line_disc'] = np.round(df['lines_on_element'])

df.loc[:, 'pos_end'] = df.groupby(['structure'])['line_disc'].cumsum()
df.loc[:, 'pos_start'] = df.groupby(['structure'])['pos_end'].shift(1).fillna(0)
df.loc[:, 'delta'] = df['pos_end'] - df['pos_start']
df.loc[:, 'Structure'] = df['tag_type'].str.title()
# df.loc[:, 'pos_end_order'] = df.groupby(['structure'])['rank'].cumcount() + 1
# df.loc[:, 'pos_start_order'] = df.groupby(['structure'])['pos_end_order'].shift(1).fillna(0).astype(int)
# df.loc[:, 'delta'] = df['pos_end_order'] - df['pos_start_order']

# # Plot gant (V0)
# fig = px.timeline(df, x_start="pos_start", x_end="pos_end", y="struct_rank", 
#                   color="tag_type", text='text', color_discrete_map=color_map_dict)

# fig.update_xaxes(type='linear')
# for i in range(0, len(fig.data)):
#     tag = fig.data[i].name
#     array = df.loc[df['tag_type'] == tag].sort_values(by=['struct_rank', 'pos_start'])
#     fig.data[i]['x'] = array['relative_size'].tolist()

# fig.update_traces(textposition='inside', width=0.8)
# fig.update_traces(insidetextanchor='middle', textfont_size=10)

# fig.update_layout(showlegend=False)
# fig.update_yaxes(autorange='reversed', title='', tickprefix='Top ', dtick=1)
# plot(fig)


# Plot gant (V1)
fig = px.timeline(df, x_start="pos_start", x_end="pos_end", y="y_text", 
                  color="tag_type", text='text', color_discrete_map=color_map_dict)

fig.update_xaxes(type='linear')
for i in range(0, len(fig.data)):
    tag = fig.data[i].name
    array = df.loc[df['tag_type'] == tag].sort_values(by=['struct_rank', 'pos_start'])
    fig.data[i]['x'] = array['delta'].tolist()

fig.update_traces(textposition='inside', width=0.8)
fig.update_traces(insidetextanchor='middle', textfont_size=12)

fig.update_layout(
    title='Most used song structures on Billboard Hot 100, from 1960 to 2020',
    showlegend=False)
fig.update_yaxes(autorange='reversed', title='')
# fig.update_xaxes(showticklabels=True, tickmode='array', tickvals=[0], ticktext=['Song start'])    
fig.update_xaxes(showticklabels=True, title='Song line', ticksuffix='th', dtick=10)                               
plot(fig)

df_elem.loc[:, '_year'] = -df_elem['year']
df_elem.sort_values(by=['chart_position', '_year'])\
        .drop_duplicates(subset=['structure'])\
        .loc[:, ['song_id', 'name', 'artist_chart', 'structure']]

# + code_folding=[0]
# What is the most common song structure of each decade?

df = seq_df.copy()

df.loc[:, 'song_count'] = df.groupby(['decade', 'structure'])['song_id'].transform('nunique')
df.loc[:, 'st_rank'] = df.groupby(['decade'])['song_count'].rank(ascending=False, method='dense')

top_df = df.loc[df['st_rank'] == 1]
songs_top_structs = top_df['song_id'].unique().tolist()

df_elem = structure_df.loc[structure_df['song_id'].isin(songs_top_structs)]
df_elem = pd.merge(left=df_elem, right=top_df.loc[:, ['song_id','structure', 'st_rank', 'song_count', 'total_songs']], on='song_id')

df = df_elem\
        .groupby(['decade', 'structure', 'tag_type', 'rank', 'song_count', 'st_rank', 'total_songs'], as_index=False)\
        .agg({'lines_on_element':'mean'})\
        .sort_values(by=['song_count', 'rank'])

df.loc[:, 'lines_disc'] = np.round(df['lines_on_element'])
# df.loc[:, 'pos_end'] = df.groupby(['decade', 'structure'])['rank'].cumcount() + 1

df.loc[:, 'pos_end'] = df.groupby(['decade', 'structure'])['lines_disc'].cumsum()


df.loc[:, 'pos_start'] = df.groupby(['decade', 'structure'])['pos_end'].shift(1).fillna(0)

# df.loc[df['tag_type'] == 'instrumental', 'pos_end'] = 5

df.loc[:, 'delta'] = df['pos_end'] - df['pos_start']

df.loc[:, 'song_percentage'] = 100*df['song_count']/df['total_songs']
df.loc[:, 'y_text'] = df.apply(lambda x: f"<b>{x['decade']}</b> ({x['song_percentage']:.1f}%)", axis=1)
df.loc[:, 'text'] = df['tag_type'].apply(lambda x: '<br>'.join(x.split('-')).title())
df.loc[df['tag_type'] == 'instrumental', 'text'] = 'Instr.'
                               
# Plot gant (V1)
fig = px.timeline(df.sort_values(by=['y_text', 'rank']), x_start="pos_start", x_end="pos_end", y="y_text", 
                  color="tag_type", text='text', color_discrete_map=color_map_dict)

fig.update_xaxes(type='linear')

for i in range(0, len(fig.data)):
    tag = fig.data[i].name
    array = df.loc[df['tag_type'] == tag].sort_values(by=['y_text', 'pos_start'])
    fig.data[i]['x'] = array['delta'].tolist()

fig.update_traces(textposition='inside', width=0.8)
fig.update_traces(insidetextanchor='middle', textfont_size=12)

fig.update_layout(
    title='Most used song structures, per decade',
    showlegend=True)
fig.update_yaxes(title='')
fig.update_xaxes(showticklabels=True, title='Song line', ticksuffix='th')                               
plot(fig)
                               
                               
df_elem.sort_values(by=['chart_position', 'year'])\
        .drop_duplicates(subset=['structure', 'decade'], keep='first')\
        .loc[:, ['song_id', 'name', 'artist_chart', 'decade', 'structure']]

# + code_folding=[0]
# Difference from most common structure distribution 

# View option 1
fig = px.violin(seq_df, x='distance_to_top', color='decade', y='decade', color_discrete_sequence=px.colors.sequential.Burg)

fig.update_traces(orientation='h', side='positive', width=20, scalemode='width',
                  points=False, spanmode='hard')

fig.update_layout(
    title='How different from the most common structure are the rest of the songs on the decade?',
    showlegend=False,
    yaxis_title='',
    xaxis_title='Difference in elements',
)
plot(fig)


# # View option 2

# fig = px.histogram(seq_df, x='distance_to_top', facet_col='decade', 
#                    histnorm='percent', opacity=0.9)
# fig.update_xaxes(title='', color='grey')
# fig.update_xaxes(title='Elements of diference', color='black', col=1)
# fig.update_yaxes(title='Percentage of songs', col=1, color='black', ticksuffix='%')
# fig.for_each_annotation(lambda x: x.update(text = f"<b>{x.text.split('=')[1]}<b>"))
# plot(fig)



# + code_folding=[0]
# Most common difference from most common structure, expressed as sum of changes on quantity of each element

songs_per_decade = structure_df.groupby(['decade']).agg({'song_id':'nunique'})['song_id'].to_dict()

diff_df = seq_df.copy()

diff_df = pd.DataFrame.explode(seq_df, column='diff_to_top')

diff_df = diff_df.loc[diff_df['diff_to_top'].apply(lambda x: bool(re.search('^[+-] ', x)))]

df = diff_df\
        .groupby(['decade', 'diff_to_top', 'song_id'], as_index=False)\
        .agg({'structure':'count'})\
        .groupby(['decade', 'diff_to_top', 'structure'], as_index=False)\
        .agg({'song_id':'nunique'})

df.loc[:, 'quant'] = df.apply(lambda x: -x['structure'] if '- ' in x['diff_to_top'] else x['structure'], axis=1)

df.loc[:, 'tag'] = df['diff_to_top'].apply(lambda x: x[1:].strip())
df.loc[:, 'quant_text'] = df.apply(lambda x: f"{x['diff_to_top'][0]}{abs(x['quant'])} {x['tag']}" , axis=1)

df.loc[:, 'total_songs'] = df['decade'].apply(lambda x: songs_per_decade[x])
df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs']
df.loc[:, 'rank'] = df.groupby(['decade'])['song_id'].rank(ascending=False).astype(int)

top = df.loc[df['rank'] <= 3].copy().sort_values(by='rank')

df

top.loc[:, 'size'] = 1/df['rank']

top.loc[:, 'text'] = df.apply(lambda x: f"{x['quant']} {x['tag'].title()}", axis=1)

fig = px.bar(top, y='decade', x='percentage', color='tag', orientation='h', facet_col='rank',
             color_discrete_map=color_map_dict, text='quant_text'
            )

fig.update_yaxes(autorange='reversed', showgrid=False, title='')
fig.update_xaxes(ticksuffix='%', title='', color='grey', dtick=50, range=[0,110])
fig.update_xaxes(ticksuffix='%', title='', color='black', col=1)

fig.for_each_annotation(lambda x: x.update(text=f"Top {x.text.split('=')[1]}"))
fig.update_layout(
    title='Most common additions on song structure, from base structure',
    margin_t = 120,
    legend=dict(
    yanchor="bottom",
    y=-0.3,
    xanchor="left",
    x=-0.01,
    orientation='h'
))

plot(fig)

# + code_folding=[0]
# # Decade most common structure (View 1)

# df = structure_df.copy()

# disc = 5

# df.loc[:, 'pos_disc'] = disc*np.floor(df['line_pos_rel']/disc)
# df.loc[:, 'decade'] = 10*np.floor(df['year']/10)

# songs_per_decade = df.groupby(['decade'])['song_id'].nunique().to_dict()

# df = df\
#         .groupby(['tag_type', 'pos_disc', 'decade'], as_index=False)\
#         .agg({'song_id':'count', 'lines_on_element':'sum'})

# df.loc[:, 'rank_ocurr'] = df.groupby(['decade', 'pos_disc'])['song_id'].rank(ascending=False)

# df = df.loc[df['rank_ocurr'] <= 1]

# regroup_tags = []
# for d in df['decade'].unique().tolist():
#     decade_df = df.loc[df['decade'] == d].sort_values(by='pos_disc')
#     list_order = []
#     for i in range(0, decade_df.shape[0]):
#         pos = decade_df.iloc[i]
#         if not list_order:
#             aux_dict={'tag':pos['tag_type'], 'ocurr':pos['song_id'], 'lines':pos['lines_on_element']}
#             list_order.append(aux_dict)
#         else:
#             last_element = list_order.pop()
#             if pos['tag_type'] == last_element['tag']:
#                 last_element['ocurr'] += pos['song_id']
#                 last_element['lines'] += pos['lines_on_element']
#                 list_order.append(last_element)
#             else:
#                 aux_dict={'tag':pos['tag_type'], 'ocurr':pos['song_id'], 'lines':pos['lines_on_element']}
#                 list_order.append(last_element)
#                 list_order.append(aux_dict)
                
#     final_df = pd.DataFrame(list_order)
#     final_df.loc[:, 'decade'] = d
#     regroup_tags.append(final_df)
    
# df = pd.concat(regroup_tags).reset_index().drop(columns=['index'])
# df.loc[:, 'total_songs_decade'] = df['decade'].apply(lambda x: songs_per_decade[x])

# df.loc[:, 'percentage'] = 100*df['ocurr']/df['total_songs_decade']
                
# df = df.loc[df['percentage'] >= 1]

# df.loc[:, 'avg_lines'] = df['lines']/df['ocurr']
# df.loc[:, 'total_lines'] = df.groupby(['decade'])['avg_lines'].transform('sum')
# df.loc[:, 'pos_line'] = 100*df['avg_lines']/df['total_lines']

# df.loc[:, 'pos_end'] = df.groupby(['decade'])['pos_line'].cumsum()
# df.loc[:, 'pos_start'] = df.groupby(['decade'])['pos_end'].shift(1).fillna(0)
# df.loc[:, 'text'] = df['tag'].str.title()

# df = df.sort_values(by=['decade', 'pos_start'])

# df.loc[:, 'delta'] = df['pos_end'] - df['pos_start']

# # Plot gant
# fig = px.timeline(df, x_start="pos_start", x_end="pos_end", y="decade", 
#                   color="tag", text='text', color_discrete_map=color_map_dict)

# fig.update_xaxes(type='linear')
# for i in range(0, len(fig.data)):
#     tag = fig.data[i].name
#     array = df.loc[df['tag'] == tag].sort_values(by=['decade', 'pos_start'])
#     fig.data[i]['x'] = array['delta'].tolist()

# fig.update_traces(textposition='inside', insidetextanchor='middle', width=8)
# fig.update_layout(showlegend=False)
# fig.update_yaxes(range=[1955, 2025])
# plot(fig)

# + code_folding=[0]
# Most common changes from most used structure, expressed as all changes in each song

# diff_df = seq_df.copy()

# diff_df.loc[:, 'only_diffs'] = diff_df['diff_to_top'].apply(lambda x: sorted([f for f in x if bool(re.search('^[+-] ', f))]))

# diff_df = pd.DataFrame.explode(diff_df, column='only_diffs')

# song_diff_df = diff_df\
#                 .groupby(['song_id', 'decade', 'only_diffs'], as_index=False)\
#                 .agg({'structure':'count'})

# song_diff_df.loc[: , 'diff'] = song_diff_df.apply(lambda x: f"{x['only_diffs'][1:].title()} ({x['only_diffs'][0]}{x['structure']})", axis=1)

# song_diff_df = song_diff_df.groupby(['decade', 'song_id'])['diff'].apply(lambda x: sorted(list(x))).to_frame().reset_index()
# song_diff_df.loc[:, 'diff_summ'] = song_diff_df['diff'].apply(lambda x: ','.join(x))
                                             
# dec_df = song_diff_df.groupby(['decade', 'diff_summ'], as_index=False).agg({'song_id':'nunique'})
# dec_df.loc[:, 'rank'] = dec_df.groupby(['decade'])['song_id'].rank(ascending=False)
# top_df = dec_df.loc[dec_df['rank'] <= 3]
# top_df  
# -


# ### Structure cohesion

# + code_folding=[0]
# Data Prep
df = structure_df.copy()

df = df\
        .groupby(['song_id', 'year', 'tag_type', 'name', 'artist_chart'], as_index=False)\
        .agg({'lines_on_element':'sum', 'order':'count'})\
        .sort_values(by='song_id')

df.loc[:, 'total_lines'] = df.groupby(['song_id'])['lines_on_element'].transform('sum')
df.loc[:, 'percentage'] = df['lines_on_element']/df['total_lines']

song_repr_df = pd.pivot_table(df, index=['song_id', 'year', 'name', 'artist_chart'], columns='tag_type', values='percentage')\
            .fillna(0).reset_index().sort_index()

# Song structure
st_df = structure_df\
        .sort_values(by=['song_id', 'rank'])\
        .groupby(['song_id'])\
        ['tag_type'].apply(lambda x: '<br>'.join(x.str.title()))\
        .to_frame()
st_df.columns = ['song_structure']

song_repr_df = pd.merge(left=song_repr_df, right=st_df, on='song_id')


# Song distance from average representation
dim_cols = structure_df['tag_type'].unique().tolist()

mean_yearly_repr = song_repr_df.groupby(['year']).agg({c:'mean' for c in dim_cols})
all_songs_repr = song_repr_df.loc[:, dim_cols].values

songs_dist_df = pd.DataFrame(cdist(all_songs_repr, mean_yearly_repr.values, metric='euclidean'))\
                        .unstack().to_frame().reset_index()

songs_dist_df.columns = ['year_index', 'song_index', 'distance']
songs_dist_df.loc[:, 'year'] = songs_dist_df['year_index'].apply(lambda x: mean_yearly_repr.index.tolist()[x])
songs_dist_df.loc[:, 'song_id'] = songs_dist_df['song_index'].apply(lambda x: song_repr_df['song_id'].tolist()[x])
songs_dist_df.head()

# + code_folding=[0]
# View song structure similarity 

# Data Prep
reducer = umap.UMAP(n_neighbors=10)
umap_df = pd.DataFrame(reducer.fit_transform(song_repr_df.loc[:, dim_cols].values))
umap_df.columns = ['dim0', 'dim1']

umap_df.index = song_repr_df.index
umap_df = pd.concat([umap_df, song_repr_df.loc[:, ['song_id', 'name', 'year', 'artist_chart', 'song_structure']]], axis=1)

umap_df.loc[:, 'Decade'] = (10*np.floor(umap_df['year']/10)).astype(int).astype(str)

fig = px.scatter(umap_df, x='dim0', y='dim1', color='Decade', hover_name='name', 
                 hover_data=['artist_chart', 'year', 'song_structure'],
                 color_discrete_sequence=px.colors.sequential.Agsunset)
fig.update_layout(
    title='Songs, grouped by structure similarity'
)
fig.update_yaxes(title='', showticklabels=False, nticks=5)
fig.update_xaxes(title='', showticklabels=False)
fig.update_traces(marker_opacity=0.8)

plot(fig)

# + code_folding=[0]
# Song structure cohesion per year
df = songs_dist_df\
        .groupby(['year'], as_index=False)\
        .agg({'distance':'mean'})

fig = px.line(df, x='year', y='distance')
fig.update_layout(title='Structure cohesion, per year')
fig.update_yaxes(rangemode='tozero')
plot(fig)


# df = songs_dist_df.copy()

# smooth = 5
# df.loc[:, 'year_disc'] = np.floor(df['year'].astype(int)/smooth)*smooth

# df = df\
#         .groupby(['year_disc'], as_index=False)\
#         .agg({'distance':'mean'})

# fig = px.line(df, x='year_disc', y='distance')
# fig.update_layout(title='Structure conformity, per year')
# plot(fig)

# + code_folding=[0]
# Song structure diversity evolution 
df = seq_df.copy()

df = df\
        .groupby(['year'], as_index=False)\
        .agg({'structure':'nunique', 'distance_to_top':'mean', 'song_id':'nunique'})

df.loc[:, 'structures_per_song'] = df['structure']/df['song_id']

fig = px.line(df, x='year', y=['structures_per_song', 'distance_to_top'], 
              facet_col='variable', facet_col_spacing=0.1)

title_dict = {'structures_per_song':'Unique song structures, per songs on year', 
              'distance_to_top':'Average distance from most used structure, per year'}
fig.update_yaxes(matches=None, showticklabels=True, rangemode='tozero', title='')
fig.update_xaxes(title='Year')
fig.for_each_annotation(lambda x: x.update(text = title_dict[x.text.split('=')[1]]))

fig.update_layout(
    title='Evolution of song structure diversity',
    showlegend=False,
    margin_t=120
)

plot(fig)
# -

# ## Sructure X Success 

# + code_folding=[0]
##### Is there a difference from chart toppers and the rest of the songs regarding song structure?
df = structure_df.copy()

cut = 20
df.loc[:, 'chart_topper'] = df['chart_position'] <= cut
df.loc[:, 'type_count'] = df.groupby(['chart_topper'])['song_id'].transform('nunique')


df = df\
        .groupby(['song_id', 'year', 'decade', 'tag_type', 'lines_on_lyrics', 'chart_topper', 'type_count'], as_index=False)\
        .agg({'name':'count', 'lines_on_element':'sum'})\
        .rename(columns={'name':'counter'})

df.loc[:, 'importance_on_song'] = 100*df['lines_on_element']/df['lines_on_lyrics']

df = df\
        .groupby(['tag_type', 'chart_topper', 'type_count'], as_index=False)\
        .agg({'song_id':'nunique', 'counter':'mean', 'importance_on_song':'mean'})

df.loc[:, 'percentage'] = 100*df['song_id']/df['type_count']
df.loc[:, 'y_text'] = df['tag_type'].str.title()

color_col = f"Top{cut}"
df.loc[:, color_col] = df['chart_topper']

# 
palette = px.colors.qualitative.Vivid
color_map = {True:palette[5], False:palette[-1]}

fig = px.bar(df, y='y_text', x=['percentage', 'counter', 'importance_on_song'], barmode='group',
             color=color_col, facet_col='variable', color_discrete_map=color_map)

facet_title = {
    'percentage':'Songs that use element',
    'counter':'Amount of elements used',
    'importance_on_song':'Share of lines on element'
}
fig.for_each_annotation(lambda a: a.update(text=facet_title[a.text.split('=')[1]]))

fig.update_xaxes(matches=None, title='')
fig.update_xaxes(range=[0, 120], col=1, dtick=100, ticksuffix='%')
fig.update_xaxes(range=[0, 4], col=2, dtick=3)
fig.update_xaxes(range=[0, 45], col=3, dtick=40, ticksuffix='%')

fig.update_traces(textposition='outside', texttemplate='%{x:.1f}%')
fig.update_traces(texttemplate='%{x:.1f}', col=2)
fig.update_layout(
    height=600,
    title='Is there a structural difference between the Top 20 and the rest of the songs?',
    margin_t=120
)
                  
fig.update_yaxes(categoryorder='total ascending', col=1, title='')

plot(fig)


# -

# ## Song structure visualization

# + code_folding=[0, 18]
def split_lyrics_lines(song_id):
    df = structure_df.loc[structure_df['song_id'] == song_id].copy()
    df.loc[:, 'lyrics_lines'] = df['section_lyrics'].apply(lambda x: [f for f in x.split('\n')])
    line_df = pd.DataFrame.explode(df, column='lyrics_lines')
    line_df.loc[:, 'last_line'] = line_df['lyrics_lines'].shift(1)
    
    line_df.loc[:, 'pos'] = 1
    line_df.loc[:, 'pos'] = line_df['pos'].cumsum()

    marker_value = {'intro':1, 'verse':2, 'pre-chorus':3, 'chorus':4, 
          'post-chorus':3, 'hook':4, 'refrain':4, 'bridge':3, 'solo':1, 'break':1, 
          'outro':1, 'interlude':1, 'outro':1}
    line_df.loc[:, 'value'] = line_df['tag_type'].apply(lambda x: marker_value[x])
    line_df.loc[:, 'size'] = 1
    
    return line_df
    

def print_lyrics_scatter(song_id):

    line_df = split_lyrics_lines(song_id)
    
    breaker = int(np.ceil(line_df['pos'].max()/2)) + 1
    line_df.loc[:, 'break'] = (np.floor(line_df['pos']/breaker)*breaker).astype(int)
    
    color_col = 'Song Element'
    line_df.loc[:, color_col] = line_df['tag_type'].str.title()
    

    fig = px.scatter(line_df, y='pos', x='value', text='lyrics_lines', 
                     size='size', size_max=8,
                     facet_col='break', facet_col_spacing=0.01, color=color_col,
                     color_discrete_map=color_map_dict
                    )
    fig.update_traces(textposition='middle right', textfont_size=12)
    break_list = sorted(line_df['break'].unique().tolist())
    break_list_text = {} 
    for b in break_list:
        if break_list.index(b) == 0:
            break_list_text[str(b)] = 'Start'
        elif break_list.index(b) < len(break_list):
            break_list_text[str(b)] = '...'
        else:
            break_list_text[str(b)] = 'End'

    fig.update_yaxes(matches=None, showticklabels=False, zeroline=False, showgrid=False, title='')
    fig.update_xaxes(range=[0, 50], showticklabels=False, showgrid=False, 
                     title='')
    for i in range(0, len(break_list)):
        df = line_df.loc[line_df['break'] == break_list[i]]
        range_y = [df['pos'].min() + breaker + 1, df['pos'].min()-1]
        fig.update_yaxes(col=i+1, range=range_y)
        if len(break_list) == 0:
            fig.update_xaxes(col=i, title='End')
        else:
            if i < len(break_list) - 1:     
                fig.update_xaxes(col=i+1, title='...')
            else:
                fig.update_xaxes(col=i+1, title='End')
                
    stretcher = (breaker//5)
    fig.update_layout(
        height=max([100*stretcher, 600]),
        margin_l=20,
        margin_r=20,
        margin_t=120,
        legend_orientation='h',
        title=f"<b>{line_df['name'].unique().tolist()[0]}</b><br><i>by {line_df['artist_chart'].unique().tolist()[0]}</i>",
        titlefont_size=18
    )
    fig.update_xaxes(titlefont_size=16)
    
    fig.for_each_annotation(lambda x: x.update(text=break_list_text[x.text.split('=')[1]], font_size=16))
    
    plot(fig)
    
print_lyrics_scatter('2020_3')
# -

# ## Next Ideas
#
# * Artist highlights
# * Songs highlights
