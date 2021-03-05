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

# # Setup

# +
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



pd.set_option('max_columns', None)
# -

data_path = Path('./data/')

# # Data Prep

raw_structure_df = pd.read_csv(data_path/'lyrics/prep/songs_paragraphs_raw.csv')
raw_structure_df.head(1)

# +
# Clean structures that have too few

# +
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
structure_df.loc[:, 'chart_position'] = structure_df['song_id'].apply(lambda x: x.split('_')[1]).astype(int)

structure_df.loc[:, 'next_structure'] = structure_df.sort_values(by=['song_id', 'rank']).groupby(['song_id'])['tag_type'].shift(-1).fillna('end')
structure_df.loc[:, 'previous_structure'] = structure_df.sort_values(by=['song_id', 'rank']).groupby(['song_id'])['tag_type'].shift(1).fillna('start')

total_songs_per_year = structure_df.groupby(['year'])['song_id'].nunique().to_dict()

structure_df.head(1)

# +
# structure_df.loc[structure_df['tag_type'] == 'other']

# +
# structure_df.loc[ structure_df['max_rank']/structure_df['total_paragraphs'] < 0.3]

# +
# print(structure_df.loc[structure_df['song_id'] == '2010_77'].iloc[0]['lyrics'])

# +
# Song structure representation
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

song_repr_df.head()

# +
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


# +
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


# -

def print_lyrics(song_id):
    print(structure_df.loc[structure_df['song_id'] == song_id].iloc[0]['lyrics'])


# +
palette = px.colors.qualitative.Pastel
structure_elements = structure_df['tag_type'].value_counts().index.tolist()
# color_map_dict

main_elements = ['verse', 'chorus', 'bridge', 'outro', 'intro', 'pre-chorus']

color_map_dict = {main_elements[i]:palette[i]  for i in range(0, len(main_elements))}
# -

# ## Graphs

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
fig.update_xaxes(row=1, col=1, title='Percentage of songs<br>that use element', range=[0,110])
fig.update_xaxes(row=1, col=2, title='Elements per song')

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
# How does the use of each element evolve over time? 

df = structure_df\
    .groupby(['song_id', 'tag_type', 'year'], as_index=False)\
    .agg({'order':'count', 'lines_on_element':'sum', 'total_lines':'mean'})

df.loc[:, 'song_share'] = 100*df['lines_on_element']/df['total_lines']

df = df\
        .groupby(['tag_type', 'year'], as_index=False)\
        .agg({'song_id':'nunique', 'order':'mean', 'song_share':'median'})\
        .sort_values(by=['year'], ascending=False)

df.columns = ['song_element', 'year', 'songs', 'average_song_appearance', 'share_in_song']

df.loc[:, 'total_songs'] = df['year'].apply(lambda x: total_songs_per_year[x])
df.loc[:, 'Element'] = df['song_element']
df.loc[:, 'presence_in_songs'] = 100*df['songs']/df['total_songs']

df = df.loc[df['song_element'].isin(main_elements)]

fig = px.line(df, x='year', y='presence_in_songs', facet_col='Element', 
              color='Element', facet_col_spacing=0.1, facet_col_wrap=3, 
              color_discrete_map=color_map_dict
             )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

fig.update_yaxes(range=[-10,100])
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
    title = 'How much song elements are used in songs?',
    font_size=13,
    margin_t=100
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
# How does a song start? View per year
aux = structure_df.loc[structure_df['rank'] <= 1].copy()

aux.loc[:, 'category'] = 'other'
aux.loc[aux['tag_type'].isin(['intro', 'chorus', 'verse']), 'category'] = aux['tag_type']

df = aux\
        .groupby(['category','year'], as_index=False)\
        .agg({'song_id':'count'})

df.loc[:, 'total_songs_on_year'] = df.groupby(['year'])['song_id'].transform('sum')

df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs_on_year']

df.loc[:, 'rolling_percentage'] = df.loc[:, 'percentage'].rolling(window=2).mean()

fig = px.line(df, x='year', y='rolling_percentage', color='category', color_discrete_map=color_map_dict)
fig.update_layout(
    title='How do songs start?<br>A yearly evolution on the type of first song element',
    yaxis_title='Songs starting with element',
    yaxis_ticksuffix='%',
    xaxis_title='Year'
)
plot(fig)

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
    0:['intro', 'instrumental', 'verse'],
    1:['pre-chorus', 'chorus', 'post-chorus'],
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
# + code_folding=[0]
# Song elements count, year evolution
df = structure_df\
        .groupby(['song_id', 'year'], as_index=False)\
        .agg({'words_on_element':'sum', 'lines_on_element':'sum', 'total_lines':'mean', 'total_paragraphs':'mean'})\
        .groupby(['year'], as_index=False)\
        .agg({'words_on_element':'median', 'total_lines':'median', 'total_paragraphs':'mean'})

fig = px.line(df, x='year', y=['words_on_element', 'total_lines', 'total_paragraphs'], 
              facet_col='variable', facet_col_spacing=0.1)
fig.update_yaxes(matches=None, showticklabels=True)
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
# Number of structures, per song

df = structure_df.drop_duplicates(subset=['song_id'])

fig = px.histogram(df, x='max_rank', histnorm='percent')
fig.update_layout(
    title='Distribution of number of elements, per song',
    yaxis_title='Percentage of songs (%)',
    xaxis_title='Number of elements'
)
plot(fig)


# + code_folding=[0]
# Position investigation on song, per rank order
df = structure_df\
        .groupby(['rank', 'tag_type'], as_index=False)\
        .agg({'song_id':'nunique'})

df.loc[:, 'total_songs_rank'] = df.groupby(['rank'])['song_id'].transform('sum')

df.loc[:, 'max_songs'] = df['total_songs_rank'].max()

df.loc[:, 'percentage_of_songs'] = 100*df['total_songs_rank']/df['max_songs']

df = df.loc[df['percentage_of_songs'] >= 10]

df.loc[:, 'percentage_position'] = df['song_id']/df['total_songs_rank']

fig = px.line(df, x='rank', y='percentage_position', color='tag_type')
plot(fig)

# + code_folding=[0]
# What comes after/before each element?

cols = ['tag_type', 'structure', 'counter', 'type']
next_df = structure_df\
            .groupby(['tag_type', 'next_structure'], as_index=False)\
            .agg({'song_id':'count'})

next_df.loc[:, 'type'] = 'After'
next_df.columns = cols

prev_df = structure_df\
            .groupby(['tag_type', 'previous_structure'], as_index=False)\
            .agg({'song_id':'count'})

prev_df.loc[:, 'type'] = 'Before'
prev_df.columns = cols

df = pd.concat([next_df, prev_df], axis=0)

df.loc[:, 'total_count'] = df.groupby(['tag_type', 'type'])['counter'].transform('sum')
df.loc[:, 'percentage'] = 100*df['counter']/df['total_count']

df.loc[:, 'rank'] = df.groupby(['tag_type', 'type'])['percentage'].rank(ascending=False)

df = df.loc[df['rank'] <= 1].sort_values(by='rank')

df

fig = px.bar(df, y='tag_type', color='structure', x='percentage', 
             facet_col='type', color_discrete_map=color_map_dict)

# fig.update_yaxes(col=1, categoryorder='category ascending')
fig.update_xaxes(title='Ocurrence', ticksuffix='%', range=[0,110], dtick=50)
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.for_each_annotation(lambda x: x.update(text=x.text.split('=')[1]))
fig.update_layout(
    title='What comes before and after each song element?',
    yaxis_title=''
)

# fig.update_xaxes(autorange='reversed', row=1, col=1)
# fig.update_xaxes(autorange=True, row=1, col=2)
# fig.update_yaxes(showticklabels=False)

plot(fig)


# + code_folding=[]
# Most common sequence of elements, to different sizes
# Need more exploration

def expand_structture_list(struct_list):
    aux_dict = {}
    for i in range(2, len(struct_list) + 1):
        aux_dict[i] = []
        start = 0
        end = i
        while end <= len(struct_list):
            s = '_'.join(struct_list[start:end])
            start += 1
            end += 1
            aux_dict[i].append(s)
            
    return aux_dict

df = structure_df.groupby(['song_id'])['tag_type'].apply(list).to_frame()
df.loc[:, 'expand'] = aux['tag_type'].apply(lambda x: expand_structture_list(x))

aux_df = df['expand'].apply(pd.Series).unstack().to_frame().reset_index()
aux_df.columns = ['sequence_size', 'song_id', 'structure']

aux_df = pd.DataFrame.explode(aux_df, column='structure').dropna()

df = aux_df\
        .groupby(['sequence_size', 'structure'], as_index=True)\
        .agg({'song_id':['count', 'nunique']})

df.columns = ['occurrences', 'songs_with_structure']
df.reset_index(inplace=True)

df.loc[:, 'total_count'] = df.groupby(['sequence_size'])['occurrences'].transform('sum')

df.loc[:, 'percentage'] = df['occurrences']/df['total_count']
df.loc[:, 'rank'] = df.groupby(['sequence_size'])['percentage'].rank(ascending=False)

top = df.loc[(df['rank'] == 1) & (df['sequence_size'] <= 10)].copy()

top.loc[:, 'element_list'] = top['structure'].str.split('_')

order_df = pd.DataFrame.explode(top, column='element_list').reset_index()
order_df.loc[:, 'order'] = order_df.groupby(['index'])['sequence_size'].rank(method='first').astype(int)

df = order_df.loc[order_df['sequence_size'] <= 10]

df.loc[:, 'size'] = 1
df.loc[:, 'text'] = df['element_list'].str.title()

fig = px.scatter(df, x='order', y='sequence_size', color='element_list', 
                 size='size', size_max=10, text='text')
fig.update_traces(textposition='top center')
fig.update_xaxes(range=[0.5,11], showgrid=False, showticklabels=False, title='')
fig.update_layout(
    showlegend=False,
    title='Most common structure found, by number of elements in sequence'
)
fig.update_yaxes(title='Sequence size')
plot(fig)

test_df = df.copy()

test_df.loc[:, 'start'] = test_df['order'].astype(int)
test_df.loc[:, 'end'] = test_df.groupby(['sequence_size'])['order'].shift(-1)
test_df.loc[test_df['end'].isnull(), 'end'] = x['order'] + 1
test_df.loc[:, 'end'] = test_df['end'].astype(int)
test_df.loc[:, 'delta'] = test_df['start'] - x['end']

fig = px.timeline(test_df, x_start="start", x_end="end", y="sequence_size", 
                  color="element_list", text='text', color_discrete_map=color_map_dict)
fig.update_xaxes(type='linear')
for i in range(0, len(fig.data)):
    fig.data[i]['x'] = test_df['delta'].tolist()
fig.update_traces(textposition='inside', insidetextanchor='middle')
plot(fig)

# + code_folding=[0]
# View song structure similarity 
reducer = umap.UMAP(n_neighbors=20)
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
# -

# ## Test

# Ideas
#
# * Intro start breakdown: repeats or not?
#
# * Most common structures on songs, decade evolution (Gantt)
#
# * Examples of strucutre, and exceptions
#
# * Lolipop graph to before and after
#
# * Analyze position on chart
#
# * Sentiment Analysis on Structure
#
#
# Other projects
# * Temporal vocabulary
# * NLP analysis
# * Per genre of song

# +
# Explor start repetition of intro
# df = structure_df.sort_values(by=['song_id', 'rank'])

# df.loc[:, 'lines_cumul'] = df.groupby(['song_id'])['lines_on_element'].cumsum()

# df = df.loc[df['lines_cumul'] <= 5]

# # rep_df = df['line_repetition_dict'].apply(pd.Series)

df = structure_df\
        .sort_values(by=['song_id', 'rank'])\
        .drop_duplicates(subset=['song_id']).set_index('song_id')

aux_df = df.loc[:, ['tag_type', 'total_lines', 'year', 'chart_position', 'lines_on_element', 'next_structure']]


rep_df = df['line_repetition_dict'].apply(pd.Series).unstack().to_frame().reset_index()
rep_df.columns = ['line', 'song_id', 'is_repeated']

# rep_melt = pd.melt(rep_df.reset_index(), id_vars=['song_id'], value_vars=line_cols)

# rep_melt.loc[:, 'variable'] = rep_melt['variable'].astype(int)
rep_df.loc[:, 'is_repeated'] = rep_df['is_repeated'].fillna(False).astype(int)

rep_df.loc[:, 'start_repeats'] = rep_df.groupby(['song_id'])['value'].transform('max').astype(bool)

# rep_df = rep_melt.loc[rep_melt['value'] == 1]\
#             .sort_values(by=['song_id', 'variable'])\
#             .drop_duplicates(subset=['song_id'])\
#             .rename(columns={'variable':'first_rep_line'})

# repetition_df = pd.merge(left=aux_df, right=rep_df, on='song_id', how='left').drop(columns=['value'])

# repetition_df.loc[:, 'structure_foreshadows'] = repetition_df['start_repeats'].fillna(False)
# # repetition_df.loc[repetition_df['first_rep_line'] > repetition_df['lines_on_element'], 'structure_foreshadows'] = False
# -

repetition_df.loc[repetition_df['structure_foreshadows']].sort_values(by=['year'], ascending=False)

# +
# Lyrical foreshadowing
df = repetition_df\
        .groupby(['year', 'structure_foreshadows'], as_index=False)\
        .agg({'song_id':'nunique'})

df.loc[:, 'total_songs'] = df.groupby(['year'])['song_id'].transform('sum')
df.loc[:, 'rate'] = 100*df['song_id']/df['total_songs']

df = df.loc[df['structure_foreshadows']]

fig = px.line(df, x='year', y='rate')
plot(fig)


# Lyrical foreshadowing, per structure
df = repetition_df.copy()

df.loc[:, 'structure'] = df['tag_type']
# df.loc[(df['tag_type'] == 'intro') & (df['structure_foreshadows']) , 'structure'] = 'foreshadow intro'
# df.loc[(df['tag_type'] == 'intro') & (~df['structure_foreshadows']) , 'structure'] = 'pure intro'

# df.loc[(df['tag_type'] == 'intro') , 'structure'] = df['structure'] + ' - ' + df['next_structure']

df = df\
        .groupby(['year', 'structure', 'tag_type'], as_index=False)\
        .agg({'song_id':'nunique'})

df.loc[:, 'total_songs'] = df.groupby(['year'])['song_id'].transform('sum')
df.loc[:, 'rate'] = 100*df['song_id']/df['total_songs']

structs_to_analyze = ['chorus', 'intro', 'verse', 'hook', 'pre_chorus']

df = df.loc[df['tag_type'].isin(structs_to_analyze)]

df.loc[:, 'category'] = 'Repetition'
df.loc[df['structure'].isin(['pure intro', 'verse']), 'category'] = 'No Repetition'

fig = px.line(df, x='year', y='rate', color='structure')
plot(fig)

# +
# # df = structure_df.groupby(['song_id'])['tag_type'].apply(lambda x: '_'.join(x)).to_frame()
# df['tag_type'].value_counts().tail(20)

# + code_folding=[0]
# Specific elements change in words and importance to songs, over time

df = structure_df\
        .groupby(['song_id', 'year', 'tag_type'], as_index=False)\
        .agg({'words_on_element':'sum', 'lines_on_element':'sum', 'total_lines':'mean'})

df.loc[:, 'total_words'] = df.groupby(['song_id'])['words_on_element'].transform('sum').fillna(0)
df.loc[:, 'relevance_on_song'] = 100*(df['words_on_element']/df['total_words'])

df = df\
        .groupby(['tag_type', 'year'], as_index=False)\
        .agg({'words_on_element':'median', 'relevance_on_song':'median'})

fig = px.line(df.sort_values(by='year'), x='year', y=['words_on_element'], 
              facet_col='tag_type', color='tag_type', facet_col_wrap=4, facet_col_spacing=0.05)

fig.update_yaxes(matches=None, showticklabels=True, rangemode='tozero')
fig.update_layout(showlegend=False, height=800)

plot(fig)


fig = px.line(df.sort_values(by='year'), x='year', y=['relevance_on_song'], 
              facet_col='tag_type', color='tag_type', facet_col_wrap=4, facet_col_spacing=0.05)

fig.update_yaxes(showticklabels=False, rangemode='tozero', title='')
fig.update_layout(showlegend=False, height=800)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))
plot(fig)

# -

