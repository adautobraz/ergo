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

pd.set_option('max_columns', None)

from ergo_utilities import lyrics_info
# -

data_path = Path('./data/')
genius_path = Path('./data/lyrics/raw/genius/per_year')

# + [markdown] heading_collapsed=true
# # Raw lyrics

# + hidden=true
dicts = []

year_folders = os.listdir(genius_path)

for y in year_folders:
    all_files = os.listdir(genius_path/y)
    for f in all_files:
        if '.json' in f:
            with open(genius_path/y/f, 'r', encoding='windows-1252') as r:
                lyric = json.load(r)
                
            lyric['song_id'] = f.split('.json')[0]
            dicts.append(lyric)

lyrics_raw_df = pd.DataFrame(dicts)

lyrics_df = lyrics_raw_df.loc[:, 'title':'song_id']#.set_index('song_id')

# + hidden=true
lyrics_df.loc[:, 'structure_tags'] = lyrics_df['lyrics'].apply(lambda x: re.findall(r"(\[.*)", x))

lyrics_df.loc[:, 'has_tags'] = lyrics_df['structure_tags'].apply(lambda a: True if a else False)

lyrics_df.head()


# + hidden=true
def define_tag_structure(tags):
    
    # Clean structure and find annotations on lyrics marked by []
    
    tag_info = []
    for i in range(0, len(tags)):
#         print(i, t)
        t = tags[i]
        aux_dict = {'raw_tag':t, 'order':i}
        if 'prod' not in t.lower() and 'video' not in t.lower():
            if len(re.findall(r"(\[[\w\d -:]*\])", t)) > 0:
                tag = re.findall(r"(\[[\w\d -:]*\])", t)[0]
            else:
                tag = t
            
            tag = re.sub(r'[\[\]]', '', tag).lower().split(':')[0]
#             print(tag)
            clean_tag = re.sub(r'[0-9]', '', tag).strip()
#             print(clean_tag)
            aux_dict['clean_tag'] = clean_tag
            aux_dict['is_structure_tag'] = True
            
        else:
            if 'prod' in t.lower(): 
                aux_dict['clean_tag'] = 'production_credits'
            elif 'video' in t.lower():   
                aux_dict['clean_tag'] = 'video_credits'

            aux_dict['is_structure_tag'] = False
            
        tag_info.append(aux_dict)
        
            
    
    # Clean tag types
    valid_tags = ['pre-chorus', 'post-chorus', 'chorus', 'verse',  'bridge', 'outro', 'intro', 
               'refrain', 'hook', 'interlude', 'break', 'solo', 'instrumental']
    
#     orderd_tags = sorted(valid_tags, key=len, reverse=True)
    
    all_tags = []
    for t in tag_info:
                
        tag = t['clean_tag'].replace(' ', '_')
        if tag in valid_tags:
            t['tag_type'] = tag
        else:
            match='other'
            for v in valid_tags:
                if fuzz.partial_ratio(v, tag) >= 80:
                    match = v
                    break
            t['tag_type'] = match
            
    return tag_info

all_errors = []

def extract_text_on_structure(tags_list, lyrics, song_id, to_print=False):
    
        valid_tags = []
        new_list = copy.deepcopy(tags_list)

        # Remove invalid structure tags from text
        clean_lyrics = lyrics.replace('\n\n', '\n') 
        
        for t in tags_list:
            if t['is_structure_tag']:
                valid_tags.append(t)
            else:
                clean_lyrics = clean_lyrics.replace(t['raw_tag'], '')
                
        if to_print:
            print(clean_lyrics)

        found_text = []

        # Split tags and attribute specific 
        for i in range(0, len(valid_tags)):

            lyrics_left = clean_lyrics.replace(''.join(found_text), '', 1)
            if to_print:
                print('Encontrado', found_text)
                print('O que resta', [lyrics_left])

            start = valid_tags[i]['raw_tag']
            start_rgx = re.escape(start)
            if i < len(valid_tags) - 1:
                end = valid_tags[i+1]['raw_tag']
                end_rgx=re.escape(end)
                regex = f'(?s)(?<={start_rgx}).*?(?={end_rgx})'
                matches = re.findall(regex, lyrics_left)
                
                if to_print:
                    print(start, end)
                    print(start_rgx, end_rgx)      
                    print(regex)
                    print(matches)
                
                text = matches[0]

            else:
                text = lyrics_left.split(start)[-1]

            new_list[i]['section_lyrics'] = text
            found_text.append(start + text)


        return new_list
    

def call_func(tags_list, lyrics, song_id):
    
    try:
        return extract_text_on_structure(tags_list, lyrics, song_id)
    except:
        all_errors.append(song_id)


# + hidden=true
lyrics_df.loc[:, 'all_tags_map'] = lyrics_df['structure_tags'].apply(lambda x: define_tag_structure(x))
lyrics_df.loc[:, 'structure_tags_map'] = lyrics_df.apply(lambda x: call_func(x['all_tags_map'], x['lyrics'], x['song_id']), axis=1)

# + hidden=true
lyrics_df.to_csv(data_path/'lyrics/prep/song_lyrics_clean.csv', index=False)

# + hidden=true
# x = lyrics_df
# maper = lyrics_df.iloc[x]['all_tags_map']
# lyrics = lyrics_df.iloc[x]['lyrics']
# extract_text_on_structure(maper, lyrics, x, True)

# + hidden=true
lyrics_with_tags = lyrics_df.loc[lyrics_df['has_tags']]

song_paragraphs_exploded = pd.DataFrame.explode(lyrics_with_tags, column ='structure_tags_map').reset_index().iloc[:, 1:]
tags_exploded = song_paragraphs_exploded['structure_tags_map'].apply(pd.Series).iloc[:, 1:]

raw_structure_df = pd.concat([song_paragraphs_exploded, tags_exploded], axis=1)

# + hidden=true
raw_structure_df.to_csv(data_path/'lyrics/prep/songs_paragraphs_raw.csv', index=False)
# -

# # Structured Lyrics

raw_structure_df = pd.read_csv(data_path/'lyrics/prep/songs_paragraphs_raw.csv')
raw_structure_df.head(1)

# +
# Clean structures that have too few

# +
structure_df = raw_structure_df.loc[raw_structure_df['is_structure_tag']].sort_values(by=['song_id', 'order'])

structure_df.loc[:, 'total_paragraphs'] = structure_df['lyrics'].apply(lambda x: len(x.split('\n\n')))
# structure_df.loc[:, 'total_valid_tags'] = structure_df['structure_tags_map'].apply(lambda x: len([x for x if x['is_structure_tag']])))

structure_df.loc[:, 'rank'] = structure_df.groupby(['song_id'])['order'].rank()
structure_df.loc[:, 'max_rank'] = structure_df.groupby(['song_id'])['rank'].transform('max')

structure_df.loc[:, 'paragraph_pos_rel'] = 100*(structure_df['rank'] - 1)/(structure_df['max_rank'] - 1)

structure_df.loc[:, 'year'] = structure_df['song_id'].apply(lambda x: x.split('_')[0]).astype(int)
structure_df.loc[:, 'chart_position'] = structure_df['song_id'].apply(lambda x: x.split('_')[1]).astype(int)

total_songs_per_year = structure_df.groupby(['year'])['song_id'].nunique().to_dict()

structure_df.head(1)
# -

structure_df.loc[ structure_df['max_rank']/structure_df['total_paragraphs'] < 0.5]


# +
# print(structure_df.loc[structure_df['song_id'] == '2010_77'].iloc[0]['lyrics'])
# -

def plot(fig):
    fig.update_layout(
        template = 'plotly_white',
        font_family='Fira Sans'
    )
    
    fig.show()


# + code_folding=[0]
# Total distribution 

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

# + code_folding=[]
# Time evolution 

df = structure_df\
    .groupby(['song_id', 'tag_type', 'year'], as_index=False)\
    .agg({'order':'count'})\
    .groupby(['tag_type', 'year'], as_index=False)\
    .agg({'song_id':'nunique', 'order':'mean'})\
    .sort_values(by=['year'], ascending=False)

df.columns = ['song_element', 'year', 'songs', 'average_song_appearance']

df = df.loc[df['song_element'].isin(['chorus', 'verse', 'bridge', 'outro', 'intro', 'pre-chorus'])]

df.loc[:, 'total_songs'] = df['year'].apply(lambda x: total_songs_per_year[x])

df.loc[:, 'Element'] = df['song_element'].str.title()


df.loc[:, 'presence_in_songs'] = 100*df['songs']/df['total_songs']


fig = px.line(df, x='year', y=['presence_in_songs', 'average_song_appearance'], 
              facet_col='variable', color='Element', facet_col_spacing=0.1)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

fig.update_yaxes(matches=None, showticklabels=True)
fig.update_yaxes(row=1, col=2, range=[-0.1, 4])
fig.update_yaxes(row=1, col=1, ticksuffix='%')

fig.update_xaxes(title='Year')

fig.update_layout(
    showlegend=True, 
    yaxis_title='',
    title = 'Evolution of most common song elements presence and appereance',
    font_size=13,
    margin_t=100
)

plot(fig)

# + code_folding=[0]
# First element on song, per year
aux = structure_df.loc[structure_df['rank'] <= 2].copy()

# aux.loc[:, 'category'] = aux['tag_type']
aux.loc[:, 'category'] = 'other'
aux.loc[aux['tag_type'] == 'verse', 'category'] = 'verse'
aux.loc[aux['tag_type'].isin(['intro', 'chorus']), 'category'] = 'intro or chorus'

# Line graph
df = aux\
        .groupby(['tag_type', 'year', 'rank'], as_index=False)\
        .agg({'song_id':'count'})\
        .sort_values(by=['rank', 'year'])

df.loc[:, 'total_songs_on_year'] = df.groupby(['year'])['song_id'].transform('sum')

df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs_on_year']

fig = px.line(df, x='year', y='percentage', facet_col='rank', color='tag_type', facet_col_spacing=0.1) 
plot(fig)

# Focus on verse x 
df = aux.groupby(['category','year'], as_index=False).agg({'song_id':'count'})

df.loc[:, 'total_songs_on_year'] = df.groupby(['year'])['song_id'].transform('sum')

df.loc[:, 'percentage'] = 100*df['song_id']/df['total_songs_on_year']

df.loc[:, 'rolling_percentage'] = df.loc[:, 'percentage'].rolling(window=3).mean()


fig = px.line(df, x='year', y='percentage', color='category') 
plot(fig)
# px.area(df, x='year', y='percentage', facet_col='category') 

# order_df.loc[:, 'start'] = order_df['first'] + '-' + 

# +
# Check repetition of intro on song follow up
# -

structure_df.head()

import numpy as np

# +
# Element position on song

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

group_df
# df
fig = px.line(group_df, x='pos_disc', y='percentage', color='tag_type', line_shape='spline',
             facet_col='tag_type',  facet_col_wrap=7)
fig.update_yaxes(matches=None, rangemode='tozero', showticklabels=False)
fig.update_xaxes(matches=None, rangemode='tozero', showticklabels=False, title='')
fig.update_xaxes(row=1, col=1, showticklabels=True)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))
fig.update_traces(fill='tozeroy')
fig.update_layout(showlegend=False)

plot(fig)


df.loc[:, 'scale'] = 'others'
df.loc[df['tag_type'].isin(['intro', 'outro']), 'scale'] = 'intro/outro'

# df = df.loc[df['scale'] == 'others']

fig = px.violin(df, x='paragraph_pos_rel', color='tag_type', facet_col='tag_type',
                points=False, violinmode='overlay', facet_col_wrap=7)

fig.update_traces(side='positive', spanmode='hard', width=2)

# fig.update_yaxes(matches=None)#=[-1,1])
fig.update_xaxes(showticklabels=False, title='')#=[-1,1])
fig.update_xaxes(showticklabels=True, row=1, col=1)#=[-1,1])

fig.update_layout(showlegend=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

plot(fig)


top = df.loc[df['tag_type'].isin(['verse', 'chorus', 'pre-chorus', 'post-chorus', 'bridge'])]
fig = px.violin(top, x='paragraph_pos_rel', color='tag_type',
                points=False, violinmode='overlay')

fig.update_traces(side='positive', spanmode='hard', width=2, opacity=0.5)

# fig.update_yaxes(matches=None)#=[-1,1])
fig.update_xaxes(showticklabels=False, title='')#=[-1,1])
fig.update_xaxes(showticklabels=True, row=1, col=1)#=[-1,1])

fig.update_layout(showlegend=True)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].replace('_', ' ').title()))

plot(fig)

# -




