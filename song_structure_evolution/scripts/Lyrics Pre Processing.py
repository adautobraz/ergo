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

charts_df = pd.read_csv(data_path/'billboard_charts_1946_2020.csv')
charts_df.head()

# # Raw lyrics

# +
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

lyrics_df = pd.merge(left=charts_df.loc[:, ['song_id', 'name', 'artist']], right=lyrics_df, on='song_id', 
                     how='inner', suffixes=('_chart', ''))

# +
lyrics_df.loc[:, 'structure_tags'] = lyrics_df['lyrics'].apply(lambda x: re.findall(r"(\[[^\[]*\])", x))

lyrics_df.loc[:, 'has_tags'] = lyrics_df['structure_tags'].apply(lambda a: True if a else False)

lyrics_df.loc[:, 'artist'] = lyrics_df['primary_artist'].apply(lambda a: a['name'])

lyrics_df.loc[:, 'artist_name_diff'] = lyrics_df.apply(lambda x: fuzz.partial_ratio(x['artist'].lower(), 
                                                                                    x['artist_chart'].lower()), axis=1)

lyrics_df.loc[lyrics_df['artist_name_diff'] < 50, 'artist'] = lyrics_df['artist_chart']

lyrics_df.head()


# +
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
#             aux_dict['is_structure_tag'] = True
            
        else:
            if 'prod' in t.lower(): 
                aux_dict['clean_tag'] = 'production_credits'
            elif 'video' in t.lower():   
                aux_dict['clean_tag'] = 'video_credits'

#             aux_dict['is_structure_tag'] = False
            
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
            t['is_structure_tag'] = True
        else:
            match='other'
            is_structure_tag=False
            for v in valid_tags:
                if fuzz.partial_ratio(v, tag) >= 80:
                    match = v
                    is_structure_tag = True
                    break
            t['tag_type'] = match
            t['is_structure_tag'] = is_structure_tag
            
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
            print(valid_tags)

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
        
        
def check_lyrics_repetition_later(raw_lyrics, all_tags, cut):
    
    lyrics = raw_lyrics
    for t in all_tags:
        lyrics = lyrics.replace(t, '')

    all_lines = [f for f in lyrics.split('\n') if f]
    line_dict = {(a+1): all_lines[a] for a in range(0, len(all_lines))}
    key_list =  list(line_dict.keys())

    first_lines = key_list[:cut]
    other_lines = key_list[cut:]

    match_dict = {}

    for f in first_lines:
        line = line_dict[f]
        match_dict[f] = False
        for o in other_lines:
            line_compare = line_dict[o]
            if fuzz.ratio(line, line_compare) > 90:
                match_dict[f] = True
                break

    return match_dict


# -

lyrics_df.loc[:, 'all_tags_map'] = lyrics_df['structure_tags'].apply(lambda x: define_tag_structure(x))
lyrics_df.loc[:, 'structure_tags_map'] = lyrics_df.apply(lambda x: call_func(x['all_tags_map'], x['lyrics'], x['song_id']), axis=1)

lyrics_df.loc[:, 'line_repetition_dict'] = lyrics_df.apply(lambda x: check_lyrics_repetition_later(x['lyrics'], x['structure_tags'], 5), axis=1)

lyrics_df.to_csv(data_path/'lyrics/prep/song_lyrics_clean.csv', index=False)

# +
# x = lyrics_df
# maper = lyrics_df.iloc[x]['all_tags_map']
# lyrics = lyrics_df.iloc[x]['lyrics']
# extract_text_on_structure(maper, lyrics, x, True)

# +
lyrics_with_tags = lyrics_df.loc[lyrics_df['has_tags']]

song_paragraphs_exploded = pd.DataFrame.explode(lyrics_with_tags, column ='structure_tags_map').reset_index().iloc[:, 1:]
tags_exploded = song_paragraphs_exploded['structure_tags_map'].apply(pd.Series).iloc[:, 1:]

raw_structure_df = pd.concat([song_paragraphs_exploded, tags_exploded], axis=1)
# -

raw_structure_df.to_csv(data_path/'lyrics/prep/songs_paragraphs_raw.csv', index=False)
