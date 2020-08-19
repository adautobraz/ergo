import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from pathlib import Path
import copy
import random
import re
import json
import os


import lyricsgenius
from flair.models import TextClassifier
from flair.data import Sentence
import spacy

import pandas as pd

def setup():
    with open('/Users/adautobrazdasilvaneto/Documents/ergo/.config') as f:
        credentials=json.load(f)

    return lyricsgenius.Genius(credentials['genius']['client_access_token'])


def clean_song_name(song):
    p = re.compile("([^\(\)]*)\(*.*\)*")
    result = p.search(song)
    song_adj = result.group(1).split(' - ')[0].lower().strip()

    return song_adj


def is_artist_on_track(artist, track):
    
    if track:
     
        artists_on_song = ' '.join([a['name'].lower() for a in track.featured_artists])
        artists_on_song += ' ' + str(track.artist).lower()
        
        if artist.lower() in artists_on_song:
            return True
        else:
            return False
    else:
        return False


def download_artist_lyrics(folder, all_songs, artist):
    
    genius = setup()
#     genius.skip_non_songs = True # Include hits thought to be non-songs (e.g. track lists)
    
    not_found = []
    # Create folder for artist
    Path(folder/"{}/genius/".format(artist)).mkdir(parents=True, exist_ok=True)
    os.chdir(folder/"{}/genius/".format(artist))
    
    # Check if there are already songs downloaded
    songs_already_downloaded = [file.split('.json')[0] for file in os.listdir() if '.json']

    # Select only songs not already on the folder
    df = all_songs.loc[(all_songs['artist'] == artist) & (~all_songs.index.isin(songs_already_downloaded))]
        
    # For all songs on dataframe, call genius lyric search
    lyrics_array = []
    for index in df.index.tolist():
        s = df.loc[index]
        song_adj = s['name']
        song_id = index
        song_lyric = genius.search_song(song_adj, artist)

        if not is_artist_on_track(artist, song_lyric):
            song_adj = clean_song_name(s['name'])
            
            song_lyric = genius.search_song(song_adj, artist)
            
            if not is_artist_on_track(artist, song_lyric):
                not_found.append(song_id)
                print("Não achou")
                
            else:
                song_lyric.to_json(filename='{}.json'.format(song_id))
            
        else:
            song_lyric.to_json(filename='{}.json'.format(song_id))
    
    return not_found 


def correct_lyrics(folder, song_ids, song_names, artist):
      
    genius = setup()
    genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
    
    not_found = []
    # Create folder for artist
    Path(folder/"{}/genius/".format(artist)).mkdir(parents=True, exist_ok=True)
    os.chdir(folder/"{}/genius/".format(artist))
    
    for i in range(0, len(song_names)):
        song_adj = song_names[i]
        song_id = song_ids[i]

        song_lyric = genius.search_song(song_adj, artist)

        if not is_artist_on_track(artist, song_lyric):
            print("Não achou")
        else:
            song_lyric.to_json(filename='{}.json'.format(song_id))



def concat_lyrics_df(folder, artist):
    
    all_songs_lyrics = []

    os.chdir(folder/'{}/genius/'.format(artist))

    for file in os.listdir():
        if 'json' in file:
        
            with open(file, 'r') as data_file:    
                s = json.load(data_file)
            
            song_id = file.split('.json')[0]

            songs_infos = []
                
            if s:

                writers = [w['name'] for w in s['writer_artists']]
                producers = [p['name'] for p in s['producer_artists']]


                info = {'song': s['title'].strip(),
                        'writers':writers,
                        'producers':producers,
                        'lyrics':s['lyrics'],
                        'lyrics_id':s['id'],
                        'spotify_id':song_id,
                        'lyrics_artist':s['primary_artist']['name'],
                        }
                
                songs_infos.append(info)

            df = pd.DataFrame(songs_infos)

            all_songs_lyrics.append(df)

    lyrics_df = pd.concat(all_songs_lyrics)
    
    os.chdir(folder/'{}'.format(artist))
    lyrics_df.to_csv('lyrics_raw.csv', index=False)
    
    
    return lyrics_df



def clean_lyrics(lyrics, list_remove):

    text = lyrics.replace('\u2005', ' ')

    for l in list_remove:
        text = text.replace(l, '')
        
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'^\n', '', text)
  
    matches = list(set(re.findall(r'[?!\.]\n{2,}', text)))
    for m in matches:
        text = text.replace(m, m[0] + ' ')
    
    text = re.sub(r'[\n]{2,}', '. ', text)
    
    matches = list(set(re.findall(r'[?!\.,]\n', text)))
    for m in matches:
        text = text.replace(m, m[0] + ' ')
        
    matches = list(set(re.findall(r'[\w\)]\n[\(\w]', text)))
    for m in matches:
        text = text.replace(m, m[0] + '. ' + m[2])
        
    text = re.sub(r'\n', '. ', text)

    return text


def clean_structure_tags(tags):
    
    all_tags = []
    for t in tags:
        if 'prod' not in t:
            tag = t[1:-1].lower().split(':')[0].strip()
            tag_alphabetic = ''.join([i for i in tag if not i.isdigit()]).strip()
            all_tags.append(tag_alphabetic)

    return all_tags


def get_song_structure(song_tags):
    
    valid_tags = ['chorus', 'verse', 'pre_chorus', 'bridge', 'outro', 'intro', 'post_chorus', 
               'refrain', 'hook', 'interlude', 'break', 'solo']
    
    all_tags = []
    for t in song_tags:
        tag = t.replace(' ', '_')
        if t in valid_tags:
            all_tags.append(tag)
        else:
            sub_tag = False
            for v in valid_tags:
                if v in tag:
                    all_tags.append(v)
                    break
    
    return '-'.join(all_tags)


def get_lyrics_df(folder, artist):
    
    lyrics_df = concat_lyrics_df(folder, artist)
    lyrics_df.loc[:, 'artist'] = artist
    
    # Create columns
    lyrics_df.loc[:, 'artist_wrote_song'] = lyrics_df.apply(lambda x: 1 if x['artist'] in x['writers'] else 0, axis=1)
    lyrics_df.loc[:, 'artist_produced_song'] = lyrics_df.apply(lambda x: 1 if x['artist'] in x['producers'] else 0, axis=1)
    lyrics_df.loc[:, 'structure_tags'] = lyrics_df['lyrics'].apply(lambda x: re.findall(r"(\[.*\])", x))
    lyrics_df.loc[:, 'lyrics_clean'] = lyrics_df.apply(lambda x: clean_lyrics(x['lyrics'], x['structure_tags']), axis=1)
    lyrics_df.loc[:, 'structure_tags_clean'] = lyrics_df['structure_tags'].apply(lambda x: clean_structure_tags(x)) 
    lyrics_df.loc[:, 'song_structure'] = lyrics_df['structure_tags_clean'].apply(lambda x: get_song_structure(x)) 

    
    # Analyze tokens and text 
    nlp = spacy.load('en', disable=['parser', 'ner'])
    lyrics_df.loc[:, 'total_word_count'] = lyrics_df['lyrics_clean'].apply(lambda x: len(nlp(x)))
    lyrics_df.loc[:, 'lyrics_lemmatized'] = lyrics_df['lyrics_clean'].apply(lambda x: ' '.join([token.lemma_.lower() for token in nlp(x) if token.is_alpha and not token.is_stop]))
    lyrics_df['lemma_count'] = lyrics_df['lyrics_lemmatized'].apply(lambda x: len(x.split(' ')))
    lyrics_df['unique_lemmas_on_song'] = lyrics_df['lyrics_lemmatized'].apply(lambda x: len(set(x.split(' '))))                                                                          

    
    # Sentiment analysis 
    classifier = TextClassifier.load('sentiment')

    sents = []

    for i in range(0, len(lyrics_df.index.tolist())):
        text = lyrics_df.iloc[i]['lyrics_clean']
        if text:
            sentence = Sentence(text)
            classifier.predict(sentence)
            sents.append(str(sentence.labels))
        
        else:
            sents.append('')
    
    lyrics_df.loc[:, 'flair_sentiment'] = sents
                  
    lyrics_df.loc[:, 'sentiment_label'] =  lyrics_df['flair_sentiment'].str[1:-1].str.split('(', expand=True).iloc[:, 0].fillna('').str.strip()
    lyrics_df.loc[:, 'sentiment_probability'] =  lyrics_df['flair_sentiment'].str[1:-1].str.split('(', expand=True).iloc[:, 1].str[:-1].fillna(0)

    lyrics_df.loc[:, 'sentiment'] =  lyrics_df['sentiment_label'].apply(lambda x: 1 if x == 'POSITIVE' else -1 if x=='NEGATIVE' else 0)

    lyrics_df.loc[:, 'sentiment_score'] = (lyrics_df['sentiment_probability'].astype(float) - 0.5)
    lyrics_df.loc[:, 'sentiment_score'] = lyrics_df['sentiment_score']*lyrics_df['sentiment']
    
    os.chdir(folder/'{}'.format(artist))
    lyrics_df.to_csv('lyrics_final.csv', index=False,)
        
    return lyrics_df