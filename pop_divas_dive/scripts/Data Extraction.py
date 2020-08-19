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
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
import copy
import random
from difflib import SequenceMatcher
import re
import json
import plotly.express as px
import lyricsgenius
from pathlib import Path
import os
from flair.models import TextClassifier
from flair.data import Sentence

pd.set_option('max_columns', None)
pd.set_option('max_rows', 20)

path = "/Users/adautobrazdasilvaneto/Documents/ergo/pop_divas_super_album/"
data_folder = Path(path)/'data'

lyrics_path = "/Users/adautobrazdasilvaneto/Documents/ergo/pop_divas_super_album/data/lyrics_v2/"
Path(lyrics_path).mkdir(parents=True, exist_ok=True)
lyrics_folder = Path(lyrics_path)

os.chdir(path)

# +
with open('../.config') as f:
    credentials=json.load(f)

client_credentials_manager = SpotifyClientCredentials(client_id=credentials['spotify']['client_id'],
                                                     client_secret = credentials['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

genius = lyricsgenius.Genius(credentials['genius']['client_access_token'])


pd.set_option('max_columns', None)
# pd.set_option('max_rows', 20)
pd.set_option('max_rows', None)

genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
# genius.excluded_terms = ["(Remix)", "(Live)"] # Exclude songs with these words in their title
# -

# ## Spotify Exctract Data

# + code_folding=[6]
## Spotify functions

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items_all = results['artists']['items']
    
    results = sp.search(q='artist:' + name, type='artist', market='BR')
    items_br = results['artists']['items']
    
    if len(items_br) > 0 and len(items_all) > 0:
        if similar(items_br[0]['name'], name) > similar(items_all[0]['name'], name):
            return items_br[0]
        else:
            return items_all[0]
    else:
        return None
    
    
def get_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album', country='BR')
    albums.extend(results['items'])
    
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
        
    albums_not_duplicates = []

    seen = set()  # to avoid dups
    albums.sort(key=lambda album: album['name'].lower())
    
    for album in albums:
        name = album['name']
        if name not in seen:
            seen.add(name)
            albums_not_duplicates.append(album)
            
    return albums_not_duplicates


def get_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    return tracks

def get_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    return tracks


def get_album_tracks_info_df(tracks):
    infos = ['explicit', 'uri', 'name', 'track_number', 'external_urls']
    
    all_tracks = []
    for t in tracks:
        #Select only relevant info about tracks
        track_dict = {k:v for k,v in t.items() if k in infos}
        
        track_dict['artists_on_track'] = len(t['artists'])
                
        all_tracks.append(track_dict)
        
    # Get audio features for track
    
    tracks_info_df = pd.DataFrame(all_tracks).reset_index().drop(columns=['index'])
    
    tracks_uris = tracks_info_df['uri'].tolist()

    audio_features = sp.audio_features(tracks_uris)
        
    audio_fts = []
    
    for a in audio_features:
        if a:
            audio_fts.append(a)
        else:
            audio_fts.append({})
            
    audio_features_df = pd.DataFrame(audio_fts)
    
    all_infos = pd.concat([tracks_info_df, audio_features_df], axis=1)
    
    # Get track popularity
    tracks_ids = [t['id'] for t in tracks]
    tracks_popularity= pd.DataFrame(sp.tracks(tracks_ids))
    
    popularities = []
    if 'tracks' in tracks_popularity:
        for p in tracks_popularity['tracks']:
            if 'popularity' in p:
                popularities.append(p['popularity'])
            else:
                popularities.append(-1)
        
        all_infos.loc[:, 'song_popularity'] = popularities
    else:
        all_infos.loc[:, 'song_popularity'] = -1

    
    return all_infos


def get_artist_full_discography_infos_df(artist_name):
    
    artist = get_artist(artist_name)

    all_albums = get_artist_albums(artist)
    
    album_infos = ['name', 'release_date', 'release_date_precision', 'total_tracks', 'type']
    
    albums_dfs = []
    
    for album in all_albums:
        print('{} - {}'.format(artist['name'], album['name']))
        tracks = get_album_tracks(album)
        tracks_df = get_album_tracks_info_df(tracks)
        
        for column in album_infos:
            tracks_df.loc[:, 'album_{}'.format(column)] = album[column]
            
        albums_dfs.append(tracks_df)
        
    full_df = pd.concat(albums_dfs, axis=0)

    artist_info = ['followers',  'name', 'popularity']   
    
    full_df.loc[:, 'artist_followers'] = artist['followers']['total']
    full_df.loc[:, 'artist'] = artist['name']
    full_df.loc[:, 'artist_popularity'] = artist['popularity']
 
    return full_df


# + code_folding=[]
artists = ['Taylor Swift', 'Beyoncé', 'Lady Gaga', 'Madonna', 'Katy Perry', 'Rihanna', 
           'Britney Spears', 'Mariah Carey', 'Ariana Grande']


infos_df = []
for a in artists:
    df = get_artist_full_discography_infos_df(a)
    infos_df.append(df)
    
pop_divas_all_albums = pd.concat(infos_df, axis=0).reset_index().drop(columns=['index'])


# -

def is_special_edition(name):
    special_names = ['special', 'edition', 'version', 'deluxe', 'tour', 'live', 'karaoke', 'mix', 
                     'remix', 'soundtrack']
    for s in special_names:
        if s in name.lower():
            return True
    return False


# + code_folding=[]
# Remove special edition albums
pop_divas_all_albums.loc[:, 'paranthesis'] = pop_divas_all_albums['album_name'].str.contains('\([\w+ ]*\)', regex=True)
pop_divas_all_albums.loc[:, 'special_ed'] = pop_divas_all_albums['album_name'].apply(lambda x: is_special_edition(x))
pop_divas_all_albums.loc[:, 'special_ed'] = pop_divas_all_albums['paranthesis'] | pop_divas_all_albums['special_ed']

pop_divas_all_albums.loc[:, 'track_position'] = (pop_divas_all_albums.groupby(['album_name']).cumcount()+1)/pop_divas_all_albums['album_total_tracks']
                     
pop_divas_df_csv = pop_divas_all_albums#pop_divas_all_albums.loc[~pop_divas_all_albums['special_ed']]
pop_divas_df_csv.head()
# -

# Save csv
pop_divas_df_csv.to_csv('./data/pop_divas_df_full.csv', index=False)

# ### Valid data

pop_divas_all_albums = pd.read_csv('./data/pop_divas_df_full.csv')
pop_divas_all_albums.head()

# +
# Remove only deluxe edition if there is other option

valid_albums = pd.read_csv('./data/pop_divas_df_full.csv')

valid_albums = valid_albums.loc[~valid_albums['album_name'].str.lower().str.contains('karaoke')]

valid_albums.loc[:, 'album_original'] = valid_albums['album_name']\
                                            .str.extract(r'(.*)[\(\[]{1}.*[\)\]]{1}').iloc[:, 0].fillna('')\
                                            .str.strip()
valid_albums.loc[:, 'album_original'] = valid_albums.apply(lambda x: x['album_name'] if x['album_original'] == '' else x['album_original'], axis=1) 

valid_albums.loc[:, 'name_size'] = valid_albums['album_name'].str.len()
valid_albums.loc[:, 'popular'] = 100 - valid_albums['song_popularity']


valid_albums = valid_albums\
                .groupby(['artist', 'album_original', 'album_name'], as_index=False)\
                .agg({'album_release_date':'min', 'special_ed':'max', 'album_total_tracks':'min', 'popular':'mean'})\
                .sort_values(by=['artist', 'album_original', 'popular', 'special_ed', 'album_total_tracks', 'album_name'])\
                .drop_duplicates(subset=['album_original'])
    
valid_albums.loc[:, 'original_special_ed'] = valid_albums['album_original'].apply(lambda x: is_special_edition(x))

valid_albums = valid_albums.loc[~valid_albums['original_special_ed']].set_index('artist')#[:, 'album_name'].to_dict()

albums_remove = ['k bye for now', 'The Lion King: The Gift', 'Unplugged', 'Complete Confection',
                'Early Years', 'I\'m Breathless', 'Me. I Am Mariah... The Elusive Chanteuse', 'Red', 
                'Cheek To Cheek'] 

valid_albums.loc[:, 'remove'] = valid_albums['album_original'].apply(lambda x: max([1 if r in x else 0 for r in albums_remove])).astype(bool)

valid_albums = valid_albums.loc[~valid_albums['remove'], 'album_name'].str.strip().tolist()

valid_albums.append('Red (Deluxe Edition)')

pop_divas_df_csv = pop_divas_all_albums.loc[pop_divas_all_albums['album_name'].str.strip().isin(valid_albums)]

# +
# pop_divas_df_csv.groupby(['artist', 'album_name']).agg({'name':'count'})

# +
# pop_divas_df_csv.loc[pop_divas_df_csv['album_name'].str.contains('Red')]
# -

pop_divas_df_csv.to_csv('./data/pop_divas_df_valid_albums.csv', index=False)

# +
#all_songs.groupby(['artist', 'album_name']).agg({'name':'count'})
# -

# ## Lyrics Extract Data

# +
all_songs = pd.read_csv(data_folder/'pop_divas_df_valid_albums.csv')#.set_index('id')

all_songs = all_songs.loc[:, ['name', 'album_name', 'artist', 'uri', 'id']]
all_songs.loc[all_songs['id'].isnull(), 'id'] = all_songs['uri'].apply(lambda x: x.split(':')[2])
# -

artists = all_songs['artist'].unique().tolist()
artists

all_songs.loc[all_songs['name'].str.contains('4 Minutes')]


# +
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


# -

not_found = []
for artist in artists:
    
    
    # Create folder for artist
    Path(lyrics_folder/"{}/".format(artist)).mkdir(parents=True, exist_ok=True)
    os.chdir(lyrics_folder/"{}/".format(artist))
    
    # Check if there are already songs downloaded
    songs_already_downloaded = [song.split('.json')[0] for song in os.listdir() if '.json']

    # Select only songs not already on the folder
    df = all_songs.loc[(all_songs['artist'] == artist) & (~all_songs['id'].isin(songs_already_downloaded))]

    #didnt_find = []
    
    # For all songs on dataframe, call genius lyric search
    for i in range(0, df.shape[0]):
        s = df.iloc[i]

        song_adj = s['name']#result.group(1)
        if song_adj == 'ScheiBe':
            song_adj = 'Scheiße'
        song_id = s['id']#result.group(1)  
        print(song_id)

        try:
            song_lyric = genius.search_song(song_adj, artist)
            
            if not is_artist_on_track(artist, song_lyric):
                print('Não encontrou música original')
       
                # Remove extra info of song lyric
                song_adj = clean_song_name(s['name'])
                song_lyric = genius.search_song(song_adj, artist)

                if not is_artist_on_track(artist, song_lyric):
                    not_found.append(song_adj)
                else:
                    song_lyric.to_json(filename='{}.json'.format(song_id))

            else:
                song_lyric.to_json(filename='{}.json'.format(song_id))

        except:
            song_adj = clean_song_name(s['name'])

            song_lyric = genius.search_song(song_adj, artist)
            
            if is_artist_on_track(artist, song_lyric):
                song_lyric.to_json(filename='{}.json'.format(song_id))
            else:
                not_found.append(song_adj)

not_found

# ## Lyrics Concat 

# +
all_songs_lyrics = []

os.chdir(lyrics_path)
for folder in os.listdir(lyrics_folder):
    artist_folder = lyrics_folder/folder
    os.chdir(artist_folder)
    
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
# -

os.chdir(data_folder)
all_song_lyrics = pd.merge(left=lyrics_df, right=all_songs, left_on='spotify_id', right_on='id')
all_song_lyrics.head()


# + code_folding=[]
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
#         print(t)
        if 'prod' not in t:
            tag = t[1:-1].lower().split(':')[0].strip()
            tag_alphabetic = ''.join([i for i in tag if not i.isdigit()]).strip()
            all_tags.append(tag_alphabetic)

    return all_tags


# +
all_song_lyrics.loc[:, 'artist_wrote_song'] = all_song_lyrics.apply(lambda x: 1 if x['artist'] in x['writers'] else 0, axis=1)
all_song_lyrics.loc[:, 'artist_produced_song'] = all_song_lyrics.apply(lambda x: 1 if x['artist'] in x['producers'] else 0, axis=1)

all_song_lyrics.loc[:, 'structure_tags'] = all_song_lyrics['lyrics'].apply(lambda x: re.findall(r"(\[.*\])", x))

all_song_lyrics.loc[:, 'lyrics_clean'] = all_song_lyrics.apply(lambda x: clean_lyrics(x['lyrics'], x['structure_tags']), axis=1)

all_song_lyrics.loc[:, 'structure_tags_clean'] = all_song_lyrics['structure_tags'].apply(lambda x: clean_structure_tags(x)) 
all_song_lyrics.head()

# +
# Finding most common tags
all_tags = all_song_lyrics.loc[:, ['structure_tags_clean']].explode('structure_tags_clean')
#display(all_tags['song_structure'].value_counts())

valid_tags = ['chorus', 'verse', 'pre_chorus', 'bridge', 'outro', 'intro', 'post_chorus', 
               'refrain', 'hook', 'interlude', 'break', 'solo']
# +
def get_song_structure(song_tags, valid_tags):
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
            
        
all_song_lyrics.loc[:, 'song_structure'] = all_song_lyrics['structure_tags_clean'].apply(lambda x: get_song_structure(x, valid_tags)) 


# # Check most common strucutre
# most_common = all_song_lyrics['song_structure'].value_counts().index.tolist()[0]
# all_song_lyrics['song_structure'].str.contains(most_common).value_counts() 
# -

# ### Lemmatazing 

# +
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])

# +
all_song_lyrics.loc[:, 'total_word_count'] = all_song_lyrics['lyrics_clean'].apply(lambda x: len(nlp(x)))
all_song_lyrics.loc[:, 'lyrics_lemmatized'] = all_song_lyrics['lyrics_clean'].apply(lambda x: ' '.join([token.lemma_.lower() for token in nlp(x) if token.is_alpha and not token.is_stop]))
    
all_song_lyrics['lemma_count'] = all_song_lyrics['lyrics_lemmatized'].apply(lambda x: len(x.split(' ')))
all_song_lyrics['unique_lemmas_on_song'] = all_song_lyrics['lyrics_lemmatized'].apply(lambda x: len(set(x.split(' '))))                                                                          

# + [markdown] code_folding=[]
# ### NRC Lexicon
# -

from nrclex import NRCLex

#Instantiate text object (for best results, 'text' should be unicode).
all_song_lyrics.loc[:, 'nrc_emotions'] = all_song_lyrics['lyrics_clean'].apply(lambda x: NRCLex(x).raw_emotion_scores)
all_song_lyrics.loc[:, 'nrc_emotions_total_words'] = all_song_lyrics['lyrics_clean'].apply(lambda x: len(NRCLex(x).words))


all_song_lyrics.head()

# Save dataset until here
all_song_lyrics.to_csv('./all_song_lyrics_info.csv', index=False)
#flair_sentiment_df.head()

# ### Sentiments

classifier = TextClassifier.load('sentiment')

all_song_lyrics.head()
#all_song_lyrics.index.tolist()

# +
sents = {}

for i in all_song_lyrics.index.tolist():
    if i%100 == 0:
        print(i)
    text = all_song_lyrics.loc[i, 'lyrics_clean']
    sentence = Sentence(text)
    classifier.predict(sentence)
    key = all_song_lyrics.loc[i, 'spotify_id']
    sents[key] = str(sentence.labels)


# +
os.chdir(data_folder)

flair_sentiment_df = pd.DataFrame.from_dict(sents, orient='index').reset_index()
flair_sentiment_df.columns = ['key', 'sentiment']
flair_sentiment_df.to_csv('./flair_sentiment.csv', index=False)

# + [markdown] heading_collapsed=true
# ### Text Embeddings


# + hidden=true
# from flair.embeddings import TransformerDocumentEmbeddings

# embedding = TransformerDocumentEmbeddings('distilbert-base-uncased')

# lyrics_embeddings = all_song_lyrics.loc[:, ['key', 'lyrics_clean']].set_index('key')
# lyrics_embeddings.loc[:, 'lyrics_uncase'] = lyrics_embeddings['lyrics_clean'].str.lower()

# all_embeddings = {}
# count = 0
# for i in lyrics_embeddings.index.tolist():
#     count += 1
#     if count % 100 == 0:
#         print(i)
#     text = lyrics_embeddings.loc[i, 'lyrics_uncase']
#     sentence = Sentence(text)
#     embedding.embed(sentence)
#     numpy_array = sentence.get_embedding().detach().numpy()
#     all_embeddings[i] = numpy_array

# + hidden=true
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
word_embeddings = WordEmbeddings('glove')
stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward-fast'),
                                        FlairEmbeddings('news-backward-fast'),
                                       ])

document_embeddings = DocumentPoolEmbeddings([stacked_embeddings])

# + hidden=true
lyrics_embeddings = all_song_lyrics.loc[:, ['key', 'lyrics_clean']].set_index('key')

all_embeddings = {}
count = 0
for i in lyrics_embeddings.index.tolist():
    count += 1
    if count % 100 == 0:
        print(count)
        text = lyrics_embeddings.loc[i, 'lyrics_clean']
        try:
            sentence = Sentence(text)
            document_embeddings.embed(sentence)
            numpy_array = sentence.get_embedding().detach().numpy()
            all_embeddings[i] = numpy_array
        except:
            all_embeddings[i] = [numpy_array]
            
flair_doc_embeddings = pd.DataFrame.from_dict(all_embeddings, orient='index')
flair_doc_embeddings.to_csv('flair_embeddings.csv')
# -

# ### Final dataset

os.chdir(data_folder)
flair_sentiment_df = pd.read_csv('./flair_sentiment.csv').set_index('key')
flair_sentiment_df.columns = ['flair_sentiment']

song_lyrics = all_song_lyrics.set_index('spotify_id').copy()
song_lyrics = song_lyrics.join(flair_sentiment_df, how='left')\
                .reset_index()\
                .drop(columns='id')\
                .rename(columns={'spotify_id':'id'})

song_lyrics.to_csv('lyrics_info.csv', index=False)
