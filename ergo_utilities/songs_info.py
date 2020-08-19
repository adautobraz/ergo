## Spotify functions
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from difflib import SequenceMatcher

from pathlib import Path
import copy
import random
import re
import json
import os

import pandas as pd

def setup(): 
    with open('../.config') as f:
        credentials=json.load(f)

    client_credentials_manager = SpotifyClientCredentials(client_id=credentials['spotify']['client_id'],
                                                        client_secret = credentials['spotify']['client_secret'])
    
    global sp 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
   
    
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
        
    albums_not_duplicates = {}  # to avoid dups

    for album in albums:
        name = album['name']
        if name not in albums_not_duplicates or album['total_tracks'] < albums_not_duplicates[name]['total_tracks']:
            albums_not_duplicates[name] = album

    return albums_not_duplicates.values()


def get_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    return tracks


def get_album_tracks_info_df(tracks):
    infos = ['id', 'explicit', 'uri', 'name', 'track_number', 'disc_number', 'external_urls']
    
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
            
    
    remove_columns = ['type', 'id', 'uri', 'track_href', 'analysis_url']   
    audio_features_df = pd.DataFrame(audio_fts).drop(columns=remove_columns)
    
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


def get_artist_full_discography_df(artist_name):
    
    setup()
    
    artist = get_artist(artist_name)

    all_albums = get_artist_albums(artist)
    
    album_infos = ['name', 'release_date', 'release_date_precision', 'total_tracks', 'type', 'images']
    
    albums_dfs = []
    
    for album in all_albums:
        print('{} - {}'.format(artist['name'], album['name']))
        tracks = get_album_tracks(album)
        tracks_df = get_album_tracks_info_df(tracks)
        
        tracks_df
        
        for column in album_infos:
            if column == 'images':
                tracks_df.loc[:, 'album_cover'] = album[column][0]['url']
            else:
                tracks_df.loc[:, 'album_{}'.format(column)] = album[column]
            
        albums_dfs.append(tracks_df)
        
    full_df = pd.concat(albums_dfs, axis=0).set_index('id')

    artist_info = ['followers',  'name', 'popularity']   
    
    full_df.loc[:, 'artist_followers'] = artist['followers']['total']
    full_df.loc[:, 'artist'] = artist['name']
    full_df.loc[:, 'artist_popularity'] = artist['popularity']
    
    full_df.loc[:, 'track_order'] = full_df\
                                        .sort_values(by=['disc_number', 'track_number'])\
                                        .groupby(['album_name'])\
                                        .cumcount() + 1
    
    full_df.loc[:, 'original_album'] = full_df['album_name']\
                                            .str.extract(r'(.*)[\(\[]{1}.*[\)\]]{1}').iloc[:, 0].fillna('')\
                                            .str.strip()
    
    full_df.loc[:, 'original_album'] = full_df.apply(lambda x: x['album_name'] if x['original_album'] == '' else x['original_album'], axis=1) 
 
    return full_df


# Get only one version of album
# Drop duplicates only if there is original option

def is_special_edition(name):
    special_names = ['special', 'edition', 'version', 'deluxe', 'tour', 'live', 'karaoke', 'mix', 
                     'remix', 'soundtrack']
    for s in special_names:
        if s in name.lower():
            return True
    return False


def get_valid_albums(disc_df):

    df = disc_df.copy()

    df.loc[:, 'special_ed'] = df['album_name'].apply(lambda x: is_special_edition(x))
    df.loc[:, 'popular'] = 100 - df['song_popularity']

    valid_albums = df\
                    .groupby(['artist', 'original_album', 'album_name'], as_index=False)\
                    .agg({'album_release_date':'min', 'special_ed':'max', 'album_total_tracks':'min', 'popular':'mean'})\
                    .sort_values(by=['artist', 'original_album', 'special_ed', 'popular', 'album_total_tracks', 'album_name'])\
                    .drop_duplicates(subset=['original_album'], keep='first')

    valid_albums.loc[:, 'valid_album'] = ~valid_albums['original_album'].apply(lambda x: is_special_edition(x))

    valid_albums = valid_albums.loc[valid_albums['valid_album'], 'album_name'].tolist()

    return disc_df.loc[disc_df['album_name'].isin(valid_albums)]