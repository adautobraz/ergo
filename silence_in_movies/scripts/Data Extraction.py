# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Setup

# +
import json
# from pythonopensubtitles.opensubtitles import OpenSubtitles
# from pythonopensubtitles.utils import File

import imdb
import pandas as pd
import os 
import time
import numpy as np
from pyYify import yify
from pathlib import Path
import aria2p

# initialization, these are the default values
aria2 = aria2p.API(
    aria2p.Client(
        host="http://localhost",
        port=6800,
        secret=""
    )
)

with open('../../.config', 'rb') as f:
    creds = json.load(f)

# + [markdown] heading_collapsed=true
# # IMDB top 250 list

# + hidden=true
ia = imdb.IMDb()
top = ia.get_top250_movies()

# + hidden=true
movies_info = []

for i in range(0, len(top)):
    if i % 50 == 0:
        print(i)
        
    movie_obj = top[i]
    ia.update(movie_obj, info = ['main', 'critic reviews']) 

    code = top[i].movieID
    
    keys = ['title', 'year', 'rating', 'runtime', 'genres', 'cover url', 'directors', 'top 250 rank',
            'votes', 'cast', 'color info', 'original air date', 'plot outline', 'box office', 'metascore'
           ]

    info_dict = {k.replace(' ', '_'): movie_obj[k] for k in keys if k in movie_obj}

    info_dict['imdb_code'] = code
    
    info_dict['directors'] = [d['name'] for d in info_dict['directors']]
    info_dict['cast'] = [d['name'] for d in info_dict['cast']]    

    movies_info.append(info_dict)
    
    
imdb_top_250_df = pd.DataFrame(movies_info)

imdb_top_250_df.loc[:, 'imdb_id'] = 'tt' + imdb_top_250_df['imdb_code']

imdb_top_250_df.head(3)

# + hidden=true
# imdb_top_250_df.to_csv('./data/imdb_top250_movies.csv', index=False)

# + [markdown] heading_collapsed=true
# # Torrents links

# + hidden=true
imdb_top_250_df = pd.read_csv('./data/imdb_top250_movies.csv')
movie_ids = imdb_top_250_df['imdb_id'].unique().tolist()

# + hidden=true
written_files = [f.split('.json')[0] for f in os.listdir('./data/torrent_files/')]

# + hidden=true
for i in range(0, len(movie_ids)):
    
    if movie_ids[i] not in written_files:
        
        try:
    
            search_results = yify.search_movies('{}'.format(movie_ids[i]))

            torr_obj = search_results[0]
            torr_obj.getinfo()

            movie_torrents = torr_obj.torrents

            all_torrents = [[i, movie_torrents[i].magnet, movie_torrents[i].size] for i in range(0, len(movie_torrents))]
            torrent_df = pd.DataFrame(all_torrents, columns=['index','magnet', 'size'])
            torrent_df.loc[:, 'scale'] = torrent_df['size'].str[-3:].str.strip()
            torrent_df.loc[:, 'value'] = torrent_df['size'].str[:-3].astype(float)
            torrent_df.loc[:, 'value_adj'] = torrent_df['value']
            torrent_df.loc[torrent_df['scale'] == 'GB', 'value_adj'] = 1000*torrent_df['value']
            smallest = torrent_df.sort_values(by='value_adj').iloc[0]
            magnet = smallest['magnet']
            size = smallest['size']
            name = smallest['name']

            info_dict = {'magnet':magnet, 'size':size, 'imdb_id':movie_ids[i]}
            print(info_dict)
            
            with open('./data/torrent_files/{}.json'.format(movie_ids[i]), 'w') as fp:
                json.dump(info_dict, fp)
            time.sleep(1)
            
        except:
            print(i)
            continue

        if i%10 == 0:
            print(i)

# + hidden=true
all_torrents_infos = []
written_files = [f for f in os.listdir('./data/torrent_files/')]
for w in written_files:
    with open('./data/torrent_files/{}'.format(w), 'r') as r:
        info = json.load(r)
    all_torrents_infos.append(info)

torrent_info_df = pd.DataFrame(all_torrents_infos)
torrent_info_df.to_csv('./data/torrent_infos.csv', index=False)

torrent_info_df.head()
# -

# # Batch Torrents

torrent_info_df = pd.read_csv('./data/torrent_infos.csv')

data_path = Path('./data')
movies_path = data_path/'movies_raw'
batch_folder = data_path/'batch_files'

all_movies = torrent_info_df['imdb_id'].tolist()
for i in range(0, len(all_movies),10):
    df = torrent_info_df.set_index('imdb_id')
    if i + 5 > len(all_movies):
        df = df.iloc[i:]
    else:
        df = df.iloc[i:(i+10)]
        
    torrents_batch = df['magnet'].to_dict()
    Path(batch_folder/'to_do/').mkdir(parents=True, exist_ok=True)
    file = batch_folder/'to_do/{}_{}.json'.format(i, i+10)
        
    with open(file, 'w') as f:
        json.dump(torrents_batch, f)

# +
file = os.listdir(batch_folder/'to_do')[0]
with open(batch_folder/'to_do'/file, 'r') as f:
    download_files = json.load(f)
    
download_files
df = pd.DataFrame.from_dict(download_files, orient='index').reset_index()
df.columns = ['id', 'magnet']

df.loc[:, 'comand'] = df.apply(lambda x: '{} dir=../movies_raw/{}'.format(x['magnet'], x['id']), axis=1)
df.loc[:, ['comand']]\
    .to_csv(batch_folder/'download/{}.txt'.format(file.split('.')[0]), header=None, index=False)

for imdb_id in df['id'].tolist():
    Path(movies_path/imdb_id).mkdir(parents=True, exist_ok=True)
# -

df

# +
# # ! aria2c "magnet:?xt=urn:btih:AC418DB33FA5CEA4FAB11BC58008FE08F291C9BE&dn=The+Shawshank+Redemption&tr=udp://open.demonii.com:1337/announce&tr=udp://tracker.openbittorrent.com:80&tr=udp://tracker.coppersurfer.tk:6969&tr=udp://glotorrents.pw:6969/announce&tr=udp://tracker.opentrackr.org:1337/announce&tr=udp://torrent.gresille.org:80/announce&tr=udp://p4p.arenabg.com:1337&tr=udp://tracker.leechers-paradise.org:6969&tr=http://track.one:1234/announce&tr=udp://track.two:80"
# -

# ## Detect Silence

# +
import os
from pathlib import Path

movies_path = Path('./data/movies_raw')
movie_files = []

movies = [m for m in os.listdir(movies_path) if '.' not in m]

for m in movies:
        files = ['{}/{}'.format(m,f) for f in os.listdir(movies_path/m) if '.mp4' in f]
        movie_files += files

filepath = movies_path/movie_files[0]
filepath
# -

audios_path = Path('./data/movies_prep/')


# +
# # Convert to wav
# from moviepy.editor import AudioFileClip, VideoFileClip
# videoclip = VideoFileClip('./{}'.format(filepath))

# audioclip = videoclip.audio.subclip(0, 600)
# -

filename = audios_path/movie_files[0].split('/')[0]/'audio/full_audio.wav'
filename

# +
# audioclip.write_audiofile('./{}'.format(filename))

# audioclip.close()
# videoclip.close()
# -

movie_id = movie_files[0].split('/')[0]
audiofile = audios_path/movie_id/'audio/audio_10min.mp3'

# +
import librosa.display
import numpy as np

start = time.time()

x, sr = librosa.load(audiofile)
non_mute = librosa.effects.split(x)

end = time.time()

print((end - start))

# +
from pydub import AudioSegment, silence
start = time.time()

myaudio = AudioSegment.from_mp3('./{}'.format(filename))

silence = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=-16)
silence = [((start/1000),(stop/1000)) for start,stop in silence] #convert to sec

end = time.time()
print((end - start))
# +
import autoscrub

autoscrub.suppress_ffmpeg_output(True)
results = autoscrub.getSilences('./{}'.format(filepath), 
                      input_threshold_dB=-46.0, 
                      silence_duration=1.0, 
                      save_silences=True)
# -

df = pd.DataFrame(results)
df.loc[:, 'start'] = df['silence_start'].apply(lambda x: '{:.0f}:{:02.0f}'.format(int(x/60), x%60))
df.loc[:, 'end'] = df['silence_end'].apply(lambda x: '{:.0f}:{:02.0f}'.format(int(x/60), x%60))
df.sort_values(by='silence_duration', ascending=False)

# +
#pysox
#librosa

#pyAudioAnalysis demorado demais
# -

from ast import literal_eval
import plotly.express as px

# +
imdb_df = pd.read_csv('./data/imdb_top250_movies.csv').set_index('imdb_id')

imdb_df.loc[:, 'genres'] = imdb_df['genres'].apply(lambda x: literal_eval(x))
imdb_df.head()


# +
df = imdb_df\
        .explode('genres')\
        .groupby(['genres'], as_index=False)\
        .agg({'title':'count'})\
        .sort_values(by='title', ascending=False)
        

px.bar(df, x='genres', y='title')

# + [markdown] heading_collapsed=true
# # Data Download

# + hidden=true
series_search = ia.search_movie('The Good Place')
series = ia.get_movie(series_search[0].movieID)
ia.update(series, 'episodes')

# + hidden=true
imdb_array = []
for s, all_episodes in series['episodes'].items():
    for e, e_info in all_episodes.items():
        episode_dict = {
            'season':s,
            'episode':e,
            'imdb_id':e_info.movieID
        }
        imdb_array.append(episode_dict)

imdb_df = pd.DataFrame(imdb_array)
imdb_df.head()

imdb_df.to_csv('./data/the_good_place/imdb_ids.csv', index=False)

# + hidden=true
imdb_df = pd.read_csv('./data/the_good_place/imdb_ids.csv')
imdb_df.head()

# + hidden=true
subtitle_ids = {}
files = [int(f.split('.srt')[0]) for f in os.listdir('./data/the_good_place/raw') if '.srt' in f]
df = imdb_df.loc[~imdb_df['imdb_id'].isin(files)]
for i in df.index.tolist():
    imdbid = str(df.loc[i, 'imdb_id'])
    print(imdbid)

    ost = OpenSubtitles() 
    ost.login(creds['opensubtitles']['user'], creds['opensubtitles']['key'])
    data = ost.search_subtitles([{'sublanguageid': 'eng', 'imdbid': imdbid}])
    if data:
        id_subtitle_file = data[0].get('IDSubtitleFile')
        ost.download_subtitles([id_subtitle_file], output_directory='./data/the_good_place/raw/', extension='srt')
        os.rename('./data/the_good_place/raw/{}.srt'.format(id_subtitle_file), './data/the_good_place/raw/{}.srt'.format(imdbid))
        time.sleep(1)


# + [markdown] heading_collapsed=true
# # Data Format 

# + hidden=true
import pysrt
    
all_subs = []
for f in os.listdir('./data/raw'):
    if 'srt' in f:
        subs = pysrt.open('./data/raw/{}'.format(f))

        episode_subtitles = []
        for i in range(0, len(subs)):
            line = subs[i]
            ep_dict = {
                'text':line.text,
                'start_min':line.start.minutes,
                'start_s':line.start.seconds,
                'start_ms':line.start.milliseconds,
                'end_min':line.end.minutes,
                'end_s':line.end.seconds,
                'end_ms':line.end.milliseconds,
                'position':i
            }
            episode_subtitles.append(ep_dict)

        episode_df = pd.DataFrame(episode_subtitles)
        episode_df.loc[:, 'imdb_id'] = int(f.split('.srt')[0])
        all_subs.append(episode_df)

            
all_subs_df = pd.concat(all_subs, axis=0).set_index('imdb_id')
all_subs_df = all_subs_df.join(imdb_df.set_index('imdb_id'), how='left').reset_index()


# + hidden=true
def check_words(text, words):
    word_size = np.sum([1 if w in text.lower() else 0 for w in words])
    return word_size == len(words)


# + hidden=true
all_subs_df.loc[:, 'time_start'] = all_subs_df['start_min'] + all_subs_df['start_s']/60 + all_subs_df['start_ms']/60000
all_subs_df.loc[:, 'time_end'] = all_subs_df['end_min'] + all_subs_df['end_s']/60 + all_subs_df['end_ms']/60000

all_subs_df.loc[:, 'text_next'] = all_subs_df\
                                    .sort_values(by=['imdb_id', 'time_start'])\
                                    .groupby(['imdb_id'])['text'].shift(-1).fillna(' ')

all_subs_df.loc[:, 'text_previous'] = all_subs_df\
                                    .sort_values(by=['imdb_id', 'time_start'])\
                                    .groupby(['imdb_id'])['text'].shift(1).fillna(' ')

all_subs_df.loc[:, 'phrase'] = (all_subs_df['text_previous'] + ' ' + all_subs_df['text'] + ' ' + all_subs_df['text_next']).str.replace('\n', ' ').fillna('')

all_subs_df.loc[:, 'phrase_match'] = all_subs_df['phrase'].apply(lambda x: check_words(x, ['sex tape']))

# + hidden=true
phrases_df = all_subs_df\
            .loc[all_subs_df['phrase_match']]\
            .sort_values(by=['imdb_id', 'start_min'])\
            .drop_duplicates(subset=['imdb_id', 'start_min'])

phrases_df.shape

# + hidden=true
phrases_df.head()

# + hidden=true
all_subs_df.loc[all_subs_df['phrase_match']]
#text = 
all_subs_df.loc[1258, 'phrase']

# + hidden=true

