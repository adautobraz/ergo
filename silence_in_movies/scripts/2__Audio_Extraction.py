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

import os
from pathlib import Path
import copy 

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
from ast import literal_eval
import plotly.express as px

data_path = Path('./data')

# initialization, these are the default values
aria2 = aria2p.API(
    aria2p.Client(
        host="http://localhost",
        port=6800,
        secret=""
    )
)

with open('../.config', 'rb') as f:
    creds = json.load(f)
# -

# # Detect Silence

# +


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

filename = audios_path/movie_files[0].split('/')[0]/'audio/audio_10min.mp3'
filename

# +
# audioclip.write_audiofile('./{}'.format(filename))

# audioclip.close()
# videoclip.close()
# -

movie_id = movie_files[0].split('/')[0]
audiofile = audios_path/movie_id/'audio/audio_10min.mp3'

# + [markdown] heading_collapsed=true
# ## Librosa

# + hidden=true
import librosa.display
import numpy as np

start = time.time()

x, sr = librosa.load(audiofile)
non_mute = librosa.effects.split(x)

end = time.time()

print((end - start))

# + [markdown] heading_collapsed=true
# ## Pydub

# + hidden=true
from pydub import AudioSegment, silence
start = time.time()


myaudio = AudioSegment.from_mp3(filename)

silence = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=-16)
silence = [((start/1000),(stop/1000)) for start,stop in silence] #convert to sec

end = time.time()
print((end - start))
# + [markdown] heading_collapsed=true
# ## Autoscrub

# + hidden=true
import autoscrub

autoscrub.suppress_ffmpeg_output(True)
results = autoscrub.getSilences('./{}'.format(filepath), 
                      input_threshold_dB=-46.0, 
                      silence_duration=1.0, 
                      save_silences=True)

# + hidden=true
df = pd.DataFrame(results)
df.loc[:, 'start'] = df['silence_start'].apply(lambda x: '{:.0f}:{:02.0f}'.format(int(x/60), x%60))
df.loc[:, 'end'] = df['silence_end'].apply(lambda x: '{:.0f}:{:02.0f}'.format(int(x/60), x%60))
df.sort_values(by='silence_duration', ascending=False)
# -

# ## FFmpeg



# +
df = pd.read_csv(audios_path/movie_id/'audio/full_audio.mp3-silences.txt', header=None)
df.columns=['start', 'end']

concat_array = []
last_row = {}
for r in df.iterrows():
    row = copy.deepcopy(r[1].to_dict())
    if last_row:
        if row['start'] == last_row['end']:
            last_row['end'] = row['end']
        else:
            concat_array.append(last_row)
            last_row = row
    
    else:
        last_row = row

concat_array.append(last_row)

df = pd.DataFrame(concat_array)
df.loc[:, 'duration'] = df['end'] - df['start']
df.loc[:, 'movie'] = 'Shawshank'
# -

fig = px.timeline(df, x_start='start', x_end='end', y='movie')
fig.layout.xaxis.type = 'linear'
fig.data[0].x = df['duration'].tolist()
fig.show()

print(4045/3600, 4045%3600/60, 4045%60)

df.sort_values(by='duration', ascending=False)

# +
#pysox
#pyAudioAnalysis demorado demais
