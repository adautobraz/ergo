from pathlib import Path

import copy
import random
import re
import json
import os

import requests
import colorgram

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import distance
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json

from PIL import Image, ImageOps, ImageFont, ImageDraw


analysis_folder = Path('/Users/adautobrazdasilvaneto/Documents/ergo/album_journey/analysis/')
data_folder = Path('/Users/adautobrazdasilvaneto/Documents/ergo/album_journey/data/infos')
#---------------------------------------View and layout configuration functions---------------------------------------------#
# def setup(folder_results, infos_folder):
#     analysis_folder = folder_results
#     data_folder = infos_folder


def update_fig(fig):
    fig.update_layout(
        font_family='Fira Sans', 
        template='plotly_white', 
        font_size=18,
        width=640, height=610)
    return fig


def plot(fig):
    fig = update_fig(fig)
    fig.show()
    

def write(fig, name, album, artist, fig_formated=False):
    if not fig_formated:
        fig = update_fig(fig)
    
    dir_name = analysis_folder/"{}/{}/raw".format(artist, album)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    fig.write_image('{}/{}.png'.format(dir_name, name), scale=2)    
    fig.show()
    
    
def add_border(input_image, output_image, color=0):
    
    # Crop images
    img = Image.open(input_image)
    
    width, height = img.size
    # Crop from left, up, right, down
    if 'line' in input_image:
        img = ImageOps.crop(img, (0, 20, 60, 40))
    elif 'bar' in input_image:
        img = ImageOps.crop(img, (0, 0, 40, 40))

    # Add border
    border=(40, 40, 40, 100)
        
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
        
    w,h = img.size
    bw,bh = bimg.size
    
    # Add Swipe
    font_size = 40
    font = ImageFont.truetype("FiraSans-Medium.ttf", font_size)
    text = 'SWIPE >>'
    #     pos = (w + (bw-w)//2 - font.getsize(text)[0], border[1] + h + font.getsize(text)[1]//2 )
    pos = (w + (bw-w)//2 - font.getsize(text)[0], border[1] + h + (border[3] - font_size)//2 - 4)
        
    draw = ImageDraw.Draw(bimg)
    draw.text(pos, text, (255,255,255), font=font)
    
    # Add Logo
    font_size = 50
    font = ImageFont.truetype("Sofia Pro Bold.ttf", font_size)
    text = 'ergo:'
    #     pos = ((bw-w)//2, border[1] + h)
    pos = ((bw-w)//2, border[1] + h +(border[3] - font.getsize(text)[1])//2 - 10)

    draw = ImageDraw.Draw(bimg)
    draw.text(pos, text, (255,255,255), font=font)
    
    bimg.save(output_image)
    
    
def pad_all_graphs(artist, album, rgb):

    os.chdir(analysis_folder/'{}/{}/'.format(artist,album))
    Path('./prep').mkdir(parents=True, exist_ok=True)

    for f in os.listdir('./raw'):
        if '.png' in f:
            add_border('./raw/{}'.format(f),
                       output_image='./prep/{}'.format(f),
                       color=rgb)


def generate_tracklist(space, discography_df, artist, album, color):
    
    df = discography_df.loc[discography_df['album_name'] == album]

    img = Image.new('RGB', (640,640), (color.r, color.g, color.b))

    font_size = 50
    font = ImageFont.truetype("FiraSans-Bold.ttf", font_size)

    name = album
    pos = ((img.size[0] -  font.getsize(name)[0])//2, 30)
    draw = ImageDraw.Draw(img)
    draw.text(pos, name, (255,255,255), font=font)

    # Add Swipe
    font_size = 18
    font = ImageFont.truetype("FiraSans-Bold.ttf", font_size)

    start_v = 150
    start_h = 50

    for i in df.index.tolist():
        s = df.loc[i]
        seconds = s['duration_ms']//1000
        name = '{}. {}'.format(s['track_number'], s['name'])
        pos = (start_h, start_v)
        draw = ImageDraw.Draw(img)
        draw.text(pos, name, (255,255,255), font=font)


        dur = '{}:{:02d}'.format(seconds//60, seconds % 60)
        pos = (640 - 50 - font.getsize(dur)[0], start_v)
        draw = ImageDraw.Draw(img)
        draw.text(pos, dur, (255,255,255), font=font)

        start_v += space
     
    
    font_size = 20
    font = ImageFont.truetype("FiraSans-Medium.ttf", font_size)
    text = 'SWIPE >>'
    pos = (590 - font.getsize(text)[0], 600)
        
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, (255,255,255), font=font)
    
    # Add Logo
    font_size = 20
    font = ImageFont.truetype("Sofia Pro Bold.ttf", font_size)
    text = 'ergo:'
    pos = (50, 595)

    draw = ImageDraw.Draw(img)
    draw.text(pos, text, (255,255,255), font=font)
    
    img.save(analysis_folder/'{}/{}/prep/tracklist.png'.format(artist, album))
    
    return img



    #img.save(output_image)



#----------------------------------- Album graphs functions--------------------------------------------------#

def prep_album_data(artist_discography_df, album_name, features):
    # Filter for album
    df = artist_discography_df.loc[artist_discography_df['album_name'] == album_name]
    
    # Standardize features and bring info
    scale = StandardScaler()
    songs_scaled = pd.DataFrame(scale.fit_transform(df.loc[:, features]))
    
    songs_scaled.index = df.index
    
    songs_scaled.columns = features
    songs_scaled.loc[:, 'artist'] = df['artist']
    songs_scaled.loc[:, 'song'] = df['name']
    songs_scaled.loc[:, 'track_order'] = df['track_order']
    songs_scaled.loc[:, 'position'] = (songs_scaled['track_order'] - 1)/(df['track_order'].max() - 1)
    
    # Calculate distance from avg
    album_mean = songs_scaled.groupby(['artist']).agg({f:'mean' for f in features}).values

    songs_scaled.loc[:, 'distance'] = distance.cdist(songs_scaled.loc[:, features].values, album_mean, 'euclidean')    
    songs_scaled.loc[:, 'uniqueness'] = scale.fit_transform(songs_scaled['distance'].values.reshape(-1,1))

    return songs_scaled
    
    
def break_album_name(name):
    padding = 10
    words = name.split(' - ')[0].split(' ')
    final_str = ''
    aux_str = ''
    for w in words:
        if len(aux_str + w) <= padding:
            aux_str += w + ' '
        else:
            final_str += aux_str.strip() + '<br>'
            aux_str = w + ' '
            
    final_str += aux_str.strip()
            
    words =  final_str.split('<br>')
    if len(words) > 3:
        final_str = '<br>'.join(final_str.split('<br>')[:3]) + '...'
    
    return final_str


def plot_feature_line(std_df, params, feature, offset_axis, color, offset_arrow, annot_array=[]):
    
    # Use dataframe of standardized features of album
    df = std_df.copy()
  
    df.loc[:, 'top_value'] = ''
    df.loc[:, 'rank'] = df[feature].rank(method='first')
    
    # Def amount of highlights on image
    sample=3
    df.loc[df.shape[0] - df['rank'] < sample, 'top_value'] = 'max'
    df.loc[df['rank'] <= sample, 'top_value'] = 'min'
    df.loc[:, 'text'] = df['song'].apply(lambda x: break_album_name(x))


    # Plot fig
    fig = go.Figure()    
    
    fig.add_trace(go.Scatter(x=df['position'], y=df[feature], mode='lines+markers+text', 
        textfont=dict(size=11), marker=dict(color=color),
        line=dict(shape='spline')
    ))
    
    # Setting bounds to figure    
    max_value = max([0.7, df[feature].max()])
    min_value = min([-0.7, df[feature].min()])
    fig.update_yaxes(tickmode='array', ticktext=params['limits'], title='', zerolinewidth=3,
                     tickvals=[min_value,0, max_value], range=[min_value-offset_axis, max_value+offset_axis])
    fig.update_xaxes(range=[-0.1, 1.05], tickmode='array', tickvals=[0,1], title='', 
                     ticktext=['Start', 'End'])
    
    # Adding min and max value indication for feature
    min_max_values = df.loc[df['top_value'] != '']#.drop_duplicates(subset=['top_value', feature], keep='first')
    for i in range(0, min_max_values.shape[0]):
        point = min_max_values.iloc[i]
        annotation = dict(
            showarrow=True,
            x=point['position'],
            y=point[feature],
            text='<i>' + point['text'] + '<i>',
            ayref='y',
            axref='x'
        )
        if annot_array:
            if annot_array[i] == 'up':
                annotation['ax']=point['position']
                annotation['ay']=point[feature] + offset_arrow[1]
            elif annot_array[i] == 'down':
                annotation['ax']=point['position']
                annotation['ay']=point[feature] - offset_arrow[1]
            elif annot_array[i] == 'left':
                annotation['ax']=point['position'] - offset_arrow[0]
                annotation['ay']=point[feature]
            elif annot_array[i] == 'right':
                annotation['ax']=point['position'] + offset_arrow[0]
                annotation['ay']=point[feature]

        else:
            annotation['ax']=point['position']
            if point['top_value'] == 'min':
                annotation['ay']=point[feature] - offset_arrow[1]

            elif point['top_value'] == 'max':
                annotation['ay']=point[feature] + offset_arrow[1]

        fig.add_annotation(annotation)

    fig.update_annotations(font_size=14)
            
    fig.update_layout(title=str('<b>' + params['name'] + '<b>'), title_font_size=24,
                      showlegend=False)

    return fig


def get_features_params(lang, context, subtitle):
        
    feature_params_pt = {
        'danceability':{
            'name':'Dançabilidade, por faixa',
            'limits':['Menos<br>dançante', 'Média<br>do álbum', 'Mais<br>dançante']
        },

        'energy':{
            'name':'Energia, por faixa',
            'limits':['Mais<br>calma', 'Média<br>do álbum', 'Mais<br>agitada']
        },

        'speechiness':{
            'name':'Nível de fala, por faixa',
            'limits':['Mais<br>cantada', 'Média<br>do álbum', 'Mais<br>falada']
        },

        'valence':{
            'name':'Sentimento sônico, por faixa',
            'limits':['Mais<br>triste', 'Média<br>do álbum', 'Mais<br>feliz']
        },

        'uniqueness':{
            'name':'Experimentalidade, por faixa',
            'limits':['Mais<br>comum', 'Média<br>do álbum', 'Mais<br>diferente']
        },

        'acousticness':{
            'name':'Acústica, por faixa',
            'limits':['Menos<br>acústica', 'Média<br>do álbum', 'Mais<br>acústica']
        } 

        ,'instrumentalness':{
            'name':'Instrumental, por faixa',
            'limits':['Menos<br>instrumental', 'Média<br>do álbum', 'Mais<br>instrumental']
        },

        'tempo':{
            'name':'Velocidade da batida, por faixa',
            'limits':['Mais<br>devagar', 'Média<br>do álbum', 'Mais<br>rápida']
        },

        'loudness':{
            'name':'Altura do som, por faixa',
            'limits':['Mais<br>baixa', 'Média<br>do álbum', 'Mais<br>alta']
        }     
    }


    feature_params_en = {
        'danceability':{
            'name':'Danceability',
            'limits':['Less<br>danceable', 'More<br>danceable']
        },

        'energy':{
            'name':'Energy',
            'limits':['More<br>calm', 'More<br>energetic']
        },

        'speechiness':{
            'name':'Speechiness',
            'limits':['More<br>singing', 'More<br>talking']
        },

        'valence':{
            'name':'Sound feeling',
            'limits':['Darker<br>sound', 'Brighter<br>sound']
        },

        'uniqueness':{
            'name':'Experimentalness',
            'limits':['More<br>common', 'More<br>experimental']
        },

        'acousticness':{
            'name':'Acousticness',
            'limits':['Less<br>acoustic', 'More<br>acoustic']
        } 

        ,'instrumentalness':{
            'name':'Instrumentalness',
            'limits':['More<br>vocal', 'More<br>instrumental']
        },

        'tempo':{
            'name':'Tempo',
            'limits':['Slower<br>beats', 'Faster<br>beats']
        },

        'loudness':{
            'name':'Loudness',
            'limits':['Quieter<br>songs', 'Louder<br>songs']
        },
        'sentiment_score':{
            'name':'Lyrics emotions',
            'limits':['More negative<br>feelings', 'More positive<br>feelings']
        },

    }
    
    for f, feature_dict in feature_params_en.items():
        feature_dict['name'] +=  ', ' + subtitle
        limits_array = feature_dict['limits']
        feature_dict['limits'] = [limits_array[0], context + '<br>average',  limits_array[1]]
    
    if lang == 'en':
        return feature_params_en
    
    else:
        return feature_params_pt



def get_album_color(artist_df, album, artist):
    
    link = artist_df\
            .loc[artist_df['album_name'] == album]\
            .drop_duplicates(subset=['album_cover']).iloc[0]['album_cover']
    
    img_data = requests.get(link).content
    
    Path(data_folder/'{}/covers/'.format(artist)).mkdir(parents=True, exist_ok=True)
    
    with open(data_folder/'{}/covers/{}.jpg'.format(artist, album), 'wb') as handler:
        handler.write(img_data)    

    # Extract 6 colors from an image.
    colors = colorgram.extract(data_folder/'{}/covers/{}.jpg'.format(artist, album), 10)
    
    # Save test colors
    df = pd.DataFrame({'proportion': [c.proportion for c in colors],
                        'color': ['rgb({},{},{})'.format(c.rgb.r, c.rgb.g, c.rgb.b) for c in colors]})\
        .reset_index()

    fig = px.bar(df, y='index', x='proportion', color='color', orientation='h',
                color_discrete_sequence = df['color'].values)
    plot(fig)

    return colors


#----------------------------------- Discography graphs functions--------------------------------------------------#

def scale_discography(discography_df, features):

    # Filter for album
    df = discography_df.copy()

    # Standardize features for all songs, rejoin with info
    scale = StandardScaler()
    songs_scaled = pd.DataFrame(scale.fit_transform(df.loc[:, features]))
    
    songs_scaled.index = df.index

    songs_scaled.columns = features
    songs_scaled.loc[:, 'artist'] = df['artist']
    songs_scaled.loc[:, 'album'] = df['album_name']
    songs_scaled.loc[:, 'year'] = df['album_release_date'].str[:4].astype(int)
    songs_scaled.loc[:, 'song'] = df['name']
    songs_scaled.loc[:, 'album_order'] = songs_scaled.groupby(['artist'])['year'].rank(method='dense').astype(int)
    
    # find each song distance from album mean
    dist = {}

    for album in discography_df['album_name'].unique().tolist():
        df = discography_df.loc[discography_df['album_name'] == album]
        songs_df = pd.DataFrame(scale.fit_transform(df.loc[:, features]))
        songs_df.index = df.index
        songs_df.columns = features

        songs_df.loc[:, 'artist'] = df['artist']

        album_mean = songs_df.groupby(['artist']).agg({f:'mean' for f in features}).values

        songs_df.loc[:, 'distance_from_album_mean'] = distance.cdist(songs_df.loc[:, features].values, album_mean, 'euclidean')    
        dist.update(songs_df['distance_from_album_mean'].to_dict())

    all_songs_uniqueness = pd.DataFrame.from_dict(dist, orient='index')
    all_songs_uniqueness.columns = ['experimentalness_on_album']
    
    songs_scaled.loc[:, 'uniqueness'] = all_songs_uniqueness['experimentalness_on_album']

    return songs_scaled


def get_album_evolution_past_disc(discography_df, artist, features, mode='all'):
    
    # Standardize discography
    songs_scaled = scale_discography(discography_df, features)

    # Calculate mean album representation
    albums_mean = songs_scaled\
                    .groupby(['artist', 'album', 'year'], as_index=False)\
                    .agg({f:'mean' for f in features})

    albums_mean.loc[:, 'album_order'] = albums_mean.groupby(['artist'])['year'].rank().astype(int)

    # Find artist mean representation until album
    artist_average = []
    total_albums = albums_mean.shape[0]
    for i in range(2, total_albums + 1):
        if mode == 'all':
            current_work_df = songs_scaled.loc[songs_scaled['album_order'] < i]
        else:
            current_work_df = songs_scaled.loc[songs_scaled['album_order'] == i - 1]

        artist_rep = current_work_df.groupby(['artist']).agg({c:'mean' for c in features}).iloc[0].to_dict()
        artist_rep['till_album'] = i      
        artist_average.append(artist_rep)

        
    mean_till_album = pd.DataFrame(artist_average)
    mean_till_album.loc[:, 'album'] = artist + '-' + mean_till_album['till_album'].astype(str)
    mean_till_album = mean_till_album.set_index('album').loc[:, features]


    artist_avg_repr = mean_till_album.values
    artist_albums_array = albums_mean.loc[:, features].values

    dist_album_progress = pd.DataFrame(distance.cdist(artist_albums_array, artist_avg_repr, 'euclidean'))
    dist_album_progress.columns = mean_till_album.index.tolist()
    dist_album_progress.loc[:, 'album_order'] = albums_mean['album_order']
    dist_album_progress.loc[:, 'artist'] = albums_mean['artist']
    dist_album_progress.loc[:, 'album'] = albums_mean['album']

    dist_album_progress = pd.melt(dist_album_progress, 
                                  id_vars=['album', 'artist', 'album_order'],
                                  value_vars = mean_till_album.index.tolist())

    dist_album_progress.loc[:, 'artist_ref'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,0]
    dist_album_progress.loc[:, 'album_ref_order'] = dist_album_progress['variable'].str.split('-', expand=True).iloc[:,1 ].astype(int)

    # Count only distance until current album
    dist_album_progress = dist_album_progress\
                            .loc[
                                (dist_album_progress['artist'] == dist_album_progress['artist_ref'])
                                & (dist_album_progress['album_order'] == dist_album_progress['album_ref_order'])
                                ]

    # Concatenate info from album with other info
    all_albums_info = albums_mean.set_index(['album', 'artist']).loc[:, ['year', 'album_order']]
    dist_album_progress = dist_album_progress.set_index(['album', 'artist']).loc[:, ['value']]

    album_diff_prev = pd.concat([all_albums_info,dist_album_progress], axis=1).fillna(0).reset_index()
    album_diff_prev.sort_values(by=['artist', 'album_order'], inplace=True)

    return album_diff_prev


def get_album_highlights(discography_df, features, album):

    df = scale_discography(discography_df, features)
    
    features_album =  features + ['uniqueness']

    albums_mean = df\
                    .groupby(['artist', 'album', 'year'], as_index=False)\
                    .agg({f:'mean' for f in features_album})
    
    

    melt = pd.melt(albums_mean, id_vars=['album', 'year', 'artist'], value_vars=features_album)

    melt.loc[:, 'max'] = melt.groupby(['variable'])[ 'value'].transform('max')
    melt.loc[:, 'min'] = melt.groupby(['variable'])[ 'value'].transform('min')

    melt.loc[:, 'is_highlight'] = melt.apply(lambda x: x['value'] in [x['max'], x['min']], axis=1)

    highlights = melt.loc[(melt['is_highlight']) & (melt['album'] == album)]
    
    return highlights


def plot_discography_evolution(discography_df, artist, features, color, mode='all'):

    # Use dataframe of standardized features of album
    df = get_album_evolution_past_disc(discography_df, artist, features, mode)
    
    df.loc[:, 'album_break'] = df['album'].apply(lambda x: break_album_name(x))
  
    # Plot fig
    fig = go.Figure()    
    
    fig.add_trace(go.Scatter(x=df['album_break'], y=df['value'], mode='lines+markers+text', 
        textfont=dict(size=11), marker=dict(color=color),
        line=dict(shape='spline', color=color)
    ))
    
    fig.update_yaxes(showticklabels=False, tickmode = 'linear', tick0 = 1, dtick = 1)
    fig.update_xaxes(titlefont_size=16)
    fig.update_layout(title='<b>Reinventing from past discography<b>', yaxis_title='Distance from previous work')
    return fig


def plot_album_feature_comparison(discography_df, feature_columns, feature, album, color, feature_params):
    
    if feature in feature_params:
        # Standardize discography
        songs_scaled = scale_discography(discography_df, feature_columns)
        
        features = feature_columns + ['uniqueness']

        # Calculate mean album representation
        albums_mean = songs_scaled\
                        .groupby(['artist', 'album', 'year'], as_index=False)\
                        .agg({f:'mean' for f in features})\
                        .sort_values(by=feature)

    #     albums_mean.loc[:, 'album_order'] = albums_mean.groupby(['artist'])['year'].rank().astype(int)

        color_dict = {a: '#EAE7E7' for a in albums_mean['album'].unique().tolist()}
        color_dict[album] = color
        
        title = feature_params[feature]['name']

        fig = px.bar(albums_mean, y='album', x=feature, color='album', color_discrete_map=color_dict)
        fig.update_layout(showlegend=False, title='<b>'+title+'<b>',
                        yaxis_title='')
        
        axis_range_limit = albums_mean[feature].abs().max()
        
        fig.update_xaxes(
            tickvals = [-axis_range_limit, 0, axis_range_limit],
            ticktext = feature_params[feature]['limits'],
            title='',
            range = [-axis_range_limit-0.1, axis_range_limit+0.1]
        )
            
        return fig