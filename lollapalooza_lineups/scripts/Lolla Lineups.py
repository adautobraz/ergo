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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
from pathlib import Path
import os
import plotly.express as px
import json
from IPython.display import display, HTML
from difflib import SequenceMatcher
import math
import numpy as np
import statistics

pd.set_option('max_rows', None)
pd.set_option('max_columns', None)


# -

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# # Data compilation

data_path = Path('/Users/adautobrazdasilvaneto/Documents/ergo/lineups_dive/data/')             

all_lineups = [f for f in os.listdir(data_path/'lineups') if 'csv' in f]

# +
dfs = []
for f in all_lineups:
    df = pd.read_csv(data_path/'lineups'/f, delimiter=';')
    df = df.loc[:, ['hour_start', 'hour_end', 'artist', 'stage', 'date']]
    dfs.append(df)
    
lineups = pd.concat(dfs, axis=0)
lineups.reset_index(inplace=True)
lineups.drop(columns=['index'], inplace=True)
# -

lineups = lineups.loc[lineups['stage'] != 'Kidzapalooza']


def format_hour(hour_str):
    h_array = hour_str.split('h')
    correct_hour = h_array[0] + ':'
    if h_array[1] == '':
        correct_hour += '00'
    else:
        correct_hour += h_array[1]
    return correct_hour


lineups.loc[:, 'hour_start_adj'] = lineups['hour_start'].apply(lambda x: format_hour(x))
lineups.loc[:, 'hour_end_adj'] = lineups['hour_end'].apply(lambda x: format_hour(x))

# +
lineups.loc[:, 'datetime_start'] = pd.to_datetime((lineups['date'] + ' ' + lineups['hour_start_adj']), format='%Y-%m-%d %H:%M')
lineups.loc[:, 'datetime_end'] = pd.to_datetime((lineups['date'] + ' ' + lineups['hour_end_adj']), format='%Y-%m-%d %H:%M')

lineups.loc[:, 'duration_min'] = (lineups['datetime_end'] - lineups['datetime_start']).dt.seconds/60
lineups.loc[:, 'year'] = lineups['datetime_start'].dt.year
lineups.loc[:, 'order_in_lineup'] = lineups['datetime_start'].dt.hour
# -

lineups.loc[:, 'date'] = lineups['date'].str.strip()
lineups.loc[:, 'stage'] = lineups['stage'].str.strip()
lineups.loc[:, 'artist_name'] = lineups['artist'].str.strip()

lineups.loc[:, 'palco'] = lineups['stage'].apply(lambda x: x.split('Palco')[1].strip() if 'Palco' in x else x)
lineups.loc[lineups['stage'].str.contains('Perry'), 'palco'] = 'Perry\'s'

lineups.loc[:, 'artist_id'] = lineups['artist_name'] + '_' + lineups['date']
lineups.set_index('artist_id', inplace=True)

# +
order_in_lineup = lineups.groupby(['date']).agg({'datetime_start':['min', 'max']})
order_in_lineup.columns=['start', 'end']
order_in_lineup = order_in_lineup.to_dict(orient='index')

def get_order_in_lineup(time, order_lineup):
    key = time.strftime('%Y-%m-%d')
    porcentage = 100*((time - order_lineup[key]['start']).seconds)/((order_lineup[key]['end'] - order_lineup[key]['start']).seconds)
    return porcentage

lineups.loc[:, 'order_in_lineup'] = lineups['datetime_start'].apply(lambda x: get_order_in_lineup(x,order_in_lineup))
# -

lineups.loc[:, 'importance_order'] = (np.rint(4*lineups['order_in_lineup']/100) + 1).astype(int)
lineups.loc[:, 'importance_log'] = lineups['order_in_lineup'].apply(lambda x: math.log10(x + 1))

lineups['importance_order'].value_counts().sort_index()

# +
#Saving artists to generate discography dict
# lineups.loc[:, ['artist']].drop_duplicates().to_csv(data_path/'all_artists.csv', index=False)
# -

# # LastFM Tags

with open(data_path/'lastfm_tags_per_act.json', 'r') as j:
    artist_tags = json.load(j)

# +
all_tags = {}
no_tags = []

for act, acts_array in artist_tags.items():
    for a in acts_array:
        if a:    
            tags_dict = a['tags']
            
            for tag, value in tags_dict.items():
                if tag in all_tags:
                    all_tags[tag.lower()] += value/100
                else:
                    all_tags[tag.lower()] = value/100
        else:
            no_tags.append(act)
            
            
all_tags_df = pd.DataFrame.from_dict(all_tags, orient = 'index').reset_index()
all_tags_df.columns = ['tag', 'presence']
# -

all_tags_df.sort_values(by='presence', ascending=False)


def get_all_tags_act(key, acts_dict):
    all_tags = {}
    act_array = acts_dict[key]
    acts_amount = len(act_array)
    for a in act_array:
        if a:     
            tags_dict = a['tags']
            for tag, value in tags_dict.items():
                if tag.lower() in all_tags:
                    all_tags[tag.lower()] += value/(100*acts_amount)
                else:
                    all_tags[tag.lower()] = value/(100*acts_amount)
    return all_tags


lineups.loc[:, 'lastfm_tags'] = lineups['artist'].apply(lambda x: get_all_tags_act(x, artist_tags))

# +
all_tag_keywords = {}
no_tags = []

for act, acts_array in artist_tags.items():
    for a in acts_array:
        if a:    
            tags_dict = a['tags']
            
            for tag, value in tags_dict.items():
                if value >= 0:
                    for s in tag.split():
                        key = s.lower().strip()
                        if key in all_tag_keywords:
                            all_tag_keywords[key] += value/100
                        else:
                            all_tag_keywords[key] = value/100
        else:
            no_tags.append(act)
                        
tags_keywords = pd.DataFrame.from_dict(all_tag_keywords, orient = 'index').reset_index()
tags_keywords.columns = ['keyword', 'presence']

tags_keywords.sort_values(by='presence', ascending=False, inplace=True)
tags_keywords.head(25)
# -

genre_words =['rock', 'indie', 'pop', 'alt', 'electro', 'house', 'rap', 'hop']

genre_tags = {f:[] for f in genre_words}
for t in all_tags_df['tag'].tolist():
    for b in genre_words:
        if b in t:
            genre_tags[b].append(t)


# +
def check_genre(artist_tags, genre_dict):
    genre_tags_count = {}
    for genre, all_tags_of_genre in genre_dict.items():
        genre_tags_count[genre] = 0
        for tag, value in artist_tags.items():
            if tag in all_tags_of_genre:
                genre_tags_count[genre] += value
    
    return genre_tags_count

lineups.loc[:, 'total_tags_amount'] = lineups['lastfm_tags'].apply(lambda x: sum(x.values()))
lineups.loc[:, 'lastfm_genre_tags'] = lineups.apply(lambda x: {k:v/x['total_tags_amount'] for k,v in check_genre(x['lastfm_tags'], genre_tags).items()} if x['total_tags_amount'] > 0 else {} , axis=1)


# -

def check_tags(tags_dict, words_to_check):
    probability = 0
    for tag, importance in tags_dict.items(): 
        for w in words_to_check:
                if w == tag:
                    probability += importance 
    return probability


# +
# Encontrando tags para artistas homens e mulheres

fem_words = ['fem', 'mulher', 'famale']
male_words = ['male', 'masc']

fem_tags = []
for t in all_tags_df['tag'].tolist():
    for f in fem_words:
        if f in t:
            fem_tags.append(t)

male_tags = []
for t in all_tags_df['tag'].tolist():
    for m in male_words:
        if m in t and t not in fem_tags:
            male_tags.append(t)
            
lineups.loc[:, 'fem_tag'] = lineups['lastfm_tags'].apply(lambda x: check_tags(x, fem_tags))
lineups.loc[:, 'male_tag'] = lineups['lastfm_tags'].apply(lambda x: check_tags(x, male_tags))

# +
# Palavras com conotação brasileira
br_words = ['brazil', 'brasil', 'mpb', 'guacho', 'nacional', 'paraense', 'brega', 'bahia']

br_tags = []
for t in all_tags_df['tag'].tolist():
    for b in br_words:
        if b in t:
            br_tags.append(t)

lineups.loc[:, 'br_tag'] = lineups['lastfm_tags'].apply(lambda x: check_tags(x, br_tags))
# -

# ## Gêneros

lineups.head(5)

# +
genre_per_act = lineups['lastfm_genre_tags'].apply(pd.Series).fillna(0)

genre_colummns = list(genre_per_act.keys())

genre_per_act.loc[:, 'total']  = genre_per_act.sum(axis=1)
genre_per_act = genre_per_act.div(genre_per_act['total'], axis=0).fillna(0).reset_index()

genre_per_act = pd.melt(genre_per_act, id_vars='artist_id', value_vars = genre_colummns, var_name = 'genre').set_index('artist_id')
genre_per_act = genre_per_act.join(lineups.loc[:, ['palco', 'year', 'artist_name', 'date', 'order_in_lineup', 'duration_min', 'importance_log', 'importance_order']])

genre_per_act.head()

# +
genres_per_year = genre_per_act.groupby(['year', 'genre'], as_index=False).agg({'value':'sum', 'artist_name':'count'})

genres_per_year.loc[:, 'genre_importance'] = genres_per_year['value']/genres_per_year['artist_name']

# +
fig = px.scatter(genres_per_year, x='year', y='genre_importance', facet_col='genre', trendline='ols', color='genre',
          title='Importância de cada gênero musical no Lineup, linha de tendência',
          category_orders={'genre':['rock', 'electro', 'indie', 'alt', 'hop', 'pop', 'house','rap']})

fig.update_layout(
    font=dict(
        family="Lato"),
    showlegend=False
)

fig.for_each_annotation(lambda a: a.update(text=a.text.replace("genre=", "")))

fig.show()
# -

# Tendências de gênero musical:
# - Menos alternativo
# - Menos electro
# - Menos hip-hop
# - Mais pop
# - Mais rap
# - Menos rock

# +
genres_per_year_order = genre_per_act.groupby(['year', 'genre', 'importance_order'], as_index=False).agg({'value':'sum', 'artist_name':'count'})

genres_per_year_order.loc[:, 'genre_importance'] = genres_per_year_order['value']/genres_per_year_order['artist_name']

# +
fig = px.scatter(genres_per_year_order, facet_col='genre', y='genre_importance', x='year',
                 color='genre', trendline='ols', facet_row='importance_order', height=800,
          title='Importância de cada gênero musical no Lineup, linha de tendência, por ordem no lineup',
          category_orders={'genre':['rock', 'electro', 'indie', 'alt', 'hop', 'pop', 'house','rap']})

fig.show()

# +
fig = px.density_heatmap(genre_per_act, animation_frame='genre', z='value', y='importance_order',range_y=[0.5,5.5],
                 x='year', histfunc='avg',color_continuous_scale=px.colors.sequential.RdPu, nbinsx=9, nbinsy=6,
          title='Evolução da relevância de cada gênero musical por horário do Lineup',
          category_orders={'genre':['rock', 'electro', 'indie', 'alt', 'hop', 'pop', 'house','rap']})

fig.show()
# -

# ## Artistas mulheres

# +
lineups.loc[:, 'female_presence'] =  0
lineups.loc[lineups['fem_tag'] > lineups['male_tag'] , 'female_presence'] = 1

# Casos em que há erros
fem_artists = ['Sofi Tukker', 'Francisco, el hombre', 'DJ Anna', 'Liniker e os Caramelows',
              'Ventre', 'Plutão já foi planeta', 'Aláfia', 'Velhas Virgens', 'Devochka',
              'BRVNKS', 'Barja', 'Mc Tha', 'Ashibah']

male_artists = ['Seed', 'Lennox', 'Fisher', 'Alan Walker', 'Goldfish']

lineups.loc[lineups['artist_name'].isin(fem_artists), 'female_presence'] = 1
lineups.loc[lineups['artist_name'].isin(male_artists), 'female_presence'] = 0


lineups.loc[lineups['female_presence'] == 1, 'Atração com vocais femininos?'] = 'Sim'
lineups.loc[lineups['female_presence'] == 0, 'Atração com vocais femininos?'] = 'Não'
# -

lineups['female_presence'].value_counts()

# +
fig = px.histogram(lineups, x='year', color='Atração com vocais femininos?', barmode='group',
            title='Artistas Homens x Mulheres no Lolla, por ano')

fig.update_layout(title_font_size=24, xaxis_title='Ano', yaxis_title='Quantidade de artistas')
fig.show()

# +
gender_stats = lineups.groupby(['year', 'female_presence']).agg({'artist':'count'})
gender_stats.loc[:, 'percentage'] = gender_stats.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))['artist']
gender_stats.reset_index(inplace=True)

gender_stats.loc[:, 'female_presence'] = gender_stats['female_presence'].astype(bool)
# -

fig = px.scatter(gender_stats, x='year', y='percentage', symbol='female_presence', trendline='ols', color='female_presence')
fig.show()

# +
import numpy as np
from sklearn.linear_model import LinearRegression

x = gender_stats.loc[gender_stats['female_presence'] == 1, 'year'].values.reshape(-1, 1)
y = gender_stats.loc[gender_stats['female_presence'] == 1, 'percentage'].values

model = LinearRegression()
model.fit(x, y)

year_of_equality = (50 - model.intercept_)/model.coef_

year_of_equality
# -

# A igualdade vem: Lolla 2050!

gender_stats_per_stage = lineups.groupby(['year', 'palco']).agg({'female_presence':'sum', 'artist':'count'})
gender_stats_per_stage.loc[:, 'female_quota'] = 100*gender_stats_per_stage['female_presence']/gender_stats_per_stage['artist']
gender_stats_per_stage.reset_index(inplace=True)

px.bar(gender_stats_per_stage, x='year', y='female_quota', color='palco', barmode='group',
      title='Participação feminina, por palco, por ano').show()

# Destaque para o Palco Perry's que desde 2016 tem uma participação reduzida de mulheres no seu lineup

acts_per_stage_per_day = lineups.groupby(['date', 'palco'], as_index=False).agg({'artist':'count'})
px.histogram(acts_per_stage_per_day, x='date', y='artist', histfunc='avg')

# Cerca de 6 artistas por Palco. Vamos dividir o lineup em faixas de 6 momentos!

fig = px.histogram(lineups, x='order_in_lineup', color='female_presence',barmode='group', histnorm='percent', nbins=6)
fig.show()

# Chances altíssimas do penúltimo artista no palco ser uma mulher!  

# +
total_female_presence = lineups.groupby(['year']).agg({'female_presence':'sum'}).to_dict(orient='index')

lineups.loc[:,'fem_participation_normalized'] = lineups['year'].apply(lambda x: 1/total_female_presence[x]['female_presence'])

# +
only_girls = lineups.loc[lineups['female_presence'] == 1].sort_values(by='year')

fig = px.density_heatmap(only_girls, y='order_in_lineup',  x='year', z='fem_participation_normalized', 
                         histfunc='sum', range_y = [0,100], nbinsy=6, nbinsx=9,
                        color_continuous_scale=px.colors.sequential.RdPu)
fig.show()

# +
genre_per_act = lineups['lastfm_genre_tags'].apply(pd.Series).fillna(0)

genre_colummns = list(genre_per_act.keys())

genre_per_act.loc[:, 'total']  = genre_per_act.sum(axis=1)
genre_per_act = genre_per_act.div(genre_per_act['total'], axis=0).fillna(0).reset_index()

genre_per_act = pd.melt(genre_per_act, id_vars='artist_id', value_vars = genre_colummns, var_name = 'genre').set_index('artist_id')
genre_per_act = genre_per_act.join(lineups.loc[:, ['palco', 'year', 'artist_name', 'date', 'order_in_lineup', 'duration_min', 'female_presence']])
# -

genres_per_year_gender = genre_per_act.groupby(['year', 'genre', 'female_presence'], as_index=False).agg({'value':'mean', 'artist_name':'count'})
genres_per_year_gender.loc[:, 'artist_gender'] = 'male'
genres_per_year_gender.loc[genres_per_year_gender['female_presence']==1, 'artist_gender'] = 'female'
genres_per_year_gender.head()

px.scatter(genres_per_year_gender, x='year', y='value', facet_col='genre', trendline='ols', color='artist_gender',
           title='Tendência de cada gênero musical, por gênero', symbol='artist_gender',
          category_orders={'genre':['rock', 'electro', 'indie', 'alt', 'hop', 'pop', 'house','rap']})

# - Menos 'rock', 'electro' e 'house'
# - Mais 'indie', 'alt' e 'pop' que os homens

# ## Artistas brasileiros

# +
lineups.loc[:, 'is_br'] = 0
lineups.loc[lineups['br_tag'] > 0, 'is_br'] = 1


is_br = ['Balls', 'Veiga e Salazar', 'Marcio Techjun', 'Daniel Belleza e os Corações em Fúria',
         'Daniel Brandão', 'King of Swingers', 'Mix Hel', 'Lennox', 'República',
         'Elekfantz', 'Cone Crew', 'Victor Ruiz AV Any Mello', 'Fatnotronic',
         'Zerb', 'Ricci', 'Gabriel Boni', 'Kilotones', 'Devochka',
         'Gustavo Mota', 'Jetlag', '89FM', 'Bruno Be', 'Maz', 'Gabriel,O Pensador',
         'KVSH', 'Pontifexx', 'Barja', 'Beowülf', 'Fractall x Rocksted',
         'Victor Lou', 'Fancy Inc', 'Evokings'
        ]

lineups.loc[lineups['artist_name'].isin(is_br), 'is_br'] = 1

# +
# Corrigir dados de artistas brasileiros

# +
total_br_presence = lineups.groupby(['year']).agg({'is_br':'sum'}).to_dict(orient='index')

lineups.loc[:,'br_participation_normalized'] = lineups['year'].apply(lambda x: 1/total_br_presence[x]['is_br'])

# +
br_stats = lineups.groupby(['year', 'is_br']).agg({'artist':'count'})
br_stats.loc[:, 'percentage'] = br_stats.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))['artist']
br_stats.reset_index(inplace=True)

only_br = br_stats.loc[br_stats['is_br'] == 1]
px.line(only_br, x='year', y='percentage', range_y=[0,50], line_shape='spline')
# -

fig = px.histogram(lineups, x='order_in_lineup', color='is_br',barmode='group', histnorm='percent', nbins=6,
                  title='Distribuição dos artistas brasileiros na ordem do lineup')
fig.show()

# 71% dos artistas brasileiros são colocados no início do lineup (até 30% do dia)

# +
only_br = lineups.loc[lineups['is_br'] == 1]

fig = px.density_heatmap(only_br, y='order_in_lineup',  x='year', z='br_participation_normalized', 
                         histfunc='sum', range_y = [0,100], nbinsy=6, nbinsx=9,
                        color_continuous_scale=px.colors.sequential.YlGn)
fig.show()

# +
br_stats_stage = lineups.groupby(['year', 'palco'], as_index=False).agg({'is_br':'sum', 'artist':'count'})
br_stats_stage.loc[:, 'br_participation'] = br_stats_stage['is_br']/br_stats_stage['artist']

px.line(br_stats_stage, x='year', y='br_participation', color='palco', range_y=[0, 0.6], line_shape='spline')
# -

# # Spotify - Reading discography dicts

lineups.head(3)

with open(data_path/'artists_discography_summary.json', 'r') as j:
    artist_info = json.load(j)

artist_info['Mano Brown']

# + [markdown] heading_collapsed=true
# ## Album Info

# + hidden=true
artists_discography = []
for act, artists_array in artist_info.items():
    for artist_dict in artists_array:
        if 'albums_infos' in artist_dict and artist_dict['albums_infos'] != None :
            for album, album_info in artist_dict['albums_infos'].items():
                album_dict = album_info.copy()
                album_dict['act'] = act
                album_dict['artist'] = artist_dict['name']
                artists_discography.append(album_dict)
                
albums_df = pd.DataFrame(artists_discography)
albums_df.loc[:, 'act'] = albums_df['act'].str.strip()

# Retirando álbuns de Remix e Comentário, Tirando duplicatas
albums_df = albums_df.loc[ ~(albums_df['name'].str.lower().str.contains('remix'))
                            & ~(albums_df['name'].str.lower().str.contains('commentary'))
                            & ~(albums_df['name'].str.lower().str.contains('comenta'))
                            & ~(albums_df['name'].str.lower().str.contains('spotify'))]

albums_df = albums_df.drop_duplicates(subset=['release_date', 'total_tracks', 'artist'])

# Coluna para versão especial de disco
albums_df.loc[:, 'album_version'] = 'studio'
albums_df.loc[(albums_df['name'].str.lower().str.contains('bonus'))
                | (albums_df['name'].str.lower().str.contains('live'))
                | (albums_df['name'].str.lower().str.contains('ao vivo'))
                | (albums_df['name'].str.lower().str.contains('version'))
                | (albums_df['name'].str.lower().str.contains('edition'))
                | (albums_df['name'].str.lower().str.contains('deluxe'))
               , 'album_version'] = 'special'


albums_df.loc[albums_df['release_date'].str.len() == 10, 'release_date_adj'] = albums_df['release_date']   
albums_df.loc[albums_df['release_date'].str.len() == 4, 'release_date_adj'] = albums_df['release_date'] + '-01-01'  

albums_df.loc[:, 'release_date_adj'] = pd.to_datetime(albums_df['release_date_adj'], infer_datetime_format=True)

albums_df.head(3)

# + hidden=true
artist_all_albums = lineups\
                        .reset_index()\
                        .set_index('artist_name').loc[:, ['date', 'artist_id']]\
                        .join(albums_df.set_index('act'))\
                        .reset_index()

albums_before_show = artist_all_albums\
                        .loc[artist_all_albums['release_date_adj'].astype(str) < artist_all_albums['date']]

total_albuns = albums_before_show\
                    .groupby('artist_id')\
                    .agg({'release_date_adj':['min','max'], 'name':'count', 'total_tracks':'sum'})

total_albuns.columns = ['first_album_release', 'latest_album_release', 'total_albums', 'total_tracks']

studio_albuns = albums_before_show\
                    .loc[albums_before_show['album_version'] == 'studio']\
                    .groupby('artist_id')\
                    .agg({'release_date_adj':'max', 'name':'count', 'total_tracks':'sum'})

studio_albuns.columns = ['latest_studio_release', 'studio_albums', 'studio_tracks']

album_info = total_albuns.join(studio_albuns)
album_info.head()

# + [markdown] heading_collapsed=true
# ## Single info

# + hidden=true
artists_single_discography = []
for act, artists_array in artist_info.items():
    for artist_dict in artists_array:
        if 'single_infos' in artist_dict and artist_dict['single_infos'] != None :
            for single, singles_info in artist_dict['single_infos'].items():
                singles_dict = singles_info.copy()
                singles_dict['act'] = act
                singles_dict['artist'] = artist_dict['name']
                artists_single_discography.append(singles_dict)
                
single_df = pd.DataFrame(artists_single_discography)
single_df.loc[:, 'act'] = single_df['act'].str.strip()


# Retirando duplicatas
single_df = single_df.drop_duplicates(subset=['release_date', 'total_tracks', 'artist'])

# Coluna para versão especial de disco
single_df.loc[:, 'song_version'] = 'original'
single_df.loc[(single_df['name'].str.lower().str.contains('remix'))
                | (single_df['name'].str.lower().str.contains('version'))
                | (single_df['name'].str.lower().str.contains('versão'))
                | (single_df['name'].str.lower().str.contains('radio edit'))
                | (single_df['name'].str.lower().str.contains('acoustic'))
                | (single_df['name'].str.lower().str.contains('live'))
                | (single_df['name'].str.lower().str.match('\(\w*\s*mix\w*\s*\)'))
               , 'song_version'] = 'special'

single_df.loc[:, 'feat'] = 0
single_df.loc[(single_df['name'].str.lower().str.contains('feat\.'))
                | (single_df['name'].str.lower().str.contains('ft\.')),
             'feat'] = 1
  

single_df.loc[single_df['release_date'].str.len() == 10, 'release_date_adj'] = single_df['release_date']   
single_df.loc[single_df['release_date'].str.len() == 4, 'release_date_adj'] = single_df['release_date'] + '-01-01'  

single_df.loc[:, 'release_date_adj'] = pd.to_datetime(single_df['release_date_adj'], 
                                                      infer_datetime_format=True, errors='coerce')

single_df = single_df.loc[~single_df['release_date_adj'].isnull()]

# + hidden=true
artist_all_singles = lineups\
                .reset_index()\
                .set_index('artist_name').loc[:, ['date', 'artist_id']]\
                .join(single_df.set_index('act'))\
                .reset_index()

singles_before_show = artist_all_singles\
                        .loc[artist_all_singles['release_date_adj'].astype(str) < artist_all_singles['date']]

total_singles = singles_before_show\
                    .groupby('artist_id')\
                    .agg({'release_date_adj':['min','max'], 'total_tracks':'sum', 'feat':'sum'})

total_singles.columns = ['first_single_release', 'latest_single_release', 'all_singles', 'all_feats']

non_remix_singles = singles_before_show\
                    .loc[singles_before_show['song_version'] == 'original']\
                    .groupby('artist_id')\
                    .agg({'total_tracks':'sum', 'feat':'sum'})

non_remix_singles.columns = ['non_remix_singles', 'non_remix_feats']

single_info = total_singles.join(non_remix_singles)
single_info.head()

# + [markdown] heading_collapsed=true
# ## Discography Info

# + hidden=true
discography_info = lineups.index.to_frame().join(single_info).join(album_info).drop(columns='artist_id')

num_cols = [f for f in discography_info.columns.tolist() if 'release' not in f]

for c in num_cols:
    discography_info.loc[:, c] = discography_info[c].fillna(0)

discography_info.head(3)

# + hidden=true
lineups_dsc = lineups.join(discography_info)
# -

# ## Analysis

lineups_dsc.head(3)


def get_first_release(first_album, first_single):
    
    delta = (first_album - first_single).days/365
    
    if delta > 0 and delta < 20:
        return first_single
    else:
        return first_album


# +
lineups_dsc.loc[:, 'first_release'] = lineups_dsc.apply(lambda x: get_first_release(x['first_album_release'], x['first_single_release']), axis=1)
lineups_dsc.loc[:, 'dias_de_carreira'] = (lineups_dsc['datetime_start'] - lineups_dsc['first_release']).dt.days
lineups_dsc.loc[:, 'anos_de_carreira'] = np.round(lineups_dsc['dias_de_carreira']/365)

lineups_dsc.loc[:, 'tempo_desde_ultimo_album'] = (lineups_dsc['datetime_start'] - lineups_dsc['latest_album_release']).dt.days
lineups_dsc.loc[:, 'tempo_desde_ultimo_single'] = (lineups_dsc['datetime_start'] - lineups_dsc['latest_single_release']).dt.days


# +
# Balto, "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans Narrow", "Raleway"

# +
fig = px.histogram(lineups_dsc, x='anos_de_carreira', histnorm='percent')
fig.show()

fig = px.histogram(lineups_dsc, x='anos_de_carreira', histnorm='percent', cumulative=True)
fig.show()
# -

# - Historicamente, a maior parte dos artistas que participam do Lolla tem até 6 anos de carreira, sendo aqueles cuja carreira está no 4-5 a maior fatia individualmente

# +
px.histogram(lineups_dsc, x='tempo_desde_ultimo_album', cumulative=True, nbins=1000, histnorm='percent').show()

px.histogram(lineups_dsc, x='tempo_desde_ultimo_album', histnorm='percent', nbins=100).show()
# -

# - Mais da metade dos artistas tem menos de 1 ano de lançamento desde o último trabalho ao participarem do Lolla
# (O que faz sentido ao considerarmos seus períodos de Tour)

# +
# Anos de carreira x posição no Lineup
px.box(lineups_dsc, y='anos_de_carreira', x='importance_order', hover_name='artist', hover_data=['year']).show()

# Tempo desde o último álbum
px.box(lineups_dsc, y='tempo_desde_ultimo_album', x='importance_order', hover_name='artist', hover_data=['year']).show()



# +
# Recortes de nacionalidade e gênero
px.box(lineups_dsc, y='anos_de_carreira', x='importance_order', points='all', hover_name='artist', hover_data=['year'], color='is_br').show()

px.box(lineups_dsc, y='anos_de_carreira', x='importance_order', points='all', hover_name='artist', hover_data=['year'], color='female_presence').show()
# -

lineups_dsc['lastfm_genre_tags'].apply(pd.Series)

px.scatter(lineups_dsc, x='order_in_lineup', y='anos_de_carreira')

# +
# Tempo desde o último álbum - Posição no Lineup e Anos


px.box(lineups_dsc, y='tempo_desde_ultimo_album', x='year', hover_name='artist', hover_data=['year']).show()


px.density_heatmap(lineups_dsc, x='year', y='importance_order', z='tempo_desde_ultimo_album', nbinsx=9, 
                   histfunc='avg').show()
#lineups_dsc.head()
# -

# Os 5 artistas favoritos do Lolla
lineups_dsc['artist'].value_counts().to_frame().head(5)

lineups_dsc.head()


# + [markdown] heading_collapsed=true
# # Spotify Genres Try

# + hidden=true
def summarize_genres(artist):
    artist_genres = []
    if artist in artist_info:
        for a in artist_info[artist]:
            if a['genres']:
                artist_genres.extend(a['genres'])
            
        return artist_genres
        

lineups.loc[:,'genres'] = lineups['artist'].apply(lambda x: summarize_genres(x))


# + hidden=true
def is_brazilian(genres):
    if genres:
        return len([g for g in genres if 'brazil' in g or 'mpb' in g]) > 0
    else:
        return False
        

lineups.loc[:,'is_brazilian'] = lineups['genres'].apply(lambda x: is_brazilian(x))
lineups.loc[:,'no_genre_info'] = lineups['genres'].apply(lambda x: len(x) == 0 if x != None else False)

lineups.loc[:,'perry_stage'] = lineups['stage'].str.lower().str.contains('perry')
# + [markdown] hidden=true
# ## Null data

# + hidden=true
# Artistas sem correspondência
nan_artists = lineups.loc[lineups['genres'].isnull(), 'artist'].nunique()
print('Artistas sem correspondência: {}, {:.2f}% do total'.format(nan_artists, 100*nan_artists/lineups.shape[0]))

# + hidden=true
# Artistas sem informação de gênero musical, por ano
no_genre_artists = lineups.loc[lineups['no_genre_info']]
print('Artistas sem informação de gênero musical: {}'.format(no_genre_artists['artist'].nunique()))
print('{:.2f}% do total'.format(100*no_genre_artists['artist'].nunique()/lineups['artist'].nunique()))
    
display(lineups.loc[lineups['no_genre_info'], 'year'].value_counts().sort_index())

# + [markdown] hidden=true
# ## Genres

# + hidden=true
genre_counter = {}
for genre_array in lineups['genres'].tolist():
    if genre_array:
        for g in genre_array:
            if g in genre_counter:
                genre_counter[g] += 1
            else:
                genre_counter[g] = 1


genre_counter = {k: v for k, v in sorted(genre_counter.items(), key=lambda item: item[1], reverse=True)}

# + hidden=true
genre_counter

# + hidden=true
genre_components = {}
for g, v in genre_counter.items():
    comps = g.split()
    for c in comps:
        if c in genre_components:
            genre_components[c] += v
        else:
            genre_components[c] = v

genre_components = {k: v for k, v in sorted(genre_components.items(), key=lambda item: item[1], reverse=True)}
genre_components


# + hidden=true
def get_components_dict(genres, main_components):
    if genres:
        dict_components = {}
        for m in main_components:
            dict_components[m] = 0
            for g in genres:
                if m in g or max([similar(m, s) for s in g.split()]) > 0.5:
                    dict_components[m] += 1

        return {k: v for k, v in sorted(dict_components.items(), key=lambda item: item[1])}
    else:
        return None


# + hidden=true
def get_main_component(num_comps):
    
    if 'main_component' in lineups.columns.tolist():
        lineups.drop(columns=['main_component'], inplace=True)

    main_components = list(genre_components.keys())[:num_comps]

    lineups.loc[:, 'genre_components'] = lineups['genres'].apply(lambda x: get_components_dict(x, main_components))
    lineups.loc[:, 'main_component'] = lineups['genre_components'].apply(lambda x: max(x, key=x.get) if x != None else None).fillna('no_genre')


# + hidden=true
# Plotar variação do Desvião padrão para entender se mais componentes ajudam a espalhar a distribuição
tries = {}

for f in range(1, 31):
    if f%10 == 0:
        print(f)
    get_main_component(f)
    tries[f] = lineups.loc[~lineups['no_genre_info'], 'main_component'].value_counts(dropna=False).to_frame()['main_component'].std()
    
tries_df = pd.DataFrame.from_dict(tries, orient='index').reset_index().dropna()
tries_df.columns = ['n_main_components', 'stddev']

px.line(tries_df, x='n_main_components', y='stddev', ).show()

# + hidden=true
lineups.loc[lineups['artist'].str.contains('Eddie')]

# + hidden=true
get_main_component(10)
lineups['main_component'].value_counts()

components_df = lineups['genre_components'].apply(pd.Series)
components_df.loc[:, 'total'] = components_df.sum(axis=1)

components_df = components_df.div(components_df['total'], axis=0)

comp_columns = components_df.columns.tolist()
comp_columns.remove('total')


components_df = components_df\
                    .join(lineups)\
                    .groupby('year')\
                    .agg({f:'mean' for f in comp_columns})

components_df = pd.melt(components_df.reset_index(), id_vars='year', value_vars=comp_columns, value_name='component')

components_df.head()

# + hidden=true
api_key = open('.api_key').read()
api_key

# + hidden=true
px.line(components_df, x='year', y='component', color='variable', line_shape='spline')

# + hidden=true
lineup_

px.histogram(lineups, x= 'year', color='main_component', barmode='overlay').show()


# + hidden=true
def summarize_discography_till_date(date, artists_info_dict):
    for a in artists_info_dict:


# -

# # Analysis

px.box(lineups, x='year', y='duration_min').show()

lineups.head()

px.scatter(lineups, x='date', y='order_in_lineup', color='palco')

# ## Distribuição do lineup

px.histogram(lineups, x='year', y='duration_min', histfunc='avg').show()

lineup_per_year = lineups.groupby('year').agg({'duration_min':'mean', 'artist':'count', 'date':'nunique'})

# +
lineup_per_year.loc[:, 'acts_per_date'] = lineup_per_year['artist']/lineup_per_year['date']

lineup_per_year.head()
# -

import wikipedia

wi
