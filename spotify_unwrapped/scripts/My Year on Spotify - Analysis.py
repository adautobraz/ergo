# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup

# %%
import pandas as pd
import json 
import plotly.express as px
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


import plotly.io as pio
import umap
import sys

sys.path.append('./../../iFood/')

from sources.general_functions import facet_prettify, plot 

pd.set_option('max_columns', None)


# %%
def write(type, fig, name, facet=True, tickdefault=True):        
    if facet:
        facet_prettify(fig)
    fig.update_layout(
        template='plotly_white',
        font_family='Helvetica'
    )
    fig.update_yaxes(color='grey')
    fig.update_xaxes(color='grey')
    if type == 'insta':
        fig.update_layout(
            title=''
        )

        if tickdefault:
            fig.update_xaxes(tickfont_size=17, titlefont_size=19)
            fig.update_yaxes(tickfont_size=17, titlefont_size=19)
        try:
            fig.update_traces(textfont_size=14)
        except:
            print('No text')
    fig.show()
    pio.write_image(fig, f'images/{type}__{name}.png', format='png', scale=5)


# %% [markdown]
# # Load data

# %%
stream_raw_df = pd.read_csv('./data/prep/stream_history.csv')

stream_raw_df.loc[:, 'endTime'] = pd.to_datetime(stream_raw_df['endTime'], infer_datetime_format=True)
stream_raw_df['week'] = stream_raw_df['endTime'].dt.to_period('W').apply(lambda r: r.start_time)
stream_raw_df['month'] = stream_raw_df['endTime'].dt.to_period('M').apply(lambda r: r.start_time)
stream_raw_df['date'] = stream_raw_df['endTime'].dt.to_period('D').apply(lambda r: r.start_time)
stream_raw_df['hour'] = stream_raw_df['endTime'].dt.hour
stream_raw_df['dow'] = stream_raw_df['endTime'].dt.day
stream_raw_df['year'] = stream_raw_df['endTime'].dt.year

stream_raw_df.loc[:, 'sPlayed'] = stream_raw_df['msPlayed']/(1000)
stream_raw_df.loc[:, 'hPlayed'] = stream_raw_df['sPlayed']/(60*60)

stream_raw_df.loc[:, 'skip'] = stream_raw_df['sPlayed'] < 30

stream_df = stream_raw_df\
            .loc[(stream_raw_df['year'] == 2021) 
                & (~stream_raw_df['skip'])
                & (~stream_raw_df['id'].isnull())
                ]

stream_df.head()

# %%
df = stream_df.copy()

feature_cols = df.iloc[:, 20:31].columns.tolist()
feature_cols.remove('mode')
feature_cols.remove('key')

df_songs = stream_df.loc[:, ['id'] + feature_cols].drop_duplicates()
scaler = StandardScaler()
song_features_df = pd.DataFrame(scaler.fit_transform(df_songs.loc[:, feature_cols]))
song_features_df.columns = feature_cols
song_features_df.index = df_songs.index
song_features_df.loc[:, 'id'] = df_songs['id']
song_features_df = song_features_df.set_index('id')
song_features_df.head()

# %%
df = stream_df.sort_values(by='endTime')

# df.loc[:, 'hour_round'] = np.round(df['hour'] + df['endTime'].dt.minute/(60))
# df.loc[:, 'hour'] = df['hour_round']

op_dict = {f:'sum' for f in feature_cols}
op_dict['hPlayed'] = 'sum'
op_dict['id'] = 'count'

df = df.groupby(['date', 'hour'], as_index=False).agg(op_dict)

df.loc[:, 'dow'] = df['date'].dt.dayofweek
df.loc[:, 'dom'] = df['date'].dt.day
df.loc[:, 'day'] = df['date'].dt.strftime('%a')

df = df.sort_values(by=['dow', 'hour'], ascending=False)

cols = df['hour'].unique().tolist()
df = pd.pivot_table(df, index=['date', 'dow', 'day', 'dom'], columns='hour',
                    values='hPlayed', aggfunc='sum')\
        .fillna(0).reset_index()
hour_df = pd.melt(df, id_vars=['date', 'dow', 'day', 'dom'], value_vars=cols)

hour_df['week'] = hour_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
hour_df['month'] = hour_df['date'].dt.to_period('M').apply(lambda r: r.start_time)

hour_df.head()

# %%
song_info_df = stream_df\
                .groupby(['id', 'artistName', 'trackName', 'release_year', 'duration_ms'], as_index=False)\
                .agg({'endTime':'count', 'hPlayed':'sum'})

song_info_df.loc[:, 'info'] = song_info_df.apply(lambda x: "{} - {}".format(x['trackName'], x['artistName']), axis=1)

# %% [markdown]
# # Analysis

# %%
spotify_colors = ['#1DB954', '#191414']
heat_pal = px.colors.sequential.YlGn

# %% [markdown]
# ## Heatmap view

# %% code_folding=[0]
# Data Prep

df = hour_df.groupby(['date', 'dow', 'day'], as_index=False).agg({'value':'sum'})
df['week_dt'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
df['month_dt'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)

df.loc[:, 'next'] = df['week_dt'] + pd.Timedelta(7, 'day')

df.loc[:, 'dom'] = df['date'].dt.day
df.loc[:, 'next_dom'] = df['dow'] + 1

df.loc[:, 'next_dow'] = df['dow'] + 1
df.loc[:, 'delta'] = df['next_dow'] - df['dow']

df.loc[:, 'week'] = df['week_dt'].dt.strftime('%d %b %Y')
df.loc[:, 'date_str'] = df['date'].dt.strftime('%d% %b %Y')

hover_dict = {'week_dt':False, 'next':False, 'week':True, 'value':':.3g'}

# %% code_folding=[0]
# Week x Dow x Timeline 
fig = px.timeline(df.sort_values(by='dom', ascending=False), x_start='dow', x_end='next_dow', y='week_dt', 
                  hover_name='date_str', color='value', hover_data=hover_dict, 
                  color_continuous_scale='YlGn'
                 )

fig.update_yaxes(
    autorange='reversed',
    dtick='M1',
    tickformat='%b/%y'
)
# fig.update_xaxes(dtick=5, range=[0.5, 32])

fig.layout.xaxis.type = 'linear'
fig.data[0].x = df['delta'].tolist()


fig.update_traces(marker_line_color='lightgrey', marker_line_width=0.5)

fig.update_xaxes(tickvals=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], #tickangle=45,
                 showgrid=False,
                 ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>:<i> When did I use Spotify?</i>',
    margin_t=80,
    yaxis_title='',
    xaxis_title='',
    coloraxis_colorbar_title='Listened<br>hours',
    coloraxis_colorbar_ticksuffix='h'
)

# fig.update_layout(coloraxis_colorbar_tickfont_size=18, height=600, width=600)
# write('insta', fig, 'heatmap')

fig.update_layout(coloraxis_colorbar_tickfont_size=12, width=500)
write('medium', fig, 'heatmap')

# %% code_folding=[0]
# # Dow x Semana x Timeline 

# fig = px.timeline(df.sort_values(by='dow', ascending=False), x_start='week_dt', x_end='next', y='day', 
#                   hover_name='date_str', color='value', 
#                   hover_data=hover_dict,
#                   color_continuous_scale=heat_pal
#                  )
# # fig.update_traces(insidetextanchor='middle')

# fig.update_layout(
#     title='When did I use Spotify?',
#     yaxis_title='Day of Week',
#     coloraxis_colorbar_title='Listened<br>hours'
# )
# plot(fig)

# %% code_folding=[0]
# # DoW x Week x Heatmap 
# fig = px.density_heatmap(df.sort_values(by='dow', ascending=False), x='week_dt', y='day', z='value', 
#                          nbinsx=100, histfunc='sum', color_continuous_scale=heat_pal, hover_name='date')

# # (df, x_start='week', x_end='next', y='day', hover_name='week')
# plot(fig)

# %% code_folding=[0]
# # NO - Month x Dom x Timeline 
# fig = px.timeline(df.sort_values(by='dom', ascending=False), x_start='dom', x_end='next_dom', y='month_dt', 
#                   hover_name='date_str', color='value', hover_data=['week'], 
#                   color_continuous_scale=heat_pal
#                  )

# fig.update_yaxes(autorange='reversed')
# fig.update_xaxes(dtick=5, range=[0.5, 32])

# fig.layout.xaxis.type = 'linear'
# fig.data[0].x = df['delta'].tolist()

# fig.update_layout(
#     title='When did I use Spotify?',
#     yaxis_title='Month',
#     coloraxis_colorbar_title='Listened<br>hours'
# )

# plot(fig)

# %% [markdown]
# ## Usage over time

# %% code_folding=[0]
# df = stream_df.sort_values(by='endTime')

# fig = px.histogram(df, x='endTime')
# fig = px.histogram(df, x='endTime', y='hPlayed', histfunc='sum')

# # fig.update_traces(side='positive', spanmode='hard')
# plot(fig)

# %% code_folding=[0]
# Weekly usage
df = hour_df.sort_values(by='date')

df = df.groupby(['month'], as_index=False).agg({'value':'sum', 'date':'nunique'})

df.loc[:, 'value'] = 100*df['value']/(24)
df.loc[:, 'value'] = df['value']/df['date']

df.loc[:, 'rank_min'] = df['value'].rank(ascending=True)
df.loc[:, 'rank_max'] = df['value'].rank(ascending=False)
df.loc[:, 'rank_abs'] = df.apply(lambda x: min([x['rank_min'], x['rank_max']]), axis=1)

df.loc[:, 'text'] = ''
df.loc[df['rank_abs'] <= 1, 'text'] = df['month'].dt.strftime('%b')

fig = px.line(df, x='month', y='value', line_shape='spline', text='text',
              color_discrete_sequence=spotify_colors)
fig.update_traces(mode='lines+markers+text', textposition='top center', line_width=2.5)
fig.update_yaxes(rangemode='tozero', ticksuffix='%')
fig.update_xaxes(tickformat='%b/%y', nticks=5)
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i> How much time did I spend on Spotify each month?',
    yaxis_title='% of total month time listening to music',
    xaxis_title=''
)
# plot(fig)

# fig.update_layout(width=800, height=600)
# write('insta', fig, 'weekly')

# fig.update_layout(coloraxis_colorbar_tickfont_size=12, width=500)
write('medium', fig, 'weekly')

# %% code_folding=[0]
# Dow x Hours
df = hour_df\
        .groupby(['date', 'dow', 'day'], as_index=False)\
        .agg({'value':'sum'})

df = df\
        .groupby(['dow', 'day'], as_index=False)\
        .agg({'value':'mean'})\
        .sort_values(by='dow')

# df.loc[:, 'dow'] = df['date'].dt.dayofweek
# df.loc[:, 'day'] = df['date'].dt.day_name()


fig = px.line(df, x='day', y='value', color_discrete_sequence=spotify_colors, line_shape='spline')
fig.update_traces(mode='lines+markers+text', texttemplate='%{y:.1f}h', textposition='top center')
fig.update_yaxes(rangemode='tozero', ticksuffix='h', nticks=2)

fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>When in the week do I listen to music the most?</i>',
    yaxis_title='Average hours listened per day',
    xaxis_title='Day of the week'
)

# fig.update_layout(width=650, height=600)
# write('insta', fig, 'dow')

# fig.update_layout(width=650, height=600)
write('medium', fig, 'dow')

# plot(fig)

# %% [markdown]
# ## Hour of day

# %% code_folding=[0]
# # Time
# df = hour_df.copy()
# df.loc[:, 'hour_disc'] = np.floor(df['hour']/1)*1

# df = df\
#         .groupby(['hour_disc'], as_index=False)\
#         .agg({'value':'mean'})\
#         .sort_values(by=['hour_disc'])

# df.loc[:, 'hour_percentage'] = 100*df['value']
# df.loc[:, 'text'] = ''
# df.loc[df['hour_disc'].isin([1, 14, 15, 18, 21]), 'text'] = df['hour_disc'].apply(lambda x: "{:.0f}h".format(x))

# fig = px.line(df, x='hour_disc', y='hour_percentage', line_shape='spline', text='text',
#               color_discrete_sequence=spotify_colors)

# fig.update_traces(mode='lines+markers+text', textposition='top center')
# fig.update_yaxes(rangemode='tozero', ticksuffix='%')
# fig.update_xaxes(title='', ticksuffix='h', tickangle=45, dtick=6)
# fig.update_layout(
#     yaxis_title='% of time (in hours)',
#     title='<b>Spotify (un)Wrapped</b>: <i>What time did I listen to music?</i>')
# plot(fig)

# # fig = px.violin(df, y='day', x='hour', points=False, orientation='h', color_discrete_sequence=['grey'])
# # fig.update_traces(meanline_visible=True, side='positive', width=3, spanmode='hard')
# # # fig.update_yaxes(autorange='reversed')
# # plot(fig)


# %% code_folding=[]
# Weekend x Weekend 
df = hour_df.copy()
df.loc[:, 'hour_disc'] = np.floor(df['hour']/1)*1

df.loc[:, 'week_period'] = df['dow'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekdays')

df = df\
        .groupby(['hour_disc', 'week_period'], as_index=False)\
        .agg({'value':'mean'})\
        .sort_values(by=['week_period', 'hour_disc'])

df.loc[:, 'hour_percentage'] = df['value']*60
df.loc[:, 'text'] = ''
df.loc[(df['week_period'] == 'Weekdays') & (df['hour_disc'].isin([1, 14, 18, 21])), 'text'] = df['hour_disc'].apply(lambda x: "{:.0f}h".format(x))
df.loc[(df['week_period'] == 'Weekend') & (df['hour_disc'].isin([16, 21])), 'text'] = df['hour_disc'].apply(lambda x: "{:.0f}h".format(x))


fig = px.line(df, x='hour_disc', y='hour_percentage', color='week_period', text='text',
              line_shape='spline', color_discrete_sequence=spotify_colors)
fig.update_traces(mode='lines+markers+text', textposition='top center', line_width=2.5)
fig.update_yaxes(rangemode='tozero', ticksuffix='min', nticks=5)
fig.update_xaxes(title='', ticksuffix='h', tickangle=45)
fig.update_layout(
#     legend_orientation='h',
    title='<b>Spotify (un)Wrapped</b>: <i>At what time did I listen to music?</i>',
    legend_title='Week period',
    legend_orientation='h',
    yaxis_title='Average time listening to music (min)',
)

# plot(fig)

# fig.update_layout(
#     legend_x=-0.2,
#     legend_font_size=14,
#     width=650,
#     height=600
# )
# write('insta', fig, 'hour')

# fig.update_layout(
#     legend_font_size=14,
#     width=650,
#     height=600
# )
write('medium', fig, 'hour')

# fig = px.violin(df, y='day', x='hour', points=False, orientation='h', color_discrete_sequence=['grey'])
# fig.update_traces(meanline_visible=True, side='positive', width=3, spanmode='hard')
# # fig.update_yaxes(autorange='reversed')
# plot(fig)


# %% [markdown]
# ## All tracks view

# %% code_folding=[0]
# Data Prep
df = song_features_df.copy().dropna()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(df)

df_embed = pd.DataFrame(embedding)
df_embed.index = df.index
df_embed.columns = ['dim0', 'dim1']
df_embed.reset_index(inplace=True)

# %%
df = song_features_df.stack().to_frame().reset_index()
df.columns = ['id', 'feature', 'value']
df.loc[:, 'ranked_value'] = 100*df.groupby(['feature'])['value'].rank(pct=True)
df.loc[:, 'features_rank_position'] = df.apply(lambda x: "<i>{}</i>: {:.1f}%".format(x['feature'], x['ranked_value']), axis=1)
df_feat_info = df.groupby(['id'], as_index=False).agg({'features_rank_position':'<br>'.join})

df_feat_info.loc[:, 'features_rank_position'] = '<br>' + df_feat_info['features_rank_position']

# %%
song_features_df.shape

# %% code_folding=[]
# View
df = df_embed.copy()

df = pd.merge(left=df, right=song_info_df)
df.loc[:, 'rank'] = df['endTime'].rank(ascending=False, method='first')
df.loc[:, 'top_ranking'] = 'Others'
df.loc[df['rank'].between(4, 10), 'top_ranking'] = df.apply(lambda x: '<b>{:.0f}th</b> - {}, <i>{}</i>'.format(x['rank'], x['trackName'], x['artistName']), axis=1)
df.loc[df['rank'] == 1, 'top_ranking'] = df.apply(lambda x: '<b>{:.0f}st</b> - {}, <i>{}</i>'.format(x['rank'], x['trackName'], x['artistName']), axis=1)
df.loc[df['rank'] == 2, 'top_ranking'] = df.apply(lambda x: '<b>{:.0f}nd</b> - {}, <i>{}</i>'.format(x['rank'], x['trackName'], x['artistName']), axis=1)
df.loc[df['rank'] == 3, 'top_ranking'] = df.apply(lambda x: '<b>{:.0f}rd</b> - {}, <i>{}</i>'.format(x['rank'], x['trackName'], x['artistName']), axis=1)
df.loc[(df['rank'] > 10) & (df['rank'] <= 100), 'top_ranking'] = 'Top 100'

df.loc[:, 'text'] = ''
df.loc[df['rank'] <= 10, 'text'] = df['rank'].apply(lambda x: '{:.0f}'.format(x))

df = pd.merge(left=df, right=df_feat_info, on='id')

top_order = df.sort_values(by='rank')['top_ranking'].unique().tolist()

fig = px.scatter(df.sort_values(by='rank', ascending=False), x='dim0', y='dim1', hover_name='info', size='endTime', text='text',
                 color='top_ranking', hover_data=['features_rank_position'], size_max=30,
                 color_discrete_map={'Others':'lightgrey'},#, 'Top 100':spotify_colors[0]},
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 opacity=0.8
#                  category_orders={'top_ranking': top_order}
                )

fig.update_traces(mode='markers+text', textfont_size=10, textfont_color='black')#marker_line_color='grey', marker_line_width=0.2)
fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showticklabels=False)
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>What were all the songs I played?<i>',
    yaxis_title='1st projected dimension',
    xaxis_title='2nd projected dimension',
    legend_title=''
)

# plot(fig)


# fig.update_layout(
#     legend_font_size=14,
#     legend_orientation='h',
#     legend_y=-0.15,
#     legend_x=-0.15,
#     width=650,
#     height=600
# )
# write('insta', fig, 'all_songs')

write('medium', fig, 'all_songs')

# %% [markdown]
# ## AOTY
# Álbum mais ouvido no Ano

# %% code_folding=[]
# AOTY 
df = stream_df.copy()

df.loc[:, 'album_adj'] = df['album_name']
df.loc[df['album_type'] == 'single', 'album_adj'] = 'Single'

df.loc[:, 'artist_adj'] = df['artist_name']
df.loc[df['album_type'] == 'single', 'artist_adj'] = 'NA'

df.loc[:, 'total_time'] = df['hPlayed'].sum()

df = df.loc[df['album_type'] == 'album']\
        .groupby(['album_adj', 'artist_adj', 'image_url', 'total_time', 'release_year'], as_index=False)\
        .agg({'hPlayed':'sum', 'id':'count', 'uri':'nunique'})

df.loc[:, 'share'] = 100*df['hPlayed']/df['total_time']

df.loc[:, 'rank'] = df['share'].rank(ascending=False)
df.loc[: , 'size'] = 1
top = df.loc[df['rank'] <= 10].copy()

top.loc[:, 'text'] = top.apply(lambda x: "<b>{:.0f} - {}</b><br><i>{}, {:.0f}</i> - {:.1f}h".format(x['rank'], x['album_adj'], x['artist_adj'], x['release_year'],  x['hPlayed']), axis=1)

fig = px.scatter(top, x='hPlayed', y='rank', hover_name='album_adj', size_max=20,
                 hover_data=['artist_adj'], size='size')

ticktext = top.sort_values(by='rank')['text'].tolist()
tickvals = top.sort_values(by='rank')['rank'].tolist()

fig.update_yaxes(autorange='reversed', tickmode='array', tickvals=tickvals, 
                 ticktext=ticktext, zeroline=False, tickfont_size=14)
fig.update_xaxes(rangemode='tozero', range=[-0.5, 30], tickfont_size=14)

# fig.update_traces(mode='markers+text', textposition='top right')

fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>What were my albums of the year?</i>',
    xaxis_title='Hours Listened',
    yaxis_title='Top 10 most listened albums'
)

multiplier = 2

for i, row in top.iterrows():
    fig.update_traces(marker_color="rgba(0,0,0,0)")
    image = row['image_url']
    fig.add_layout_image(
        dict(
            source=f'{image}',
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            x=row["hPlayed"],
            y=row["rank"],
            sizex=multiplier, 
            sizey=multiplier, 
#             sizey=np.sqrt(row["uri"] / top["uri"].max()) * multiplier,
#             sizex=np.sqrt(row["uri"] / top["uri"].max()) * multiplier,
            sizing="contain",
            opacity=0.8,
            layer="above"
        )
    )
    
write('medium', fig, 'aoty', True, False)
    
# fig.update_layout(
#     yaxis_titlefont_size=18,
#     xaxis_titlefont_size=18,
#     height=600,
#     width=650
# )
# write('insta', fig, 'aoty', True, False)

# %% [markdown]
# ## Release Date View 

# %% code_folding=[]
# New flame or Nostalgia?
df = stream_df.copy()

df.loc[:, 'decade'] = np.floor(df['release_year']/10)*10
df.loc[:, 'decade_cat'] = df['decade'].apply(lambda x: "{:.0f}-{:.0f}".format(x, min([2021, x+9])))
# df.loc[df['release_year'] >= 2020, 'decade_cat'] = df['release_year'].apply(lambda x: "{:.0f}".format(x))
# df.loc[df['release_year'] == 2021, 'decade_cat'] = '2021'


df = df\
        .loc[(df['release_year'] <= 2021)]\
        .groupby(['decade_cat', 'artistName'], as_index=False)\
        .agg({'hPlayed':'sum'})

df.loc[: , 'rank'] = df.groupby(['decade_cat'])['hPlayed'].rank(ascending=False, method='dense')
df.loc[:, 'total_hours'] = df.groupby(['decade_cat'])['hPlayed'].transform('sum')

top = df.loc[(df['rank'] == 1)].copy()

top.loc[:, 'decade_share'] = 100*top['total_hours']/top['total_hours'].sum()
top.loc[:, 'artist_share'] = 100*top['hPlayed']/top['total_hours']

fig = px.bar(top, y='decade_cat', x='decade_share', orientation='h', hover_data=['artistName', 'artist_share'],
             color_discrete_sequence=spotify_colors
            )
fig.update_xaxes(ticksuffix='%', range=[0, 60])
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>What is my musical decade?',
    yaxis_title='Song\'s decade release',
    xaxis_title='% of total listened time'
)

write('medium', fig, 'decade', True, True)


# fig.update_layout(
#    width=650,
#     height=600
# )
# write('insta', fig, 'decade', True, True)


# plot(fig)

# %% code_folding=[0]
# df = stream_df.copy()

# df = df.groupby(['release_year', 'artistName'], as_index=False).agg({'hPlayed':'sum'})

# df.loc[: , 'rank'] = df.groupby(['release_year'])['hPlayed'].rank(ascending=False, method='dense')
# df.loc[:, 'total_hours'] = df.groupby(['release_year'])['hPlayed'].transform('sum')

# top = df.loc[df['rank'] == 1].copy()

# top.loc[:, 'year_share'] = 100*top['total_hours']/top['total_hours'].sum()
# top.loc[:, 'artist_share'] = 100*top['hPlayed']/top['total_hours']

# fig = px.bar(top, x='release_year', y='year_share', hover_data=['artistName', 'artist_share'])
# # fig.update_yaxes(dtick=10)
# # fig.update_traces(texttemplate='%{x:.1f}%')
# plot(fig)

# %% [markdown]
# ## Seasons

# %% code_folding=[0]
# Song of the Season
df = stream_df.copy()

df.loc[:, 'date_month'] = df['endTime'].dt.strftime('%m-%d')
df.loc[(df['date_month'] >= '12-21') | (df['date_month'] <= '03-19'), 'season'] = 'Summer'
df.loc[df['date_month'].between('03-21', '06-21'), 'season'] = 'Fall'
df.loc[df['date_month'].between('06-22', '09-22'), 'season'] = 'Winter'
df.loc[df['date_month'].between('09-23', '12-20'), 'season'] = 'Spring'

df = df.groupby(['season', 'artistName', 'trackName'], as_index=False).agg({'hPlayed':'sum', 'id':'count'})

# df.loc[:, 'total_hours'] = df.groupby(['season'])['hPlayed'].transform('sum')
df.loc[:, 'info'] = df.apply(lambda x: "{} - {}".format(x['trackName'], x['artistName']), axis=1)

df.loc[: , 'rank'] = df.groupby(['season'])['hPlayed'].rank(ascending=False, method='dense')

top = df.loc[df['rank'] == 1].copy()

fig = px.bar(top, y='season', x='hPlayed', text='info', 
             category_orders={'season':['Summer', 'Fall', 'Winter', 'Spring']})
plot(fig)

# %% [markdown]
# ## Song of the month

# %% code_folding=[0]
# Song of the Season
df = stream_df.copy()

df_track = df.groupby(['month', 'trackName', 'artistName'], as_index=False).agg({'hPlayed':'sum', 'id':'count'})
df_track.loc[:, 'category'] = 'Top Track'

df_artist = df.groupby(['month', 'artistName'], as_index=False).agg({'hPlayed':'sum', 'id':'count'})
df_artist.loc[:, 'trackName'] = '' 
df_artist.loc[:, 'category'] = 'Top Artist'

df = pd.concat([df_track, df_artist]).reset_index().iloc[:, 1:]

df.loc[: , 'rank'] = df.groupby(['category', 'month'])['id'].rank(ascending=False, method='first')

df.loc[:, 'total_songs'] = df.groupby(['category', 'month'])['id'].transform('sum')

df.loc[df['category'].str.contains('Track'), 'info'] = df.apply(lambda x: "{} - {}".format(x['trackName'], x['artistName']) , axis=1)
df.loc[df['category'].str.contains('Artist'), 'info'] = df['artistName']

df.loc[df['category'] == 'Top Artist', 'id'] = 100*df['id']/df['total_songs']

top = df.loc[df['rank'] <= 1].copy()

top

fig = px.bar(top, y='month', x='id', facet_col='category', text='info', facet_col_spacing=0.1,
                color_discrete_sequence=spotify_colors)

fig.update_yaxes(range=['2021-12-31', '2020-12-01'], tickformat='%b/%y')
fig.update_xaxes(matches=None, nticks=5)
fig.update_traces(textposition='outside')
fig.update_xaxes(col=1, title='# Streams on month', range=[0,200])
fig.update_xaxes(col=2, title='% Streams on month', ticksuffix='%', range=[0, 30])

fig.for_each_annotation(lambda a: a.update(text='<b>{}</b>'.format(a.text)))

# fig.update_traces(textposition='inside')
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>What was the top song and artist of each month?</i>',
    yaxis_title=''
)
# plot(fig)

fig.update_layout(margin_t=120, width=900)
fig.update_traces(textfont_size=14)
fig.for_each_annotation(lambda a: a.update(font_size=14))
write('medium', fig, 'top_month')


# fig.update_layout(width=650,height=600)
# write('insta', fig, 'top_month')


# %% [markdown]
# ## Mood swings
# Quando você foi mais triste? E mais feliz?

# %% [markdown]
# ### Per month

# %% code_folding=[0]
# Data prep 
df_top = stream_df\
                .groupby(['month', 'id', 'artistName', 'trackName'], as_index=False)\
                .agg({'hPlayed':'sum', 'endTime':'count'})

df_top.loc[:, 'total_month_streams'] = df_top.groupby(['month'])['endTime'].transform('count')

df_top.loc[:, 'total_artist_time'] = df_top.groupby(['month', 'artistName'])['hPlayed'].transform('sum')
df_top.loc[:, 'total_artist_streams'] = df_top.groupby(['month', 'artistName'])['endTime'].transform('sum')
df_top.loc[:, 'rank_song'] = df_top.sort_values(by='hPlayed', ascending=False).groupby(['month'])['endTime'].rank(ascending=False, method='first')
df_top.loc[:, 'rank_artist'] = df_top.groupby(['month'])['total_artist_streams'].rank(ascending=False, method='dense')

df_top.loc[:, 'song_share'] = 100*df_top['endTime']/df_top['total_month_streams']
df_top.loc[:, 'artist_share'] = 100*df_top['total_artist_streams']/df_top['total_month_streams']

df_top.loc[:, 'top_songs'] = ''
df_top.loc[:, 'top_artists'] = ''

df_top.loc[df_top['rank_song'] <= 5, 'top_songs'] = df_top.apply(lambda x: "\t<i>Top {:.0f}</i>: {} - {} ({:.1f}%)".format(x['rank_song'], x['trackName'], x['artistName'], x['song_share']), axis=1)
df_top.loc[df_top['rank_artist'] <= 5, 'top_artists'] = df_top.apply(lambda x: "\t<i>Top {:.0f}</i>: {} ({:.1f}%)".format(x['rank_artist'], x['artistName'], x['artist_share']), axis=1)

df_top_info = df_top\
                .loc[(df_top['rank_song'] <= 5) | (df_top['rank_artist'] <= 5)]\
                .groupby(['month'], as_index=False)\
                .agg({'top_songs': 'unique', 'top_artists':'unique'})\
                .sort_values(by='month')

df_top_info.loc[:, 'top_songs'] = df_top_info['top_songs'].apply(lambda x: '<br>' + '<br>'.join(sorted([i for i in x if i])))
df_top_info.loc[:, 'top_artists'] = df_top_info['top_artists'].apply(lambda x: '<br>' + '<br>'.join(sorted([i for i in x if i])))

# %% code_folding=[0]
# Mood swings 

df = stream_df\
        .groupby(['month'], as_index=False)\
        .agg({'hPlayed':'sum', 'danceability':'median', 'energy':'median', 'valence':'median'})

scaler = StandardScaler()

df_features = df.iloc[:, 2:]
cols = df_features.columns.tolist()

df_features = pd.DataFrame(scaler.fit_transform(df_features))
df_features.columns = cols

df.iloc[:, 2:] = df_features

df = pd.melt(df, id_vars=['month', 'hPlayed'], value_vars=cols)

df.loc[:, 'rank_max'] = df.groupby(['variable'])['value'].rank(ascending=False, method='first')
df.loc[:, 'rank_min'] = df.groupby(['variable'])['value'].rank(ascending=True, method='first')
df.loc[:, 'rank'] = df.apply(lambda x: min([x['rank_min'], x['rank_max']]), axis=1)

df.loc[:, 'text'] = ''
df.loc[df['rank'] <= 1, 'text'] = df['month'].dt.strftime('%b')

df = pd.merge(left=df, right=df_top_info, on=['month'])

# %% code_folding=[0]
# Insta
fig = px.line(df, x='month', y='value', facet_row='variable', facet_row_spacing=0.08, text='text', 
              color_discrete_sequence=spotify_colors, line_shape='spline', hover_data=['top_songs', 'top_artists'],
              category_orders={'variable':['danceability', 'energy', 'valence']}
             )

fig.update_yaxes(showticklabels=True, tickvals=[-2,0, 2],  
                 tickmode='array', title='', range=[-3, 3], tickfont_size=14)
fig.update_yaxes(row=3, ticktext=['Less<br>dancing', 'Year avg', 'More<br>dancing'])
fig.update_yaxes(row=2, ticktext=['Chill', 'Year avg', 'Frenzy'])
fig.update_yaxes(row=1, ticktext=['Downer','Year avg', 'Upbeat'])

fig.update_traces(mode='lines+markers+text', textposition='middle right', 
                  textfont_size=12, line_width=2.5
                 )

fig.update_xaxes(title='', tickvals=['2021-01-01', '2021-07-01', '2021-12-01'],
                 tickformat='%b/%y', range=['2020-12-15', '2022-01-15'], tickfont_size=16
                )
fig.update_layout(
    title='<b>Spotify (un)Wrapped: </b><i>How was my musical emotional state during 2021?</i>',
    margin_t=120,
    height=600,
    width=650,
    showlegend=False
)
fig.for_each_annotation(lambda a: a.update(text="<b>{}</b>".format(a.text.split('=')[1].title())))
# plot(fig)
write('insta', fig,'moods', True, False)


# %% code_folding=[0]
# Medium
fig = px.line(df, x='month', y='value', facet_col='variable', facet_col_spacing=0.08, text='text', 
              color_discrete_sequence=spotify_colors, line_shape='spline', hover_data=['top_songs', 'top_artists'],
              category_orders={'variable':['danceability', 'energy', 'valence']}
             )

fig.update_yaxes(showticklabels=True, tickvals=[-2,0, 2],  
                 tickmode='array', title='', range=[-3, 3], tickfont_size=14)
fig.update_yaxes(col=1, ticktext=['Less<br>dancing', 'Year avg', 'More<br>dancing'])
fig.update_yaxes(col=2, ticktext=['Chill', ' ', 'Frenzy'])
fig.update_yaxes(col=3, ticktext=['Downer',' ', 'Upbeat'])

fig.update_traces(mode='lines+markers+text', textposition='middle right', 
                  textfont_size=12, line_width=2.5
                 )

fig.update_xaxes(title='', tickvals=['2021-01-01', '2021-07-01', '2021-12-01'],
                 tickformat='%b/%y', range=['2020-12-15', '2022-01-15']
                )
fig.update_layout(
    title='<b>Spotify (un)Wrapped: </b><i>How was my musical emotional state during 2021?</i>',
    margin_t=120,
    showlegend=False
)
fig.for_each_annotation(lambda a: a.update(text="<b>{}</b>".format(a.text.split('=')[1].title())))
# plot(fig)
write('medium', fig,'moods', True, False)


# %% [markdown]
# ### Per hour

# %% code_folding=[0]
# Mood per hour of day
df = stream_df.sort_values(by='endTime')

n = 4

df.loc[:, 'hour_disc'] = np.floor(df['hour']/n)*n

df.loc[:, 'week_period'] = 'Weekdays'
df.loc[df['dow'] >= 5, 'week_period'] = 'Weekend'

op_dict = {f:'median' for f in feature_cols}

df = df.groupby(['week_period', 'hour_disc']).agg(op_dict)

df = df.stack().to_frame().reset_index()
df.columns = ['week_period', 'hour', 'feature', 'value']

df.loc[:, 'period'] = df['hour'].apply(lambda x: "{:.0f}h-{:.0f}h".format(x, x+n-1))
df.loc[:, 'avg'] = df.groupby(['feature'])['value'].transform('mean')
df.loc[:, 'max'] = df.groupby(['feature'])['value'].transform('max')
df.loc[:, 'min'] = df.groupby(['feature'])['value'].transform('min')
df.loc[:, 'norm_value'] = (df['value'] - df['avg'])/(df['max'] - df['min'])
df.loc[:, 'norm_value_abs'] = df['norm_value'].abs()

# df.loc[:, 'rank'] = df.groupby(['week_period''feature'])['norm_value'].rank(ascending=False)
# df.loc[:, 'rank'] = df.groupby(['feature'])['norm_value'].rank(ascending=False)

df = df.loc[df['feature'].isin(['danceability', 'energy', 'valence'])]
df.loc[:, 'rank_max'] = df.groupby(['feature'])['norm_value'].rank(ascending=False)
df.loc[:, 'rank_min'] = df.groupby(['feature'])['norm_value'].rank(ascending=True)

x = 1

df.loc[:, 'text'] = ''
df.loc[(df['rank_max'] <= x) | (df['rank_min'] <= x), 'text'] = df['period']

# %% code_folding=[0]
# Insta 
fig = px.line(df, x='hour', y='norm_value', facet_row='feature', text='text',
              line_shape='spline', color='week_period', hover_data=['period'],
              color_discrete_sequence=spotify_colors, facet_row_spacing=0.05,
              category_orders={'feature':['danceability', 'energy', 'valence']})

fig.update_traces(mode='lines+markers+text', textposition='top center', line_width=2.5)

fig.update_xaxes(title='', tickangle=45, nticks=5, ticksuffix='h', tickfont_size=16)

fig.update_yaxes(showticklabels=True, tickvals=[-0.8,0, 0.8],  
                 tickmode='array', title='', range=[-1, 1], tickfont_size=16)

fig.update_yaxes(row=3, ticktext=['Less<br>dancing', 'Day avg', 'More<br>dancing'])
fig.update_yaxes(row=2, ticktext=['Chill', 'Day avg', 'Frenzy'])
fig.update_yaxes(row=1, ticktext=['Downer','Day avg', 'Upbeat'])

fig.for_each_annotation(lambda a: a.update(text = "<b>{}</b>".format(a.text.split('=')[1].title())))

fig.update_layout(
    legend_orientation='h',
    width=700,
    height=600,
    legend_title='Week period',
    legend_x=-0.14,
    legend_y=-0.08,
    legend_font_size=14
)

# fig.show()
# plot(fig)
write('insta', fig, 'mood_hour', True, False)

# %% code_folding=[]
# Medium
fig = px.line(df, x='hour', y='norm_value', facet_col='feature', text='text',
              line_shape='spline', color='week_period', hover_data=['period'],
              color_discrete_sequence=spotify_colors, facet_col_spacing=0.05,
              category_orders={'feature':['danceability', 'energy', 'valence']})

fig.update_traces(mode='lines+markers+text', textposition='middle left', line_width=2.5)

fig.update_xaxes(title='', nticks=5, ticksuffix='h')

fig.update_yaxes(showticklabels=True, tickvals=[-0.8,0, 0.8],  
                 tickmode='array', title='', range=[-1, 1])

fig.update_yaxes(col=1, ticktext=['Less<br>dancing', 'Day avg', 'More<br>dancing'])
fig.update_yaxes(col=2, ticktext=['Chill', ' ', 'Frenzy'])
fig.update_yaxes(col=3, ticktext=['Downer',' ', 'Upbeat'])

fig.for_each_annotation(lambda a: a.update(text = "<b>{}</b>".format(a.text.split('=')[1].title())))

fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>How does my mood change during the day?</i>',
    legend_orientation='h',
    margin_t=120,
    legend_title='Week period',
    legend_y=-0.08
)

# fig.update_layout(
#     legend_x=-0.14,
# )
# write('insta', fig, 'mood_hour', True, False)

# fig.show()
# plot(fig)
write('medium', fig, 'mood_hour', True, False)

# %% [markdown]
# ## Paretto 
# Top 80% músicas e dos artistas

# %% code_folding=[0]
# Paretto artists and tracks
df = stream_df.copy()

df_artist = df.groupby(['artistName']).agg({'hPlayed':'sum', 'id':'count'})
df_artist.index.name = 'name'
df_artist.loc[:, 'category'] = 'artists'
df_track = df.groupby(['trackName']).agg({'hPlayed':'sum', 'id':'count'})
df_track.index.name = 'name'
df_track.loc[:, 'category'] = 'tracks'

df = pd.concat([df_track, df_artist]).reset_index()

df.loc[:, 'rank'] = df.sort_values(by='id', ascending=False).groupby(['category'])['hPlayed'].rank(ascending=False, method='first')
df.loc[:, 'cumulative_hours'] = df.sort_values(by='rank').groupby(['category'])['hPlayed'].cumsum()
df.loc[:, 'cumulative_share'] = 100*df['cumulative_hours']/df.groupby(['category'])['hPlayed'].transform('sum')


df.loc[:, 'text'] = ''
df.loc[df['cumulative_share'].round().isin([20, 50, 80]), 'text'] = df.apply(lambda x: "{:.0f}%: {:.0f} {}".format(np.round(x['cumulative_share']), x['rank'], x['category']), axis=1)

df.loc[:, 'last_text'] = df.sort_values(by='cumulative_share').groupby(['category'])['text'].shift(1)
df.loc[(df['text'] != '') & (df['last_text'] != ''), 'text'] = ''

fig = px.scatter(df.sort_values(by='rank'), x='rank', y='cumulative_share', 
                 facet_col_spacing=0.05, facet_col='category', 
                 text='text', color_discrete_sequence=spotify_colors,
                 category_orders={'category':['artists', 'tracks']}
                )
fig.update_xaxes(matches=None, nticks=4)
fig.update_xaxes(col=1, title='#Artists')
fig.update_xaxes(col=2, title='#Tracks')

fig.update_traces(mode='lines+markers+text', marker_size=2, 
                  textfont_size=12, textposition='bottom right')
fig.update_yaxes(ticksuffix='%', tickvals=[0, 20, 50, 80, 100])

fig.for_each_annotation(lambda a: a.update(text="<b>{}</b>".format(a.text.split('=')[1].title()), font_size=14))

fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>How is my musical taste distributed?',
    yaxis_title='% of total listened time', 
    margin_t=120
)

# plot(fig)

# fig.update_layout(
#     width=600,
#     height=500
# )
# write('insta', fig, 'paretto', True, True)

write('medium', fig, 'paretto', True, True)


# %%
df.loc[df['rank']<=9].sort_values(by=['category', 'rank'])

# %% [markdown]
# ## Sticker or Skipper
# Quanto tempo até passar pra próxima?

# %% code_folding=[]
# Skipping time
df = stream_raw_df.loc[stream_raw_df['year'] == 2021].copy()

df.loc[:, 'sec_disc'] = np.floor(df['sPlayed'])

df = df.groupby(['sec_disc'], as_index=False).agg({'id':'count'}).sort_values(by='sec_disc')
df.loc[:, 'percent'] = df.sort_values(by='sec_disc')['id'].cumsum()/df['id'].sum()

df = df.loc[df['percent'] < 0.99]

fig = px.histogram(df, x='sec_disc', y='id', histfunc='sum', histnorm='percent', nbins=50, 
                   color_discrete_sequence=spotify_colors)
fig.update_yaxes(ticksuffix='%')
fig.update_xaxes(dtick=30)
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>When do I usually skip a song?</i>',
    yaxis_title='Share of streams',
    xaxis_title='Seconds of song'
)

# fig.update_layout(
#     width=650,
#     height=600
# )
# write('insta', fig, 'skip')

write('medium', fig, 'skip')

# %% code_folding=[0]
# # Share of song played 
# df = stream_raw_df.loc[stream_raw_df['year'] == 2021].copy()

# df.loc[:, 'share_played'] = np.floor(100*df['msPlayed']/df['duration_ms'])

# df = df.groupby(['share_played'], as_index=False).agg({'id':'count'}).sort_values(by='share_played')
# df.loc[:, 'percent'] = df.sort_values(by='share_played')['id'].cumsum()/df['id'].sum()

# df = df.loc[df['percent'] < 0.99]

# fig = px.histogram(df, x='share_played', y='id', histfunc='sum', histnorm='percent', nbins=50, 
#                    color_discrete_sequence=spotify_colors)
# fig.update_yaxes(ticksuffix='%')
# fig.update_xaxes(dtick=30)
# fig.update_layout(
#     title='<b>Spotify (un)Wrapped</b>: <i>When do I usually skip a song?</i>',
#     yaxis_title='Share of streams',
#     xaxis_title='Seconds of song'
# )
# plot(fig)

# %% [markdown]
# ## Compulsive Listening Behaviour
# Dia com mais repetições

# %% code_folding=[0]
# Skip rate 
df = stream_raw_df.loc[stream_raw_df['year'] == 2021].sort_values(by='endTime')

df.loc[:, 'skip_count'] = df.groupby(['skip']).cumcount()
df.loc[:, 'count'] = df.groupby(['year'])['id'].cumcount()
df.loc[:, 'cycle_id'] = df['count'] - df['skip_count']

df.loc[:, 'date'] = df['month']

df = df.groupby(['date', 'cycle_id', 'skip'], as_index=False)\
        .agg({'id':'count', 'endTime':'min'})\
        .sort_values(by=['date', 'skip', 'cycle_id'])

df = df.groupby(['date', 'skip'], as_index=False).agg({'id':['sum', 'max']})
df.columns = ['date', 'skip', 'song_count', 'max_streak']

df.loc[:, 'total_songs'] = df.groupby(['date'])['song_count'].transform('sum')

df = df.loc[df['skip']]

df.loc[:, 'skip_rate'] = 100*df['song_count']/df['total_songs']
df.loc[:, 'max_songs_skipped_on_sequence'] = df['max_streak']

fig = px.line(df, x='date', y=['skip_rate', 'max_songs_skipped_on_sequence'], facet_col='variable', 
              color='skip', facet_col_spacing=0.08,
              color_discrete_sequence=spotify_colors)
fig.update_yaxes(matches=None, showticklabels=True, rangemode='tozero', title='')
fig.update_yaxes(col=1, ticksuffix='%')
fig.update_xaxes(tickangle=45, title='',
                 nticks=6)
#                  tickvals=['2021-01-01', '2021-07-01', '2021-12-01'], tickformat='%b/%y')
fig.update_traces(mode='lines+markers+text')
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>Did I skip songs too much this year?',
    margin_t=120,
    showlegend=False
)
fig.for_each_annotation(lambda a: a.update(text = a.text.split('=')[1].replace('_', ' ').title()))

plot(fig)

x = df.groupby(['skip']).agg({'song_count':'sum', 'total_songs':'sum', 'max_streak':'max'})
x.loc[:, 'share'] = 100*x['song_count']/x['total_songs']
x

# %% [markdown]
# ## Song on repeat

# %% code_folding=[0]
# Song streaks 
df = stream_df.copy()

df.loc[:, 'skip_count'] = df.groupby(['id']).cumcount()
df.loc[:, 'count'] = df.groupby(['year'])['id'].cumcount()
df.loc[:, 'cycle_id'] = df['count'] - df['skip_count']

df = df.groupby(['id','artistName', 'trackName','cycle_id'], as_index=False)\
        .agg({'endTime':'count'})

df.sort_values(by='endTime', ascending=False).head(10)

# %% [markdown]
# ## Mainstream

# %% code_folding=[]
# Info Top
df = stream_df.copy()

df = df.groupby(['month', 'artistName'], as_index=False).agg({'endTime':'count'})

df.loc[:, 'rank'] = df.groupby(['month'])['endTime'].rank(ascending=False, method='first')
df.loc[:, 'song_info'] = df.apply(lambda x: "<b>{:.0f}</b>: {} ({:.0f})".format(x['rank'], 
#                                                                                      x['trackName'],
                                                                                     x['artistName'],
                                                                                     x['endTime']
                                                                                    ), axis=1)

df_top = df\
        .loc[df['rank'] <= 10]\
        .sort_values(by='rank')\
        .groupby(['month'], as_index=False)\
        .agg({'song_info': '<br>'.join})

df_top.loc[:, 'song_info'] = '<br>' + df_top['song_info']

# %% code_folding=[0]
# Popularity

df = stream_df.copy()
df = df.groupby(['month']).agg({'popularity':'mean'})

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df))
df_scaled.index = df.index
df_scaled.columns = df.columns.tolist()
df = df_scaled.reset_index()

df.loc[:, 'rank_max'] = df['popularity'].rank(ascending=False)
df.loc[:, 'rank_min'] = df['popularity'].rank(ascending=True)
df.loc[:, 'rank'] = df.apply(lambda x: min([x['rank_min'], x['rank_max']]), axis=1)
df.loc[:, 'text'] = ''
df.loc[df['rank'] <= 2, 'text'] = df['month'].dt.strftime('%d/%b')

df = pd.merge(left=df, right=df_top, on=['month'])

df

range = [df['popularity'].min(), 0, df['popularity'].max()]

fig = px.line(df, x='month', y='popularity', line_shape='spline', text='text',
              color_discrete_sequence=spotify_colors,
              hover_data=['song_info']
             )

fig.update_yaxes(tickvals=range, title='',
                 ticktext=['Less<br>Popular', 'Year<br>average', 'More<br>Popular'])
fig.update_traces(mode='lines+markers+text', textposition='middle right', line_width=2.5)
fig.update_xaxes(nticks=5, tickformat='%b/%y', range=['2020-12-15', '2022-02-01'])
fig.update_layout(
    title='<b>Spotify (un)Wrapped</b>: <i>When was I listening to more mainstream music?</i>',
    xaxis_title='',
    width=650,
    height=600
)
write('insta', fig, 'popularity', True, True)
# plot(fig)

# %%
df

# %% [markdown]
# ## Diversification
# Desvio padrão das features ao longo do tempo (Dia mais coeso, dia mais exploratório)

# %% [markdown]
# ### Data Prep

# %%
df_intra_dist = pd.DataFrame(cdist(song_features_df.loc[:, feature_cols], song_features_df.loc[:, feature_cols]))
df_intra_dist.columns = song_features_df.index.tolist()
df_intra_dist.index = song_features_df.index

df = df_intra_dist.stack().to_frame().reset_index()
df.columns = ['id1', 'id2', 'dist']

df.loc[:, 'key'] = df['id1'] + '-' + df['id2']

intra_dist_df = df.copy()
intra_dist_df.head()

# %%
df = stream_df.copy()

df = df.groupby(['date'], as_index=False).agg({'id':'unique'})
df.loc[:, 'id1'] = df['id']
df.loc[:, 'id2'] = df['id']
df = df.explode('id1')
df = df.explode('id2').drop(columns=['id'])

df.loc[:, 'key'] = df['id1'] + '-' + df['id2']
df = df.loc[df['id1'] != df['id2']]
df.head()

# %%
df_dist = pd.merge(left=df, right=intra_dist_df, how='inner').drop_duplicates(subset=['date', 'key'])
df_dist.loc[:, 'rank'] = df_dist.groupby(['date'])['dist'].rank(ascending=False, method='first')

df = df_dist.loc[df_dist['rank'] == 1].sort_values(by='date')
df.to_csv('./data/prep/distance_between_songs_per_day.csv', index=False)

df.head()

# %% [markdown]
# ### Plot

# %%
df_info = song_info_df.loc[:, ['id', 'info']].copy()
df_info.head()

# %% code_folding=[0]
# Conciseness
df = stream_df.copy()
df.loc[:, 'date'] = df['month']
df = df.groupby(['date'], as_index=False).agg({'id':'unique'})

df_per_week = df.explode('id').reset_index().iloc[:, 1:]

df_with_features = pd.merge(left=df_per_week, right=song_features_df, on='id')

df_avg = df_with_features.groupby(['date']).agg({f:'mean' for f in feature_cols})                          
dist_stream_avg = pd.DataFrame(cdist(df_with_features.loc[:, feature_cols], df_avg))
dist_stream_avg.columns = df_avg.index.tolist()
dist_stream_avg.index=df_with_features.index
dist_stream_avg = dist_stream_avg.stack().to_frame().reset_index()
dist_stream_avg.columns = ['table_id', 'date', 'dist']
dist_stream_avg = pd.merge(left=dist_stream_avg, right=df_per_week.reset_index(),
                           left_on='table_id', right_on='index', suffixes=('_avg', ''))
df = dist_stream_avg.loc[dist_stream_avg['date_avg'] == dist_stream_avg['date']]
df = df.groupby(['date']).agg({'dist':['min', 'max', 'mean', 'median', 'std']})
df.columns = [f'{c[0]}_{c[1]}' for c in df.columns.tolist() if len(c) > 1]
df = df.reset_index()
df.loc[:, 'delta_dist'] = df['dist_max'] - df['dist_min']
df.loc[:, 'norm'] = (df['dist_mean'] - df['dist_mean'].mean())/(df['dist_mean'].max() - df['dist_mean'].min()) 

# range = [df['dist_to_avg_mean'].min(), df['dist_to_avg_mean'].max()]

fig = px.line(df, x='date', y='norm', line_shape='spline',
              color_discrete_sequence=spotify_colors)
fig.update_traces(mode='lines+markers+text')
fig.update_yaxes(tickvals=[-0.5, 0, 0.5], ticktext=['More<br>concise', 'Year avg', 'More<br>expansive'])
plot(fig)
     #'dist_to_avg_min', 'dist_to_avg_max',])
# fig.update_yaxes(tickvals=range, ticktext=['Concise', 'Explorer'])
# plot(fig)
