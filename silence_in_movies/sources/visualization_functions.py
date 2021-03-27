import pandas as pd
import plotly.express as px

from .general_functions import plot, facet_prettify, vectorize_column, break_text
import streamlit as st
import numpy as np


def sound_type_distribution_timeline(movies_melt, sound_color_map):
    # Distribution of each sound type
    df = movies_melt\
            .groupby(['var_name'])\
            .agg({'value':['mean', 'median', 'sum']})\
            .droplevel(0, axis=1)\
            .reset_index()\
            .sort_values(by=['mean'], ascending=False)

    df.loc[:, 'total'] = df['sum'].sum()

    df.loc[:, 'percent'] = 100*df['sum']/df['total']
    # df.loc[:, 'percent'] = np.round(df['percentage'], 2)
    df.loc[:, 'x_end'] = df['percent'].cumsum()
    df.loc[:, 'x_start'] = df['x_end'].shift(1).fillna(0)
    df.loc[:, 'delta'] = df['x_end'] - df['x_start']
    df.loc[:, 'text'] = df.apply(lambda x: "<b>{}</b><br>({:.1f}%)".format(x['var_name'], x['percent']), axis=1)

    df.loc[:, 'y'] = 1

    fig = px.timeline(df, y='y', x_start = 'x_start', x_end='x_end', color='var_name' ,
                    color_discrete_map=sound_color_map, text='text')

    fig.layout.xaxis.type = 'linear'
    for i in range(0, len(fig.data)):
        var_name = fig.data[i].name
        aux = df.loc[df['var_name'] == var_name]
        fig.data[i].x = aux['delta'].tolist()

    fig.update_yaxes(title='', showticklabels=False)
    fig.update_xaxes(title='', tickvals=[0, 100], ticksuffix='%')

    fig.update_traces(insidetextanchor="middle")

    fig.update_layout(
        height=200,
        showlegend=False,
        title='Sound type distribution'
    )
    plot(fig)


def sound_type_share_by__position(subs_df, silences_df, sound_color_map):
    # Evolution of each sound type during movie
    df_subs = subs_df.loc[:,['imdb_id', 'start', 'end', 'duration']]
    df_subs.loc[:, 'type'] = 'Dialogue'
    df_sil = silences_df.loc[:, ['imdb_id', 'start', 'end', 'duration', 'total_duration']]
    df_sil.loc[:, 'type'] = 'Silence'
    pos_df = pd.concat([df_subs, df_sil], axis=0)
    pos_df.loc[:, 'total_duration'] = pos_df.groupby(['imdb_id']).transform('max')

    smooth=1
    pos_df.loc[:, 'pos_rel'] = np.floor(100*pos_df['start']/pos_df['total_duration']/smooth)*smooth
    pos_df.loc[:, 'dur_rel'] = 100*pos_df['duration']/pos_df['total_duration']

    pos_df = pos_df\
                .groupby(['imdb_id','pos_rel', 'type'], as_index=False)\
                .agg({'dur_rel':'sum'})\
                .sort_values(by=['imdb_id', 'pos_rel', 'type'])

    pos_df.loc[:, 'total'] = pos_df.groupby(['imdb_id', 'pos_rel'])['dur_rel'].transform('sum')

    pos_df.loc[:, 'duration'] = pos_df['dur_rel']
    pos_df.loc[pos_df['total'] > smooth, 'duration'] = smooth*pos_df['dur_rel']/pos_df['total']


    other_df = pos_df.groupby(['imdb_id', 'pos_rel'], as_index=False).agg({'duration':'sum'})
    other_df.loc[:, 'duration'] = smooth - other_df['duration']
    other_df.loc[:, 'type'] = 'Other sounds'

    all_df = pd.concat([pos_df, other_df], axis=0).dropna(axis=1)

    pvt_df = pd.pivot_table(all_df, index=['imdb_id'], columns=['pos_rel', 'type'], values='duration')\
                .unstack().to_frame().reset_index()
    pvt_df.columns = ['position', 'type', 'imdb_id', 'value']
    pvt_df.loc[pvt_df['type'].isin(['silence', 'dialogue']) & (pvt_df['value'].isnull()), 'value'] = 0
    pvt_df.loc[pvt_df['type'].isin(['other']) & (pvt_df['value'].isnull()), 'value'] = smooth

    # pvt_df = pvt_df.loc[pvt_df['position'] <= 100]

    group_df = pvt_df\
                .groupby(['type', 'position'])\
                .agg({'value':['mean', 'median', 'sum']})\
                .droplevel(0, axis=1)

    group_df = group_df.div(smooth/100).reset_index()
    group_df.loc[:, 'total'] = group_df.groupby(['position'])['sum'].transform('sum')
    group_df.loc[:, 'percent'] = 100*group_df['sum']/group_df['total']

    fig = px.line(group_df, x='position', y='percent',
                color='type', color_discrete_map=sound_color_map
                )

    fig.update_layout(
        title='Sound evolution during movie',
        legend_title='Sound Type',
        yaxis_title='Percentage of sound',
        yaxis_ticksuffix='%',
        xaxis_ticksuffix='%',
        xaxis_title='Position on movie',
        xaxis_tickvals=[50,100],
        xaxis_range=[0,101],
        xaxis_tick0=25,
        yaxis_tickvals=[0,50, 100],
        yaxis_range=[0,101],
        yaxis_rangemode='tozero',
    )

    plot(fig)


def sound_type_share_by__year__type(movies_melt, sound_color_map):
    # Sound composition, by movie
    df = movies_melt.copy()

    df.loc[:, 'decade'] = (np.floor(df['year']/20)*20).astype(int)
    df.loc[:, 'decade'] = df['decade'].apply(lambda x: f"{x} - {x+19}")

    fig = px.strip(df.sort_values(by='decade'), y='value', color='var_name',
                facet_col='decade', color_discrete_map=sound_color_map,
                hover_name='title', hover_data=['year', 'top_250_rank'])

    fig.update_yaxes(dtick=50, range=[0,110], ticksuffix='%')
    fig.update_xaxes(title='', showticklabels=False)

    fig.update_layout(
        title = 'By which type of sound each movie is composed?', 
        showlegend=True, 
        margin_t=120,
        legend_title='Type of sound',
        yaxis_title='')

    fig = facet_prettify(fig)
    plot(fig)


def top_movies_by__type(movies_melt, sound_color_map):
    # Top movies of each sound type
    df = movies_melt.copy()

    df.loc[:, 'var_rank'] = df.groupby(['variable'])['value'].rank(ascending=False)

    df = vectorize_column(df, 'genres')

    df.loc[:, 'movie_genres'] = df['genres'].apply(lambda x: ', '.join(x))

    top = df.loc[df['var_rank'] <= 10].copy()

    top.loc[:, 'text'] = top['title'].apply(lambda x:f"<i>{break_text(x, 16)}</i>")

    fig = px.bar(top, y='var_rank', x='value', facet_col='var_name', orientation='h', color='var_name',
                color_discrete_map= sound_color_map, text='text', facet_col_spacing=0.1,
                hover_name='title', hover_data=['movie_genres']
                )

    fig.update_yaxes(autorange='reversed')
    fig.update_yaxes(col=1, tickvals=[1, 5, 10], title='Ranking', tickprefix='Top')

    fig.update_xaxes(range=[-1, 101], dtick= 50, ticksuffix='%', title='Share of movie (%)')

    fig.update_traces(textposition='inside')

    fig = facet_prettify(fig, False)

    fig.update_layout(
        showlegend=False, 
        title='Top 10 movies, for each sound type',
        margin_t=100,
        height=600
    )
    plot(fig)


def sound_share_by__type__color(movies_melt, main_palette):
    # Black and White x Color
    df = movies_melt.copy()
    # df = vectorize_column(df, 'color_info')
    df.loc[:, 'color_type'] = df['color_info'].apply(lambda x: 'Color' if 'Color' in x else 'Black and White')

    df = df\
            .groupby(['color_type', 'var_name'])\
            .agg({'value':['mean', 'median']})\
            .droplevel(0, axis=1)\
            .reset_index()

    df.loc[:, 'color_name'] = df['color_type']
    df.loc[df['color_type'] == 'Color', 'color_name'] = df.apply(lambda x: "{}".format(x['var_name'][0]), axis=1)

    op = 'median'
    df.loc[:, 'rank'] = df.groupby(['var_name'])[op].rank(ascending=False)
    df.loc[:, 'text'] = df[op].apply(lambda x: "{:.1f}%".format(x))
    df.loc[df['rank'] == 1, 'text'] = df[op].apply(lambda x: "<b>{:.1f}%</b>".format(x))

    fig = px.bar(df, x='color_type', y='median', facet_col='var_name', color='color_type', text='text',
                    color_discrete_map={'Color':main_palette[1], 'Black and White':'black'}
                )
                                                
    fig = facet_prettify(fig)  

    fig.update_xaxes(title='')
    fig.update_yaxes(title='Median usage of each sound type', col=1, 
                    ticksuffix='%')
    fig.update_yaxes(tickvals=[0,60], range=[0, 61])
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title='Color x Black and White: Is there a difference in sound use?',
        margin_t=120,
        showlegend=False
    )
    plot(fig)