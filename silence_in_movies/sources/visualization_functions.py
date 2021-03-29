import pandas as pd
import plotly.express as px
import streamlit as st
import copy

from .general_functions import facet_prettify, vectorize_column, break_text, format_fig, leave_only_slider
import numpy as np


def space_out(space):
    for i in range(0, space):
        st.text('')

def pad_cols(col_list):
    new_list = [1, *col_list, 1]
    all_cols = st.beta_columns(new_list)
    return all_cols[1:-1]


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

    fig = format_fig(fig)

    fig.update_layout(
        height=200,
        showlegend=False,
        title='Sound type distribution'
    )
    return fig


def sound_type_distribution_bar(movies_melt, sound_color_map):
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
    # df.loc[:, 'text'] = df.apply(lambda x: "<b>{}</b><br>({:.1f}%)".format(x['var_name'], x['percent']), axis=1)

    df.loc[:, 'y'] = 1

    fig = px.bar(df, y='var_name', x='percent', color='var_name' ,
                    color_discrete_map=sound_color_map)

    fig.update_traces(texttemplate='%{x:.1f}%', textposition='inside')

    fig = format_fig(fig)

    fig.update_layout(
        height=300,
        xaxis_range=[-1, 60],
        xaxis_dtick = 50,
        yaxis_title='',
        xaxis_title='Share of movie (%)',
        xaxis_ticksuffix='%',
        showlegend=False,
        title='Sound type distribution'
    )
    return fig


def sound_type_share_by__position(positions_df, sound_color_map):
    
    smooth = 100/(positions_df['position'].nunique() - 1)

    group_df = positions_df\
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

    fig = format_fig(fig)
    return fig


def top_movies_by__type(movies_melt, sound_color_map):
    # Top movies of each sound type
    df = movies_melt.copy()

    df.loc[:, 'var_rank'] = df.groupby(['variable'])['value'].rank(ascending=False)

    df = vectorize_column(df, 'genres')

    df.loc[:, 'movie_genres'] = df['genres'].apply(lambda x: ', '.join(x))

    top = df.loc[df['var_rank'] <= 5].copy()

    top.loc[:, 'text'] = top['title'].apply(lambda x:f"<i>{break_text(x, 10)}</i>")

    fig = px.bar(top, x='var_rank', y='value', facet_row='var_name', orientation='v', color='var_name',
                color_discrete_map= sound_color_map, text='text', facet_row_spacing=0.05,
                hover_name='title', hover_data=['movie_genres']
                )

    # fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(tickvals=[1, 5], title='Ranking', tickprefix='Top ', row=1)

    fig.update_yaxes(range=[-1, 150], dtick= 100, ticksuffix='%', title='Share <br>of movie (%)')

    fig.update_traces(textposition='outside')

    fig = facet_prettify(fig, False)

    fig.update_layout(
        showlegend=False, 
        title='Top 5 movies, for each sound type',
    )
    fig = format_fig(fig)
    return fig


def sound_type_per_position(positions_df, sound_color_map, smooth):
    df = positions_df.copy()
    df.loc[:, 'position'] = np.floor(df['position']/smooth)*smooth

    df = df\
            .groupby(['position', 'imdb_id', 'type', 'top_250_rank', 'title'], as_index=False)\
            .agg({'value':'sum'})

    df.loc[:, 'percent'] = 100*df['value']/smooth

    df.loc[:, 'Movie'] = df.apply(lambda x: f" ({int(x['top_250_rank'])}) {x['title']}", axis=1)

    df = df.sort_values(by=['top_250_rank', 'position'])

    fig = px.line(df, y='percent', x='position', color='type', color_discrete_map=sound_color_map)

    fig.update_xaxes(range=[0,100], title='Position of movie', ticksuffix='%')
    fig.update_yaxes(range=[0,100], ticksuffix='%', title='Relative duration of sound', dtick=25)

    fig.update_layout(
        title='Sound type of each movie, by position',
        legend_title= 'Sound type'
        )
    fig = format_fig(fig)
    return fig


def sound_share_by__type__color(movies_melt, main_palette):
    # Black and White x Color
    df = movies_melt.copy()
    # df = vectorize_column(df, 'color_info')
    df = df\
            .groupby(['color_type', 'var_name'])\
            .agg({'value':['mean', 'median']})\
            .droplevel(0, axis=1)\
            .reset_index()

    df.loc[:, 'color_name'] = df['color_type']
    df.loc[df['color_type'] == 'Color', 'color_name'] = df.apply(lambda x: "{}".format(x['var_name'][0]), axis=1)

    op = 'median'
    df.loc[:, 'rank'] = df.groupby(['var_name'])[op].rank(ascending=False)
    df.loc[:, 'text'] = df[op].apply(lambda x: "{:.0f}%".format(x))
    df.loc[df['rank'] == 1, 'text'] = df[op].apply(lambda x: "<b>{:.0f}%</b>".format(x))

    fig = px.bar(df, x='median', y='var_name', color='color_type', text='text', barmode='group',
                    color_discrete_map={'Color':main_palette[6], 'Black and White':'black'}
                )
                                                
    # fig = facet_prettify(fig)  

    fig.update_yaxes(title='')
    fig.update_xaxes(title='Median usage', col=1, 
                    ticksuffix='%')
    fig.update_xaxes(tickvals=[0,60], range=[0, 65])
    fig.update_traces(textposition='outside')
    fig = format_fig(fig)
    fig.update_layout(
        title='Color x Black and White<br>Is there a difference in sound use?',
        margin_t=120,
        legend_orientation='h',
        legend_title='Color type',
        legend_y=-0.4
    )
    return fig


def sound_share_by__type__genre(movies_melt_df, sound_color_map):
  # All genres
    df = movies_melt_df.copy()

    df.loc[:, 'genres'] = df['genres'].apply(lambda x: eval(x))

    df = pd.DataFrame.explode(df, column='genres')

    df = df\
            .groupby(['genres', 'var_name'])\
            .agg({'value':['median', 'mean'], 'imdb_id':'nunique'})\
            .droplevel(0, axis=1)\
            .reset_index()

    df = pd.melt(df, id_vars=['genres', 'var_name', 'nunique'], var_name='op',
                value_vars=['median', 'mean'])\
            .rename(columns={'nunique':'#Movies'})

    df.loc[:, 'rank_var'] = df.groupby(['var_name', 'op'])['value'].rank(ascending=False)
    df.loc[:, 'text'] = df.apply(lambda x: "{:.1f}%".format(x['value']), axis=1)
    df.loc[:, 'y_axis'] = df.apply(lambda x: "<b>{}</b> ({})".format(x['genres'], x['#Movies']), axis=1)
    df.loc[:, 'color_highlight'] = df.apply(lambda x: x['var_name'] if x['rank_var'] <= 3 else 'Mute', axis=1)
    df.loc[:, 'not_highlight'] = df['color_highlight'] == 'Mute'
    df.loc[:, 'Operation'] = df['op'].str.title()


    df = df.sort_values(by=['Operation', 'not_highlight','var_name', 'value'], ascending=True)

    fig = px.bar(df, x='value', y='y_axis', text='text', orientation='h',
                facet_col_spacing=0.1, facet_col='var_name', animation_frame='Operation',
                hover_name='genres', hover_data=['#Movies'], #category_orders={'y_axis':order},
                color_discrete_map=sound_color_map,  color='color_highlight')

    fig.update_xaxes(title='Share<br>of movie (%)', ticksuffix='%', tickvals=[0, 100], range=[0,110])
    fig.update_yaxes(col=1, autorange='reversed', title='')

    fig = facet_prettify(fig)
    fig = format_fig(fig)

    fig.update_layout(
        title='Genre sound distribution,<br>highlight to Top 3 of each category ',
        margin_t=120,
        margin_r=50,
        height=700,
        showlegend=False)

    fig = leave_only_slider(fig)
    fig['layout']['sliders'][0]['pad']=dict(r= 10, t= 100)

    return fig


def sound_share_strip_per_genre(movies_melt_df, sound_color_map):
    # Strip details
    df = movies_melt_df.copy()

    df.loc[:, 'genres'] = df['genres'].apply(lambda x: eval(x))

    df = pd.DataFrame.explode(df, column='genres')
    df.loc[:, 'Genre'] = df['genres'].apply(lambda x: "<b>{}</b>".format(x))

    df.loc[:, 'rank_var'] = df.groupby(['genres', 'var_name'])['value'].rank(ascending=False)

    df = df.sort_values(by=['Genre'])

    # df.loc[:, 'text'] = df.apply(lambda x: x['title'] if x['rank_var'] == 1 else  '', axis=1)

    fig = px.strip(df, x='value', animation_frame='Genre', y='var_name', hover_data=['year', 'top_250_rank'],
                hover_name='title', color='var_name', color_discrete_map=sound_color_map
                )
    # fig.update_yaxes(ticksuffix='%', dtick=25, rangemode='tozero', range=[0,101])
    fig.update_xaxes(matches=None, range=[-1,101], dtick=25, ticksuffix='%')

    fig = facet_prettify(fig)
    fig = format_fig(fig)
    fig.update_layout(
        showlegend=False, 
        height=400,
        yaxis_title='Sound type',
        title='Sound distribution, per genre',
        xaxis_title='Share of movie')

    fig = leave_only_slider(fig)
    
    return fig


def sound_share_by__type__year(movies_melt_df, sound_color_map):
    df = movies_melt_df.copy()

    df.loc[:, 'decade'] = np.floor(df['year']/10)*10

    df = df\
            .groupby(['decade', 'var_name'])\
            .agg({'imdb_id':'nunique', 'value':['median', 'mean']})\
            .droplevel(0, axis=1)\
            .reset_index()

    df.loc[:, 'x_axis'] = df.apply(lambda x: "{:.0f} ({})".format(x['decade'], x['nunique']), axis=1)

    fig = px.line(df, x='decade', y='mean', color='var_name', color_discrete_map=sound_color_map)
    fig = format_fig(fig)
    fig.update_layout(
        title='Evolution of share of each sound type on movies, per decade',
        yaxis_title='Share of sound type',
        yaxis_ticksuffix='%',
        legend_title='Sound type',
        xaxis_title='Decade',
        xaxis_tickangle=45,
        yaxis_dtick=25,
        height=400
    )
    return fig


def sound_share_strip_by__period(movies_melt_df, sound_color_map):
    # Sound composition, by movie
    df = movies_melt_df.copy()

    df.loc[:, 'period'] = (np.floor(df['year']/20)*20).astype(int)
    df.loc[:, 'period'] = df['period'].apply(lambda x: f"{x}-<br>{x+19}")

    fig = px.strip(df.sort_values(by='period'), y='value', color='var_name',
                facet_col='period', color_discrete_map=sound_color_map,
                hover_name='title', hover_data=['year', 'top_250_rank'])

    fig.update_yaxes(dtick=50, range=[0,110], ticksuffix='%')
    fig.update_xaxes(title='', showticklabels=False)

    fig.update_layout(
        title = 'By which type of sound each movie is composed?', 
        showlegend=True, 
        legend_orientation='h',
        legend_y=-0.1,
        margin_t=120,
        legend_title='Type of sound',
        yaxis_title='')

    fig = facet_prettify(fig)
    fig = format_fig(fig)
    return fig


def sound_type_by__position_genre(positions_df, smooth, sound_type, sound_color_map):
    h = 230
    grey = f"rgb({h},{h},{h})"
    # Test 2: Line
    df = positions_df.copy()
    df.loc[:, 'position_disc'] = np.ceil(df['position']/smooth)*smooth

    df = df\
            .groupby(['imdb_id', 'position_disc', 'genres', 'type'], as_index=False)\
            .agg({'value':'sum'})
            
    df = vectorize_column(df, 'genres')

    df = pd.DataFrame.explode(df, column='genres')

    df = df.groupby(['position_disc', 'genres', 'type'])\
            .agg({'value':['median', 'mean'], 'imdb_id':'nunique'}).droplevel(0, axis=1)\
            .reset_index()

    df = df.loc[df['type'] == sound_type]
    df.loc[:, 'percent'] = 100*df['mean']/smooth

    all_dfs = []
    for g in df['genres'].unique().tolist():
        h_df = df.copy()
        h_df.loc[:, 'animation_group'] = g
        h_df.loc[:, 'is_highlight'] = h_df['genres'] == g
        all_dfs.append(h_df)
        
    highlight_df = pd.concat(all_dfs, axis=0).sort_values(by=['animation_group', 'is_highlight'])
    highlight_df.loc[:, 'Genre highlighted'] = highlight_df['animation_group'].apply(lambda x: f"<b>{x}</b>")
    highlight_df.loc[:, 'Genre'] = highlight_df['genres']

    max_range = max(highlight_df['percent'].max(), 26)

    fig = px.line(highlight_df, x='position_disc', y='percent', color='is_highlight', animation_frame='Genre highlighted',
                color_discrete_map={True:sound_color_map[sound_type], False:grey},
                line_group='Genre')

    fig.update_traces(patch={"line":{"width":3}}, 
                    selector={"name":"True"})


    fig.update_layout(
        title='At which part the movie is the quietest? (View per genre)',
        yaxis_title=f'{sound_type} relevance',
        yaxis_dtick=25,
        yaxis_ticksuffix='%',
        xaxis_title='Position of movie',
        xaxis_ticksuffix='%',    
        showlegend=False,
        xaxis_dtick=25,
        yaxis_range=[0, max_range]
    )

    fig = leave_only_slider(fig)
    fig = format_fig(fig)
    
    return fig


def all_movies_similarity(umap_df):
    # View all movies
    df = umap_df.copy()

    df.loc[:, 'period'] = (np.floor(df['year']/20)*20).astype(int)
    df.loc[:, 'period'] = df['period'].apply(lambda x: f"{x} - {x+19}")
    df.loc[:, 'popularity'] = 1/np.sqrt(df['top_250_rank'].astype(int))

    df = vectorize_column(df, 'genres')
    df.loc[:, 'movie_genres'] = df['genres'].apply(lambda x: ', '.join(x))

    df = df.sort_values(by='period')

    fig = px.scatter(df, x='dim0', y='dim1', hover_name='title', color='period',
                    size='popularity', size_max=20,
                    hover_data=['year', 'top_250_rank', 'movie_genres', 
                                'color_type', 'rating', 'metascore'],
                    color_discrete_sequence=px.colors.sequential.Agsunset
                    )
    fig = format_fig(fig)
    fig.update_xaxes(title='', showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title='', showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        legend_title='Year of release',
        title='Top 150 movies, grouped by similarity',
        height=500
    )

    # text = """
    # To group, we consider each movie release year, sound distribution, associated genres and coloring technique.<br>
    # Here, the dimensions are a projection, so the only real meaning is that closer points have more similar characteristics.
    # """
    # fig.add_hline(y=-3, line_color='white', 
    #             annotation_text=text, annotation_position='top left')
    
    return fig



def all_movies_hist_summary(movies_df, main_palette):
    # General infos
    df = movies_df.copy()

    df.loc[:, 'duration'] = df['total_duration']//(3600)
    df.loc[:, 'duration'] = df['duration'].apply(lambda x: f"{x:.0f}-{x+1:.0f} h")
    df
    df.loc[:, 'period'] = (np.floor(df['year']/20)*20).astype(int)
    df.loc[:, 'period'] = df['period'].apply(lambda x: f"{x} - {x+19}")

    melt = pd.melt(df, id_vars=['imdb_id'], value_vars=['duration', 'year'])

    fig = px.histogram(melt, x='value', facet_col='variable', nbins=10, histnorm='percent',
                    color_discrete_sequence=main_palette[5:])

    fig.update_xaxes(matches=None, categoryorder='category ascending', title='')
    fig = format_fig(fig)
    fig = facet_prettify(fig)
    fig.update_layout(
        title='Distribution of movies regarding duration and year of release',
        yaxis_ticksuffix='%',
        yaxis_title='Share of movies',
        height=400
    )
    return fig
