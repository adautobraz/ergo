import streamlit as st
import pickle
import pandas as pd
import numpy as np


def data_sound_type_share_by__position(subs_df, silences_df, movie_info_df, smooth):
    # Evolution of each sound type during movie
    df_subs = subs_df.loc[:,['imdb_id', 'start', 'end', 'duration']]
    df_subs.loc[:, 'type'] = 'Dialogue'
    df_sil = silences_df.loc[:, ['imdb_id', 'start', 'end', 'duration', 'total_duration']]
    df_sil.loc[:, 'type'] = 'Silence'
    pos_df = pd.concat([df_subs, df_sil], axis=0)
    pos_df.loc[:, 'total_duration'] = pos_df.groupby(['imdb_id']).transform('max')

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
    pvt_df.loc[pvt_df['type'].isin(['Silence', 'Dialogue']) & (pvt_df['value'].isnull()), 'value'] = 0
    pvt_df.loc[pvt_df['type'].isin(['Other sounds']) & (pvt_df['value'].isnull()), 'value'] = smooth

    positions_df = pd.merge(left=pvt_df, right=movie_info_df, on='imdb_id')   

    return positions_df



@st.cache
def load_data(data_path):
    # Subtitles info
    subs_df = pd.read_csv(data_path/'prep/all_subtitles.csv')

    # IMDB data, only to top 150 movies
    imdb_df = pd.read_csv(data_path/'prep/imdb_top250_movies.csv')
    imdb_df = imdb_df.loc[imdb_df['top_250_rank'] <= 150]
    movie_info_df = imdb_df.loc[:, ['imdb_id', 'title', 'year', 'rating', 'genres', 'top_250_rank','color_info']]

    # Movie duration dict
    with open(data_path/ 'prep/movie_duration_dict.pk', 'rb') as r:
        movie_duration_dict = pickle.load(r)

    # Silences info
    silences_df = pd.read_csv(data_path/'prep/silences_info.csv')

    silences_df = pd.merge(left=silences_df, right=movie_info_df, on='imdb_id', how='inner')

    silences_df.loc[:, 'total_duration'] = silences_df['imdb_id'].apply(lambda x: movie_duration_dict[x])
    silences_df.loc[:, 'pos_rel'] = 100*silences_df['start']/silences_df['total_duration']
    silences_df.loc[:, 'dur_rel'] = 100*silences_df['duration']/silences_df['total_duration']

    # Movie summary info
    movies_df = pd.read_csv(data_path/'prep/movies_infos.csv')

    aux_dict = {'silence_dur':'Silence', 'dialogue_dur': 'Dialogue', 'other_dur':'Other sounds'}
    cols = movies_df.columns.tolist()

    for k, v in aux_dict.items():
        cols.remove(k)

    movies_melt = pd.melt(movies_df, id_vars=cols, value_vars=list(aux_dict.keys()))
                
    movies_melt.loc[:, 'var_name'] = movies_melt['variable'].apply(lambda x: aux_dict[x])

    # Positions info
    positions_df = data_sound_type_share_by__position(subs_df, silences_df, movie_info_df, 1)

    # Umap data
    umap_df = pd.read_csv(data_path/'prep/umap_df.csv')


    return {'subs_df':subs_df, 'silences_df': silences_df, 'movies_df':movies_df, 
            'movies_melt_df':movies_melt, 'positions_df':positions_df, 'umap_df':umap_df}