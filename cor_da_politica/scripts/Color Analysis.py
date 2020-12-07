# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# +
import pandas as pd
import colorgram
import os
from pathlib import Path
import cv2
from PIL import Image
from ast import literal_eval
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from colormath.color_conversions import convert_color
from colormath.color_objects import *
from sklearn.decomposition import PCA

pd.set_option('max_columns', None)
# -

Path('./data/graphs').mkdir(exist_ok=True, parents=True)
analysis_folder = Path('./data/graphs')
analysis_folder


# +
def plot(fig, write=False):
    fig.update_layout(template='plotly_white', font_family='Fira Sans')
    
    if write:
        dir_name = analysis_folder/"{}/{}/raw".format(artist)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        
    fig.show()
    
def write(fig, name):
    fig.update_layout(template='plotly_white', font_family='Fira Sans', width=1000)
    fig.write_image('{}/{}.png'.format(analysis_folder, name), scale=10)    


# + [markdown] heading_collapsed=true
# # Color extraction

# + hidden=true
images_folder = './data/logos/png'
backgrounded_folder = './data/logos/backgrounded'

files = [f for f in os.listdir(images_folder) if '.png' in f]

for f in files:
    filename = '{}/{}'.format(images_folder, f)
    im = Image.open(filename)
    fill_color = (220,220,220)  # your new background color

    im = im.convert("RGBA")   # it had mode P after DL it from OP
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1]) # omit transparency
        im = background

    new_filename = '{}/{}'.format(backgrounded_folder, f)

    im.convert("RGB").save(new_filename)

# + hidden=true
colors_dict = {}

for f in files:
    party_colors = []
    colors = colorgram.extract('{}/{}'.format(backgrounded_folder, f), 10)
    party = f.split('.')[0]
    print(party)

    for color in colors:
        party_colors.append({
            'rgb': [color.rgb.r, color.rgb.g, color.rgb.b],
            'hsl': [color.hsl.h, color.hsl.s, color.hsl.l],
            'proportion': color.proportion
            }
        )
    
    
    colors_dict[party] = party_colors

# + hidden=true
colors_df = pd.DataFrame.from_dict(colors_dict, orient='index')
colors_df.index.name = 'party'
cols = colors_df.columns.tolist()
colors_df = pd.melt(colors_df.reset_index(), id_vars='party', value_vars=cols).dropna().set_index('party')
colors_df = colors_df['value'].apply(pd.Series).sort_index()
colors_df.head()

# + hidden=true
colors_df.to_csv('./data/colors_with_background.csv', index=True)

# + [markdown] heading_collapsed=true
# # Color Prep Analysis

# + [markdown] heading_collapsed=true hidden=true
# ## Remove background influence 

# + hidden=true
colors_raw = pd.read_csv('./data/colors_with_background.csv')

colors_raw['rgb'] = colors_raw['rgb'].apply(literal_eval)
colors_raw['hsl'] = colors_raw['hsl'].apply(literal_eval)

colors_raw.loc[:, 'lab'] = colors_raw['rgb'].apply(lambda x: convert_color(sRGBColor(x[0]/255, x[1]/255, x[2]/255), LabColor))
colors_raw.loc[:, 'lab'] = colors_raw['lab'].apply(lambda x: [x.lab_l, x.lab_a, x.lab_b])

colors_raw.loc[:, 'b_mean'] = (np.floor(colors_raw['rgb'].apply(lambda x: np.mean(x))/10)*10).astype(int)
colors_raw.loc[:, 'b_std'] = (colors_raw['rgb'].apply(lambda x: np.std(x)).astype(int))

colors_raw.loc[:, 'rank'] = colors_raw.groupby(['party'])['proportion'].rank(ascending=False)

colors_raw.loc[:, 'is_background'] = False

background_colors = colors_raw\
                        .loc[
                            (colors_raw['b_mean'].between(210, 220)) 
                            & (colors_raw['b_std'] <= 10)]\
                        .sort_values(by='proportion', ascending=False)\
                        .drop_duplicates(subset=['party'], keep='first')\
                        .index.tolist()

colors_raw.loc[colors_raw.index.isin(background_colors), 'is_background'] = True

colors_raw.head()

# + hidden=true
colors_raw.loc[(colors_raw['party'] == 'PT')]

# + hidden=true
real_colors = colors_raw.loc[~colors_raw['is_background']].copy()

real_colors.loc[:, 'total_area'] = real_colors.groupby(['party'])['proportion'].transform('sum')

real_colors.loc[:, 'color_importance'] = real_colors['proportion']/real_colors['total_area']

real_colors.head()

# + hidden=true
real_colors.loc[(real_colors['party'] == 'PT')]

# + hidden=true
real_colors.loc[:, ['party', 'rgb', 'hsl', 'lab', 'color_importance']].to_csv('./data/colors_cleaned.csv', index=False)

# + [markdown] heading_collapsed=true hidden=true
# ## Main Colors

# + hidden=true
colors_df = pd.read_csv('./data/colors_cleaned.csv')

colors_df.loc[:, 'rank'] = colors_df.groupby(['party'])['color_importance'].rank(ascending=False)

colors_df['rgb'] = colors_df['rgb'].apply(literal_eval)
colors_df['hsl'] = colors_df['hsl'].apply(literal_eval)
colors_df['lab'] = colors_df['lab'].apply(literal_eval)

colors_df.loc[:, 'key'] = colors_df['party'] + '-' + colors_df['rank'].astype(int).astype(str)

# + hidden=true
colors_df.sort_values(by='rank', inplace=True)

colors_df.loc[:, 'cumul_importance'] = colors_df.groupby(['party'])['color_importance'].cumsum()

colors_df.head()

# + hidden=true
colors_df['party'].nunique()

# + hidden=true
px.line(colors_df, x='rank', y='cumul_importance', color='party')

# + hidden=true
# Choose colors threshold
main_colors = colors_df.copy()

# + hidden=true
df = main_colors.groupby(['party'], as_index=False).agg({'key':'count'}).sort_values(by='key')
px.bar(df, x='party', y='key')

# + [markdown] heading_collapsed=true hidden=true
# ## Color clusters

# + hidden=true
rgb_features = main_colors['rgb'].apply(pd.Series)
rgb_features.columns = ['rgb_r', 'rgb_g', 'rgb_b']

hsl_features = main_colors['hsl'].apply(pd.Series)
hsl_features.columns = ['hsl_h', 'hsl_s', 'hsl_l']

lab_features = main_colors['lab'].apply(pd.Series)
lab_features.columns = ['lab_l', 'lab_a', 'lab_b']

# features = pd.concat([rgb_features, hsl_features, lab_features], axis=1)
features = pd.concat([lab_features], axis=1)

features.head()

# + hidden=true
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

scaled_features = features

inertias = {}    
for k in range(1, 30):
    model = KMeans(k)
    model.fit(scaled_features)
    inertias[k] = model.inertia_
    
cluster_size_df = pd.DataFrame.from_dict(inertias, orient='index').reset_index()
cluster_size_df.columns = ['cluster_size', 'inertia']

px.line(cluster_size_df, x='cluster_size', y='inertia')

# + hidden=true
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

scaled_features = features
model = KMeans(5)
model.fit(scaled_features)

labels = model.labels_

# + hidden=true
features_df = features.copy()
features_df.loc[:, 'cluster'] = labels
features_df.loc[:, 'importance'] = main_colors['color_importance']
features_df.loc[:, 'party'] = main_colors['party']


cols = ['rgb_r', 'rgb_g', 'rgb_b']
features_df.loc[:, cols] = rgb_features.loc[:, cols]

for c in cols:
    features_df.loc[:, c.split('_')[1]] = features_df[c]# * features_df['importance']

oper_dict = {
    'party':['count', 'nunique'],
    'importance':'sum',
    'r':'median',
    'g':'median',
    'b':'median',    
}

cluster_df = features_df.groupby(['cluster'], as_index=False).agg(oper_dict)

cluster_df.columns = ['cluster', 'total_colors', 'presence', 'prop_weight', 'r', 'g', 'b']

cluster_df.sort_values(by='total_colors', ascending=False, inplace=True)

cluster_df.loc[:, 'cluster'] = cluster_df['cluster'].astype(str)
# cluster_df.loc[:, 'prop_weight'] = cluster_df['total_colors']
cluster_df.loc[:, 'prop_weight'] = 1

cluster_df.loc[:, 'color'] = cluster_df.apply(lambda x: 'RGB({},{},{})'.format(int(x['r']/x['prop_weight']), 
                                                                               int(x['g']/x['prop_weight']), 
                                                                               int(x['b']/x['prop_weight'])), axis=1)

group_name = ['Pastéis', 'Azuis', 'Amarelos', 'Vermelhos', 'Verdes'] 
cluster_df.loc[:, 'cluster_name'] = group_name

color_map = cluster_df.set_index('cluster')['color'].to_dict()
color_names = cluster_df.set_index('cluster')['cluster_name'].to_dict()

display(cluster_df.head())

fig = px.bar(cluster_df, x='cluster', y='total_colors', color='cluster', color_discrete_map=color_map)
plot(fig)

# + hidden=true
main_colors_fit = main_colors.copy()
main_colors_fit.loc[:, 'color_label'] = features_df['cluster'].astype(str)
main_colors_fit.loc[:, 'color_cluster'] = main_colors_fit['color_label'].apply(lambda x: color_names[x])

main_colors_fit.loc[:, 'group_color_repr'] = main_colors_fit['color_label'].apply(lambda x: color_map[x])

main_colors_fit.loc[:, 'actual_color'] = main_colors_fit['rgb'].apply(lambda x: 'RGB({},{},{})'.format(x[0], x[1], x[2]))

main_colors_fit.loc[:, 'hue'] = main_colors_fit['hsl'].apply(lambda x: x[0])
main_colors_fit.loc[:, 'sat'] = main_colors_fit['hsl'].apply(lambda x: x[1])
main_colors_fit.loc[:, 'lum'] = main_colors_fit['hsl'].apply(lambda x: x[2])

actual_colors_map = main_colors_fit.set_index('key')['actual_color'].to_dict()

main_colors_fit.head()

# + hidden=true
main_colors_fit.to_csv('./data/party_colors_infos.csv', index=False)
# -

# # Data Analysis

# +
# Load data, column prep and info merge
party_df = pd.read_csv('./data/partidos_infos.csv')

party_df.loc[:, 'year'] = party_df['creation_date'].str[-4:].astype(int)
party_df.loc[:, 'year_disc'] = 10*np.floor(party_df['year']/10).astype(int)
party_df.loc[:, 'number_disc'] = 10*np.floor(party_df['electoral_num']/10).astype(int)

party_colors_df = pd.read_csv('./data/party_colors_infos.csv')

party_colors_df.loc[:, 'color_label'] = party_colors_df['color_label'].astype(str)

party_colors_df = pd.merge(left=party_colors_df, right=party_df, left_on='party', right_on='code', how='left')

party_colors_df.head()
# -

color_map = party_colors_df.set_index('color_cluster')['group_color_repr'].to_dict()
actual_colors_map = party_colors_df.set_index('key')['actual_color'].to_dict()

# ### Visão de todas as cores

# +
df = party_colors_df.sort_values(by=['color_cluster', 'hue', 'sat', 'lum'])
df.loc[:, 'aux'] = 1

df.loc[:, 'rank'] = df['hue'].rank(method='first')
fig = px.bar(df, x='rank', y='aux', color='key',  color_discrete_map=actual_colors_map)
fig.update_yaxes(showticklabels=False, title='')
fig.update_xaxes(showticklabels=False, title='')
fig.update_traces(marker_line_color='white', marker_line_width=0.0001)
fig.update_layout(showlegend=False, 
#                   title='Todas as cores encontradas nos logos dos partidos',
                  titlefont_size=18
                 )

plot(fig)
write(fig, 'all_party_colors_bar')

# + code_folding=[]
#Labels
for c in ['hue', 'sat', 'lum']:
    df = party_colors_df.sort_values(by=c)
    df.loc[:, 'aux'] = 1
    all_labels = df['key'].tolist() + ['']

    #Parents
    key_parents = ['' for i in range(0, df.shape[0])]
    all_parents = key_parents + ['']

    #Values
    key_values = df['aux'].tolist()

    all_values = key_values + [df['aux'].sum()]

    #Colors
    marker_colors = df['actual_color'].tolist()

    import plotly.graph_objects as go


    fig = go.Figure(go.Treemap(
            branchvalues = "total",
        labels = all_labels,
        parents = all_parents,
        values = all_values,
        marker_colors=marker_colors,
        textinfo='none'
    ))

    fig.update_layout(title='Todas as cores presentes nas logos dos partidos', titlefont_size=24)
    
    fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.12)

    plot(fig)
    write(fig, 'party_colors_{}'.format(c))

# +
#Labels
df = party_colors_df.loc[:, ['hue', 'sat', 'lum']]

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=['dim0', 'dim1'])
df_pca.index = party_colors_df.index

df_pca.loc[:, 'actual_color'] = party_colors_df['actual_color']
df_pca.loc[:, 'party'] = party_colors_df['party']
df_pca.loc[:, 'key'] = party_colors_df['key']

df_pca.head()

fig = px.scatter(df_pca, x='dim0', y='dim1', symbol='party', size_max=30,
                 color='key', color_discrete_map=actual_colors_map)
plot(fig)

# +
df = party_colors_df.sort_values(by=['color_importance', 'hue', 'party'], ascending=False)
df.loc[:, 'predominace'] = 100*df['color_importance']
                                 
fig = px.bar(df, y='party', x='predominace', color='key', 
             color_discrete_map=actual_colors_map, height=700)
fig.update_layout(
    showlegend=False, 
    yaxis_title='Partido', 
    xaxis_title='Predominância da cor (%)',
    title='Partidos e suas cores',
    titlefont_size=24
)
fig.update_traces(marker=dict(line_color='lightgray'))

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.12)
plot(fig)
write(fig, 'party_colors')
# -

# ### Cores agrupadas

# +
# All clusters view
df = party_colors_df\
        .groupby(['color_cluster', 'color_label'], as_index=False)\
        .agg({'color_importance':'sum', 'party':'nunique'})

df.loc[:, 'counter'] = 1
df.loc[:, 'counter'] = 1
df.loc[:, 'total_parties'] = party_colors_df['party'].nunique()
df.loc[:, 'presence'] = 100*df['party']/df['total_parties']
df.loc[:, 'rank'] = df['presence'].rank(method='first', ascending=False) - 1

df.loc[:, 'text'] = df['presence'].apply(lambda x: 'Presente em <b>{:.1f}%</b><br>dos partidos'.format(x))


df.sort_values(by='presence', ascending=False, inplace=True)

fig = px.scatter(df, x='color_cluster', y='counter', color='color_cluster', size='presence', text='color_cluster',
           color_discrete_map=color_map, size_max=90)

fig.update_traces(
    marker=dict(line=dict(width=0.1,color='lightgray'), opacity=1))

fig.update_traces(textposition='top center', textfont_size=16, texttemplate='<b>%{text}</b>')

fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0,2])
fig.update_xaxes(showgrid=False, tickfont_size=14, tickmode='array', tickvals=df['rank'],
                ticktext=df['text'])

fig.update_layout(
    title='5 grupos de cor principais',
    titlefont_size=24,
    xaxis_title='', 
    yaxis_title='', 
    showlegend=False)

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.2)

plot(fig)
write(fig, 'color_clusters')

# +
df = party_colors_df.sort_values(by=['color_cluster', 'hue'])
df.loc[:, 'aux'] = 1.0
fig = px.bar(df, y='color_cluster', x=['aux', 'color_importance'], facet_col='variable',
             color='key',  color_discrete_map=actual_colors_map)
fig.update_traces(marker_line_color='grey', marker_line_width=0.05)
fig.update_xaxes(matches=None)
fig.update_layout(showlegend=False, title='5 grandes grupos de cor')

plot(fig)

# +
#Labels
df = party_colors_df.sort_values(by='hue')
df.loc[:, 'aux'] = 1
all_labels = df['key'].tolist() + df['color_cluster'].sort_values().unique().tolist() + ['Todas as cores']

#Parents
key_parents = df['color_cluster'].tolist()
group_parents = ['Todas as cores' for f in range(0, df['color_cluster'].nunique())] 
all_parents = key_parents + group_parents + ['']

#Values
key_values = df['aux'].tolist()

group_values_df = df.groupby(['color_cluster'], as_index=False).agg({'aux':'sum'})
group_values = group_values_df.sort_values(by='color_cluster')['aux'].tolist()

all_values = key_values + group_values + [df['aux'].sum()]

#Colors
marker_colors = df['actual_color'].tolist() + df.sort_values(by='color_cluster')['group_color_repr'].unique().tolist() + ['white']

import plotly.graph_objects as go


fig = go.Figure(go.Treemap(
        branchvalues = "total",
    labels = all_labels,
    parents = all_parents,
    values = all_values,
    marker_colors=marker_colors,
    textinfo='none'
))

fig.update_layout(title='Todas as cores presentes em cada grupo', titlefont_size=24)
fig.update_traces(textfont_size=14)

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.12)

plot(fig)
write(fig, 'color_clusters_all_colors')
# -

# ### Partidos e Cores

# +
df = party_colors_df\
        .groupby(['party', 'color_cluster', 'group_color_repr'], as_index=False)\
        .agg({'color_importance':'sum'})\
        .sort_values(by=['color_importance'], ascending=False)\
        .reset_index()

df.loc[:, 'predominace'] = 100*df['color_importance']
df.loc[:, 'color'] = df['group_color_repr']
df.loc[:, 'key'] = df['index'].astype(str)

aux_dict = df.set_index('key')['color'].to_dict()
                                 
fig = px.bar(df, y='party', x='predominace', color='key',
             color_discrete_map=aux_dict, height=700)

fig.update_layout(
    showlegend=False, 
    yaxis_title='Partido',
    title='Partidos e suas cores, por grupo de cor'
)
fig.update_xaxes(title='Predominância (%)')
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
plot(fig)
write(fig, 'party_colors_disc')

# +
df = party_colors_df\
        .groupby(['party', 'color_cluster', 'group_color_repr'], as_index=False)\
        .agg({'color_importance':'sum'})\
        .sort_values(by=['color_importance'])

df.loc[:, 'predominace'] = 100*df['color_importance']

df.loc[:, 'max_value'] = df.groupby(['color_cluster'])['predominace'].transform('max')

df.loc[:, 'text'] = df['predominace'].apply(lambda x: '{:.0f}%'.format(x))
df.loc[df['max_value'] == df['predominace'], 'text'] = df['predominace'].apply(lambda x: '<b>{:.0f}%<b>'.format(x))


fig = px.bar(df, y='party', x='predominace', color='color_cluster', facet_col='color_cluster',
             color_discrete_map=color_map, height=700, text='text', 
             category_orders={'color_cluster':['Vermelhos', 'Escuros', 'Verdes', 'Amarelos', 'Claros']}
            )

fig.update_layout(
    showlegend=False, 
    yaxis_title='Partido',
    title='Partidos e suas cores, por grupo de cor ',
    titlefont_size=24
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))

fig.update_traces(textposition='outside')
fig.update_xaxes(title='Predominância (%)', range=[0, 130])


parties_order = df['party'].sort_values(ascending=False).unique().tolist()
fig.update_yaxes(categoryorder='array', categoryarray=parties_order)

plot(fig)
write(fig, 'party_colors_per_cluster')
# +
df = party_colors_df\
        .groupby(['party', 'color_cluster', 'group_color_repr'], as_index=False)\
        .agg({'color_importance':'sum'})\
        .sort_values(by=['color_importance'])

df.loc[:, 'predominance'] = 100*df['color_importance']

df.loc[:, 'rank'] = df.groupby(['color_cluster'])['predominance'].rank(method='first', ascending=False)

df.loc[:, 'max_value'] = df.groupby(['color_cluster'])['predominance'].transform('max')

df.loc[:, 'text'] = df['predominance'].apply(lambda x: '{:.0f}%'.format(x))
df.loc[df['max_value'] == df['predominance'], 'text'] = df['predominance'].apply(lambda x: '<b>{:.0f}%<b>'.format(x))

top_10 = df.loc[df['rank'] < 11]
fig = px.bar(top_10, y='party', x='predominance', color='color_cluster', facet_col='color_cluster',
             color_discrete_map=color_map, text='text', facet_col_spacing=0.1,
             category_orders={'color_cluster':['Vermelhos', 'Escuros', 'Verdes', 'Amarelos', 'Claros']}
            )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))

fig.update_yaxes(matches=None, showticklabels=True)

fig.update_layout(
    showlegend=False, 
    yaxis_title='Partido',
    title='Top 10 partidos com maior presença de cada grupo de cor',
    titlefont_size=24,
    height=600
)


fig.update_traces(textposition='outside')
fig.update_xaxes(title='Predominância (%)', range=[0, 150], tickvals=[0,100])

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.17)

plot(fig)
write(fig, 'cluster_colors_top_10_party')
# -

# ### Posição política

color_map

# +
df = party_colors_df\
        .groupby(['party', 'color_cluster', 'position'], as_index=False)\
        .agg({'color_importance':'sum'})\
        .sort_values(by=['color_importance'], ascending=False)

df.loc[:, 'predominace'] = 100*df['color_importance']
df.loc[:, 'Grupo de Cor'] = df['color_cluster']
                                 
fig = px.bar(df, y='party', x='predominace', color='Grupo de Cor', facet_col='position',
             color_discrete_map=color_map, height=600, facet_col_spacing=0.12, 
             category_orders={'position':['esquerda', 'centro', 'direita']})

fig.update_layout(
    showlegend=True, 
    yaxis_title='Partido',
    title='Partidos, sua composição cromática e posição política',
    titlefont_size=24
)

# fig.update_traces(texttemplate='%{x:.0f}%', textposition='outside')
fig.update_xaxes(title='Predominância (%)', range=[0, 130])
fig.update_yaxes(matches=None, showticklabels=True)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.15,
            y=-0.18)

plot(fig)
write(fig, 'party_colors_per_position')

# +
df = party_colors_df\
        .groupby(['color_cluster', 'position'], as_index=False)\
        .agg({'party':'nunique', 'color_importance':'sum'})\
        .sort_values(by=['position', 'color_importance'], ascending=False)

df.loc[:, 'total_importance'] = df.groupby(['position'])['color_importance'].transform('sum')
df.loc[:, 'color_dist'] = 100*df['color_importance']/df['total_importance']
                                 
fig = px.bar(df, x='color_cluster', y='color_dist', color='color_cluster', barmode='relative',
            height=600, facet_col_spacing=0.12, facet_col='position', color_discrete_map=color_map,
             category_orders={'position':['esquerda', 'centro', 'direita']})

fig.update_xaxes(matches=None, categoryorder='total descending', title='', tickfont_size=14)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.update_traces(texttemplate='<b>%{y:.0f}%<b>', textposition='inside')


fig.update_layout(
    showlegend=False, 
    yaxis_title='Proporção das cores (%)',
    yaxis_titlefont_size=16,
    title='A Esquerda é vermelha; o Centro e a Direita, azuis',
    titlefont_size=24
)

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.18)


plot(fig)
write(fig, 'position_colors')

# +
df = party_colors_df\
        .groupby(['color_cluster', 'position'], as_index=False)\
        .agg({'party':'nunique', 'color_importance':'sum', 'affiliates':'sum'})\
        .sort_values(by=['position', 'color_importance'], ascending=False)

df.loc[:, 'prop_aff'] = df['color_importance']*df['affiliates']
df.loc[:, 'total_weights'] = df.groupby(['position'])['prop_aff'].transform('sum')
df.loc[:, 'color_dist_aff'] = 100*df['prop_aff']/df['total_weights']

df.loc[:, 'total_importance'] = df.groupby(['position'])['color_importance'].transform('sum')
df.loc[:, 'color_dist_area'] = 100*df['color_importance']/df['total_importance']
 
                                 
fig = px.bar(df, x='color_cluster', y='color_dist_aff', color='color_cluster', barmode='relative',
            height=600, facet_col_spacing=0.12, facet_col='position', color_discrete_map=color_map,
             category_orders={'position':['esquerda', 'centro', 'direita']})

fig.update_xaxes(matches=None, categoryorder='total descending', title='', showticklabels=True)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.update_traces(texttemplate='<b>%{y:.0f}%<b>', textposition='inside')

fig.update_layout(
    height=500,
    showlegend=False, 
    title='Composição cromática dos partidos de cada posição política (Ponderada por número de afiliados)',
    yaxis_title='Distribuição das cores x posição política (%)'
)


plot(fig)
write(fig, 'position_colors_pond_aff')

# +
df = party_colors_df\
        .groupby(['color_cluster', 'position'], as_index=False)\
        .agg({'party':'nunique', 'color_importance':'sum'})\
        .sort_values(by=['position', 'color_importance'], ascending=False)

df.loc[:, 'total_importance'] = df.groupby(['color_cluster'])['color_importance'].transform('sum')
df.loc[:, 'color_dist'] = 100*df['color_importance']/df['total_importance']

df.loc[:, 'max'] = df.groupby(['color_cluster'])['color_dist'].transform('max')

df.loc[:, 'text'] = df['color_dist'].apply(lambda x: '{:.0f}%'.format(x))
df.loc[df['color_dist'] == df['max'], 'text'] = df['color_dist'].apply(lambda x: '<b>{:.0f}%<b>'.format(x))
                                 
fig = px.bar(df, y='color_cluster', x='color_dist', color='color_cluster', barmode='relative', text='text',
            height=600, facet_col_spacing=0.12, facet_col='position', color_discrete_map=color_map,
             category_orders={'position':['esquerda', 'centro', 'direita']})

fig.update_xaxes(matches=None, categoryorder='total descending',  
                 tickfont_size=14, range=[0,110], tickvals=[0,100], title='Proporção da cor por posição(%)'
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.update_traces(textposition='outside')


fig.update_layout(
    showlegend=False, 
    yaxis_titlefont_size=16,
    yaxis_title='Grupo de cor',
    title='Vermelho à Esquerda; Amarelo ao Centro; Azuis, Verdes e Pastéis à Direita',
    titlefont_size=22
)

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.18)

plot(fig)
write(fig, 'colors_position')
# -

# ### Data de Criação

# +
df = party_colors_df\
        .groupby(['year_disc'], as_index=False)\
        .agg({'party':'nunique'})\
        .sort_values(by=['year_disc'])

fig = px.line(df, x='year_disc', y='party')

fig.update_layout(
    height=500,
    showlegend=False, 
    title='Criação dos partidos ainda em atividade, por década',
    yaxis_title='Número de partidos'
)
fig.update_traces(mode='lines+markers')

plot(fig)

df = party_colors_df\
        .groupby(['year_disc', 'position'], as_index=False)\
        .agg({'party':'nunique'})\
        .sort_values(by=['year_disc'])

fig = px.line(df, x='year_disc', y='party', facet_col='position')

fig.update_layout(
    height=500,
    showlegend=False, 
    title='Criação dos partidos ainda em atividade, por década e posição política',
    yaxis_title='Número de partidos'
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.update_traces(mode='lines+markers')

plot(fig)

# +
df = party_colors_df\
        .groupby(['year_disc', 'color_cluster'], as_index=False)\
        .agg({'color_importance':'sum'})\
        .sort_values(by=['year_disc'], ascending=True)


df.loc[:, 'total_importance'] = df.groupby(['year_disc'])['color_importance'].transform('sum')
df.loc[:, 'color_dist'] = 100*df['color_importance']/df['total_importance']

fig = px.line(df, x='year_disc', y='color_dist', color='color_cluster', 
              color_discrete_map=color_map
              ,facet_col='color_cluster', facet_col_spacing=0.05
             )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))
fig.update_xaxes(title='Década de criação')

fig.update_layout(
    height=500,
    showlegend=False, 
    title='Presença de cada cor nos partidos criados, por década',
    yaxis_title='Presença nas logos (%)',
    titlefont_size=24,
    yaxis_titlefont_size=16,
    
)

fig.update_traces(mode='lines+markers')

fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.22)

plot(fig)
write(fig, 'time_colors')

# + [markdown] heading_collapsed=true
# ### Número eleitoral

# + hidden=true
df = party_colors_df\
        .groupby(['number_disc'], as_index=False)\
        .agg({'party':'nunique'})\
        .sort_values(by='number_disc', ascending=False)

df.loc[:, 'number'] = df['number_disc'].apply(lambda x: '{} - {}'.format(x, x+9))

df.loc[:, 'text'] = df['party'].apply(lambda x: '{} partido'.format(x))
df.loc[df['party'] > 1, 'text'] = df['text'] + 's'

fig = px.bar(df, y='number', x='party', orientation='h', text='text')
fig.update_traces(textposition='inside')
fig.update_layout(
    yaxis_title='Faixa de número do partido',
    title='Total de partidos ativos, por faixa de número eleitoral'
)
fig.update_xaxes(showgrid=False, title='', showticklabels=False)

plot(fig)

# + hidden=true
df = party_colors_df\
        .groupby(['number_disc', 'color_cluster'], as_index=False)\
        .agg({'party':'nunique', 'color_importance':'sum'})

df.loc[:, 'number'] = df['number_disc'].apply(lambda x: '{} - {}'.format(x, x+9))
df.loc[:, 'total_importance'] = df.groupby(['number_disc'])['color_importance'].transform('sum')
df.loc[:, 'color_prop'] = 100*df['color_importance']/df['total_importance']
#.apply(lambda x: '{} - {}'.format(x, x+9))


df.sort_values(by=['number_disc'], ascending=True)
yaxis_order = df['number'].tolist()

fig = px.bar(df, y='number', x='color_prop', orientation='h', color_discrete_map=color_map,
             color='color_cluster', facet_col='color_cluster', category_orders={'number':yaxis_order})
fig.update_traces(texttemplate='%{x:.0f}%', textposition='outside')

fig.update_layout(
    yaxis_title='Faixa de número do partido',
    title='Distribuição de cor dos partidos ativos, por faixa de número eleitoral',
    showlegend=False
)

fig.update_layout(title={'y':0.97})

fig.update_xaxes(showgrid=True, title='', showticklabels=False, range=[0,110])

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].title()))

plot(fig)
# -

# ### Representativade

# +
vereadores_sp = pd.read_csv('./data/vereadores_sp.csv')
vereadores_sp.head()

vereadores_sp.loc[:, 'party'] = vereadores_sp['partido']
vereadores_sp.loc[vereadores_sp['party'] == 'PODEMOS', 'party'] = 'PODE'

ver_sp_colors = pd.merge(left=vereadores_sp, right=party_colors_df, on='party', how='left')

ver_sp_colors.loc[:, 'prop_occup'] = ver_sp_colors['ocupantes'] * ver_sp_colors['color_importance']
ver_sp_colors.head(3)

# +
df = ver_sp_colors.sort_values(by=['color_cluster', 'hue', 'sat', 'lum'])
df.loc[:, 'aux'] = 1

df.loc[:, 'rank'] = df['hue'].rank(method='first')

fig = px.bar(df, x='rank', y='aux', color='key',  color_discrete_map=actual_colors_map)
fig.update_yaxes(showticklabels=False, title='')
fig.update_xaxes(showticklabels=False, title='')
fig.update_traces(marker_line_color='grey', marker_line_width=0.05)
fig.update_layout(showlegend=False, 
                  title='Todas as cores encontradas nos logos dos partidos',
                  titlefont_size=18
                 )

plot(fig)

# +
df = ver_sp_colors.groupby(['color_cluster'], as_index=False).agg({'prop_occup':'sum'})

fig = px.bar(df, x='color_cluster', y='prop_occup', color='color_cluster',  color_discrete_map=color_map)
fig.update_yaxes(showticklabels=False, title='')
fig.update_xaxes(title='', categoryorder='total descending')
fig.update_traces(texttemplate='%{y:.0f} cadeiras', textposition='outside')
fig.update_layout(showlegend=False, 
                  title='Ocupação Cromática da Câmara de Vereadores de São Paulo, para 2021',
                  titlefont_size=24
                 )


fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.22)

plot(fig)
write(fig, 'sp_camara_vereadores')

# +
df = ver_sp_colors

df["all"] = "all" # in order to have a single root node

color_dict = actual_colors_map.copy()
color_dict['(?)'] = 'white'
color_dict['Escuros'] = 'black'

fig = px.treemap(df, path=['all', 'color_cluster', 'key'], values='prop_occup', color='key',
                  color_discrete_map=color_dict
                )

          
fig.show()

# +
#Labels
df = ver_sp_colors
all_labels = df['key'].tolist() + df['color_cluster'].sort_values().unique().tolist()

#Parents
key_parents = df['color_cluster'].tolist()
group_parents = ['' for f in range(0, df['color_cluster'].nunique())]
all_parents = key_parents + group_parents

#Values
key_values = df['prop_occup'].tolist()

group_values = df.groupby(['color_cluster'], as_index=False).agg({'prop_occup':'sum'})
group_values = group_values.sort_values(by='color_cluster')['prop_occup'].tolist()

all_values = key_values + group_values

#Colors
marker_colors = df['actual_color'].tolist() + df.sort_values(by='color_cluster')['group_color_repr'].unique().tolist()

import plotly.graph_objects as go


fig = go.Figure(go.Treemap(
        branchvalues = "total",
    labels = all_labels,
    parents = all_parents,
    values = all_values,
    marker_colors=marker_colors,
    textinfo='label'
))

fig.update_layout(
    title='Ocupação Cromática da Câmara de Vereadores de São Paulo para 2021',
    titlefont_size=24
)


fig.add_annotation(text='Autor: Adauto Braz',
            align='right',
            showarrow=False,
            font_size=11,
            xref='paper',
            yref='paper',
            x=1.08,
            y=-0.22)

plot(fig)
write(fig, 'sp_colors_details')
# -


