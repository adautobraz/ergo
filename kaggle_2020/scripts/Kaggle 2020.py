# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python (Data Science)
#     language: python
#     name: datascience
# ---

# +
import pandas as pd
import plotly.express as px
import itertools
# import enchant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import unidecode
from prince import MFA
import math
from IPython.display import display

pd.set_option('display.max_colwidth', None)

# d = enchant.Dict("en_US")
pd.set_option('max_columns', None)
# -

# # Story Map

# Stories to cover
#
# The 5 Dimensions of Data Science
# Understanding our landscape through the lens of 
# Identity, Practice, Context, Toolset and Learning
#
# Identity/Self/
# - Country
# - Gender
# - Age
# - Education
# - Profession
# - Compensation
# - Time of coding
#
#
# Apparatus/ Toolset 
# - Languages
# - Tools
#     - IDE's
#     - Notebooks
#     - Machine
#     - TPU
#     - Viz
#     - Deploy
#     
# Practice/Praxis/Experience/
# - ML time
# - Frameworks
# - Algorithms
# - Computer Vision
# - NLP
# - Automation
#
# Context/ Ambience/ Surroundings/ 
# - Employer size
# - Team size
# - Model use
# - Work roles
# - Investment
# - Platform
# - Database
# - Tools
#
# Learning/ Training/ Knowledge
# - Courses
# - Process
# - Knowledge interests

# +
# A Decolonial look to Data Science
# The Brazilian Way/ Tropical
# ALICE in DataScienceLand: Understanding Apparatus, Learning, Identity, Context and Experience in the Data Sciece Community

# +
anagrams = []
for d1 in ['I', 'S']:
    for d2 in ['A', 'T']:
        for d3 in ['P', 'E']:
            for d4 in ['C', 'A', 'S']:
                for d5 in ['L', 'T', 'K']:
                    string = d1+d2+d3+d4+d5
                    anagrams.extend(["".join(perm) for perm in itertools.permutations(string)])

anagrams = sorted(list(set(anagrams)))

english_words = [a for a in anagrams if d.check(a.title())]
english_words

# + [markdown] heading_collapsed=true
# # Data Load

# + hidden=true
survey_raw_df = pd.read_csv('./data/kaggle_survey_2020_responses.csv')
survey_raw_df.iloc[1:, 1:].to_csv('./data/kaggle_survey_2020_responses_no_header.csv', index=False)

# + hidden=true
survey_df = pd.read_csv('./data/kaggle_survey_2020_responses_no_header.csv')
survey_df.head()

# + [markdown] heading_collapsed=true
# # General Functions

# + hidden=true
questions_dict = {k:v[0] for k, v in survey_raw_df.head(1).to_dict().items()}

select_choice = []
for k, v in questions_dict.items():
    if 'Part' in k or 'OTHER' in k:
        select_choice.append(k)

select_choice_dict = {}
for k in select_choice:
    if '_Part' in k:     
        question_key = k.split('_Part')[0]
    elif '_OTHER' in k:     
        question_key = k.split('_OTHER')[0]
    question = questions_dict[k].split(' - ')[0]
    value = questions_dict[k].split(' - ')[-1]
    if question_key in select_choice_dict:
        select_choice_dict[question_key]['columns'][k] = value
    else:
        select_choice_dict[question_key] = {'question':question, 'columns':{k:value}, 'type':'multiple'}


multiple_choice_dict={}
multiple_choice = [a for a in questions_dict.keys() if a not in select_choice][1:]
for k in multiple_choice:
    multiple_choice_dict[k] = {'question':questions_dict[k], 'columns':{k:survey_df[k].unique().tolist()}, 'type':'unique'}
    

all_questions_dict = multiple_choice_dict.copy()
all_questions_dict.update(select_choice_dict)
all_questions_dict
# + code_folding=[0] hidden=true
def select_columns(dataset, app_dict_cols, to_dummy=True):
    
    questions = app_dict_cols.keys()
    
    multiple_choice = {k: v for k, v in app_dict_cols.items() if all_questions_dict[k]['type'] == 'multiple'}
    unique_choice = {k: v for k, v in app_dict_cols.items() if all_questions_dict[k]['type'] == 'unique'}
    
    # Multiple choice questions
    cols_per_question = {name: list(all_questions_dict[question]['columns'].keys()) for question, name in multiple_choice.items()}
    question_names = {item: name for name, sublist in cols_per_question.items() for item in sublist}

    names_per_question = [all_questions_dict[a]['columns'] for a in multiple_choice.keys()]
    question_options = {}
    for option_dict in names_per_question:
        question_options.update(option_dict)

    renamer = {}
    for k, v in question_names.items():
        new_name = v
        if k in question_options and question_options[k]:
            new_name += '__' + question_options[k].lower().strip().replace(' ', '').split('(')[0]
        renamer[k] = new_name

    multiple_df = dataset.loc[:, renamer.keys()].rename(columns=renamer)
    multiple_df.index = dataset.index
    for c in multiple_df.columns.tolist():
        multiple_df.loc[:, c] = (~multiple_df[c].isnull()).astype(int)
        
    
    # Single choice questions
    
    df = dataset.loc[:,unique_choice.keys()].rename(columns=unique_choice)
    
    for c in df.columns.tolist():
        df.loc[:, c] = df[c].str.split('(', expand=True).iloc[:, 0].str.lower().str.strip().str.replace(' ', '')
    single_df = pd.get_dummies(df, prefix_sep='__')
    single_df.index = survey_df.index
    
    final_df = pd.concat([multiple_df, single_df], axis=1)
        
    return final_df

# + [markdown] heading_collapsed=true
# # Identity

# + [markdown] hidden=true
# ## Data Prep

# + hidden=true
identity_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# + hidden=true
identity_raw_df = survey_df.loc[:, identity_questions]
identity_raw_df.columns = ['age', 'gender', 'country', 'schooling', 'profession']
identity_raw_df.head()


# + hidden=true
def str_normalize(x):
    return unidecode.unidecode(x).lower().replace(' ', '_').split(',')[0].strip()

identity_raw_df.loc[:, 'country_key'] = identity_raw_df['country'].apply(lambda x: str_normalize(x))

# + hidden=true
country_rematch = {'IR': 'Iran', 'TW': 'Taiwan', 'KR':'South Korea', 'KP':'Republic of Korea', 'RU': 'Russia'}

# + hidden=true
countries_infos = pd.read_csv('./data/countries_info.csv')

countries_infos.loc[:, 'name_adj'] = countries_infos.apply(lambda x: country_rematch[x['alpha-2']] if x['alpha-2'] in country_rematch else x['name'], axis=1)

countries_infos.loc[:, 'country_key'] = countries_infos['name_adj'].apply(lambda x: str_normalize(x))
countries_infos.head()

# + hidden=true
identity_df = pd.merge(left=identity_raw_df, 
                       right=countries_infos.loc[:, ['region', 'sub-region', 'country_key']],
                       on='country_key',
                       how='left'
                      )

# + hidden=true
all_ages = identity_df['age'].sort_values().unique().tolist()
all_ages_dict = {all_ages[i]:i for i in range(0, len(all_ages))}

identity_df.loc[:, 'age_adj'] = identity_df['age'].apply(lambda x: all_ages_dict[x])
identity_df.head()

# + [markdown] heading_collapsed=true hidden=true
# ## Cluster

# + hidden=true
# K-prototypes
# K-modes
# 

id_cluster_df = identity_df.loc[:, ['gender', 'schooling', 'profession', 'sub-region', 'age_adj']]
id_cluster_df.head()

# + hidden=true
id_dummies_df = pd.get_dummies(id_cluster_df)
id_dummies_df.head()

# + hidden=true
id_dummies_df.shape

# + hidden=true
import prince

famd = prince.FAMD(
     n_components=39,
     n_iter=10,
     copy=True,
     check_input=True,
     engine='auto',       ## Can be "auto", 'sklearn', 'fbpca'
     random_state=42)

## Fit FAMD object to data 
famd = famd.fit(id_cluster_df) ## Exclude target variable "Churn"

famd_data = famd.row_coordinates(id_cluster_df)

# + hidden=true
np.sum(famd.explained_inertia_)

# + hidden=true
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(id_dummies_df)

# pca = PCA(n_components=0.8)
# pca_data = pca.fit_transform(scaled_data)

# pca_data.shap

# + hidden=true
inertias = {}
for k in range(1, 50):
    if k % 5 == 0:
        print(k)
    model = KMeans(n_clusters=k)
    model.fit(famd_data)
    inertias[k] = model.inertia_
    
inertia_df = pd.DataFrame.from_dict(inertias, orient='index').reset_index()
inertia_df.columns = ['cluster_size', 'inertia']

inertia_df.head()

px.line(inertia_df, x='cluster_size', y='inertia')

# + hidden=true
model = KMeans(n_clusters=5)
model.fit(famd_data)
labels = model.labels_

# + hidden=true
df = id_dummies_df.copy()
opps = {c: 'mean' for c in df.columns.tolist()}
df.loc[:, 'cluster'] = labels
df.groupby('cluster').agg(opps)
# -

# # Apparatus

# + [markdown] heading_collapsed=true
# ## Data Prep

# + hidden=true
app_dict_cols = {
 'Q7': 'lang',
 'Q9': 'ide',
 'Q10': 'notebook',
 'Q11': 'platform',
 'Q12': 'hardware',
 'Q14': 'viz',
 'Q36': 'deploy'
}

apparatus_raw_df = select_columns(survey_df, app_dict_cols)
apparatus_raw_df.head()

# + hidden=true
# To users who didn't signup all questions, lead them to None option
# for any of the questions in this subset, didn't signaled any of the options

cols_map = {}
for c in apparatus_raw_df.columns.tolist():
    dim = c.split('__')[0]
    if dim in cols_map:
        cols_map[dim].append(c)
    else:
        cols_map[dim] = [c]

apparatus_df = apparatus_raw_df.copy()
        
skippers_index = []
for dim, cols in cols_map.items():
    df = apparatus_raw_df.loc[:, cols].sum(axis=1).to_frame()
    df.loc[:, 'empty'] = (df.iloc[:,0] == 0)    
    none_indexes = df.loc[df['empty']].index.tolist()
    
    none_col =  dim + '__none'
    if none_col in apparatus_df.columns.tolist():
        apparatus_df.loc[apparatus_df.index.isin(none_indexes), none_col] = 1
    
    
apparatus_df.head()

# + hidden=true
groups = {}
for c in apparatus_df.columns.tolist():
    group = c.split('__')[0] 
    if group in groups:
        groups[group].append(c)
    else:
         groups[group] = [c]

# + [markdown] heading_collapsed=true
# ## Graph view

# + hidden=true
matrix = apparatus_df.corr()
matrix.index.name ='feature'
matrix.reset_index(inplace=True)

melt = pd.melt(matrix, id_vars=['feature'], value_vars=apparatus_df.columns.tolist())
melt.columns = ['origin', 'destination', 'corr']

melt.loc[:, 'corr_abs'] = melt['corr'].abs()

non_cyclical = melt.loc[melt['origin'] != melt['destination']].copy()

non_cyclical.loc[:, 'origin_type'] = non_cyclical['origin'].str.split('__', expand=True).iloc[:, 0]
non_cyclical.loc[:, 'origin_name'] = non_cyclical['origin'].str.split('__', expand=True).iloc[:, 1]
non_cyclical.loc[:, 'dest_name'] = non_cyclical['destination'].str.split('__', expand=True).iloc[:, 1]
non_cyclical.loc[:, 'connection_rank'] = non_cyclical.groupby(['origin_name'])['corr'].rank(ascending=False)

top_corr = non_cyclical.loc[non_cyclical['corr_abs'] > 0.25]
top_corr.head()

# + hidden=true
node_dict_raw = apparatus_df.sum().to_dict()
i = 0
node_dict = {}
for node, size in node_dict_raw.items():
#     if '__' in node:
#         name = node.split('__')[1]
#     else:
    name = node
        
    node_dict[name] = {'size':size, 'index':i}
    i+=1
    
node_ref = pd.DataFrame.from_dict(node_dict, orient='index').reset_index()
node_ref.columns = ['name', 'size', 'index']


# + hidden=true
# Custom function to create an edge between node x and node y, with a given text and width
def make_edge(x, y, text, width):
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'cornflowerblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')


# + hidden=true
graph = nx.Graph()

# + hidden=true
# Add node for each character
for node, infos in node_dict.items():
    if size > 0:
        graph.add_node(infos['index'], size = infos['size'])

# For each co-appearance between two characters, add an edge
for k, row in top_corr.to_dict(orient='index').items():
    # Only add edge if the count is positive
    start = node_dict[row['origin']]['index']
    end = node_dict[row['destination']]['index']
    graph.add_edge(start, end, weight = row['corr_abs'])
    
    
pos_ = nx.spring_layout(graph)
pos_df = pd.DataFrame.from_dict(pos_, orient='index')
pos_df.columns = ['x', 'y']
pos_df = pd.concat([pos_df, node_ref.set_index('index')], axis=1)

# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in graph.edges():
    
    if graph.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]
        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]
#         text   = '{}--{}: {}'.format(char_1, char_2, graph.edges()[edge]['weight'])
        text='',
        trace  = make_edge([x0, x1, None], [y0, y1, None], text, 
                           width = 2*graph.edges()[edge]['weight'])
    edge_trace.append(trace)
    



# # Make a node trace
# node_trace = go.Scatter(x         = pos_df['x'],
#                         y         = pos_df['y'],
#                         text      = pos_df['name'],
#                         textposition = "top center",
#                         textfont_size = 10,
#                         mode      = 'markers+text',
#                         hoverinfo = 'none',
#                         marker    = dict(
#                                          size  = pos_df['size'],
#                                          line  = None))

# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "top center",
                        textfont_size = 10,
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        marker    = dict(
                                         size  = [],
                                         line  = None))


# # For each node in midsummer, get the position and size and add to the node_trace
# for node in graph.nodes():
    
#     x, y = pos_[node]
#     node_trace['x'] += tuple([x])
#     node_trace['y'] += tuple([y])
#     node_trace['marker']['color'] += tuple(['blue'])
#     node_trace['marker']['size'] = tuple([5*graph.nodes()[node]['size']])
#     node_trace['text'] += tuple(['<b>' + str(node) + '</b>'])


# Customize layout
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', # transparent background
    plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
    xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
    yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
)
# Create figure
fig = go.Figure(layout = layout)
# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)
# Add node trace
fig.add_trace(node_trace)
# Remove legend
fig.update_layout(showlegend = False)
# Remove tick labels
fig.update_xaxes(showticklabels = False)
fig.update_yaxes(showticklabels = False)
# Show figure
fig.show()

# + hidden=true
top_corr.shape

# + hidden=true
import plotly.graph_objects as go
import networkx as nx

G = nx.random_geometric_graph(200, 0.125)


# -

# ## Cluster

# +
def test_cluster_size(dataset):
    inertias = {}
    for k in range(1, 25):
        if k % 5 == 0:
            print(k)
        model = KMeans(n_clusters=k)
        model.fit(dataset)
        inertias[k] = model.inertia_

    inertia_df = pd.DataFrame.from_dict(inertias, orient='index').reset_index()
    inertia_df.columns = ['cluster_size', 'inertia']
    
    return inertia_df
    
def mfa_method(n_components):
    mfa = MFA(groups = groups, n_components = n_components, n_iter = 5, random_state = 42)
    return mfa.fit_transform(apparatus_df)

size = apparatus_df.shape[1]
pace = math.floor(size/5)
inertias = []
for s in range(pace, size, pace):
    if s <= size:
        dataset = mfa_method(s)
        print('{} colunas - {:.0f}%'.format(s, 100*s/size))
        df = test_cluster_size(dataset)
        df.loc[:, 'num_cols'] = s
        inertias.append(df)
        
inertia_df = pd.concat(inertias, axis=0)
# -

fig = px.line(inertia_df, x='cluster_size', y='inertia', facet_col='num_cols')
fig.update_yaxes(matches=None)
fig.show()

# +
dataset = mfa_method(14)
model = KMeans(n_clusters=9)
model.fit(dataset)

labels = model.labels_

# +
cluster_df = apparatus_df.copy()

cols = cluster_df.columns.tolist()
op_dict = {c:['mean', 'median'] for c in cols}

cluster_df.loc[:, 'cluster_num'] = labels
cluster_df.loc[:, 'counter'] = 1

op_dict['counter'] = 'sum'

grouped = cluster_df.groupby('cluster_num').agg(op_dict)
grouped.columns = ['{}={}'.format(c[0], c[1]) for c in grouped.columns.tolist()]
melt = pd.melt(grouped.reset_index(), id_vars=['cluster_num'], value_vars=grouped.columns.tolist())
melt.loc[:, 'var_type'] = melt['variable'].str.split('=', expand=True).iloc[:, 1]
melt.loc[:, 'var'] = melt['variable'].str.split('=', expand=True).iloc[:, 0]
melt.loc[:, 'dim'] = melt['var'].str.split('__', expand=True).iloc[:, 0]
melt.loc[:, 'dim_value'] = melt['var'].str.split('__', expand=True).iloc[:, 1]

cluster_infos = melt.loc[melt['var_type'] == 'mean'].drop(columns=['variable'])
cluster_infos.loc[:, 'rank'] = cluster_infos.groupby(['var'])['value'].rank(ascending=False)
cluster_infos.head()

# +
clusters = cluster_infos['cluster_num'].unique().tolist()

all_clusters_summary = []
all_clusters_detail = []

for c in clusters:
    df = cluster_infos.loc[cluster_infos['cluster_num'] == c].sort_values(by='value', ascending=False)
    df.loc[:, 'rank_presence'] = df['value'].rank(ascending=False)
#     df = df.loc[df['value'] > 0.5]
    top = df.loc[((df['rank_presence'] <= 5) | (df['rank'] == 1)) & (df['value'] >= 0.1)]
    
    group = top.groupby(['dim'])['dim_value'].agg(lambda x: list(x)).to_frame().sort_index()
    
    group.columns = [c]
    all_clusters_summary.append(group)
    all_clusters_detail.append(top)
# -

all_clusters_df = pd.concat(all_clusters_summary, axis=1).sort_index()
all_clusters_df.head(7)

# 9 clusters
cluster_names = {
    0:'mathy', #Matlab
    1:'fresh', #No coding
    2:'brainy', # Deep Learning
    3:'pro', #Does it all
    4:'pirate', # RRRRRRr
    6:'roots', # Bash
    7:'dev', # Visual Studio
    8:'cloudy', # Cloud platforms
    9:'generalist' # Python with jupyter, in his local machine
}

# # Experience

# Practice/Praxis/Experience/
# - ML time
# - Frameworks
# - Algorithms
# - Computer Vision
# - NLP
# - Automation

all_questions_df = pd.DataFrame.from_dict(all_questions_dict, orient='index').reset_index()
all_questions_df.loc[:, 'order'] = all_questions_df['index'].apply(lambda x: int(str(x).split('_')[0][1:]))
all_questions_df.sort_values(by='order')

# +
app_dict_cols = {
 'Q15': 'time',
 'Q16': 'frame',
 'Q17': 'algo',
 'Q18': 'vision',
 'Q19': 'nlp',
 'Q14': 'viz',
 'Q36': 'deploy'
}

apparatus_raw_df = select_columns(survey_df, app_dict_cols)
apparatus_raw_df.head()

# +
# import numpy as np
# from kmodes.kmodes import KModes

# # random categorical data
# data = np.random.choice(20, (100, 10))

# km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

# clusters ExperienceExperience= km.fit_predict(data)

# # Print the cluster centroids
# print(km.cluster_centroids_)
# -

pd.DataFrame(data)
