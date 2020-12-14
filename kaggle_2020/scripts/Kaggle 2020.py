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
import plotly.express as px
import itertools
import enchant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import unidecode

d = enchant.Dict("en_US")
pd.set_option('max_columns', None)

# + [markdown] heading_collapsed=true
# # Story Map

# + [markdown] hidden=true
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
# - Time of coding
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

# + hidden=true
# A Decolonial look to Data Science
# The Brazilian Way/ Tropical
# ALICE in DataScienceLand: Understanding Apparatus, Learning, Identity, Context and Experience in the Data Sciece Community

# + hidden=true
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
# -

# # Data Load

survey_raw_df = pd.read_csv('./data/kaggle_survey_2020_responses.csv')
survey_raw_df.iloc[1:, 1:].to_csv('./data/kaggle_survey_2020_responses_no_header.csv', index=False)

survey_df = pd.read_csv('./data/kaggle_survey_2020_responses_no_header.csv')
survey_df.head()

questions_dict

# +
questions_dict = {k:v[0] for k, v in survey_raw_df.head(1).to_dict().items()}

select_choice = []
for k, v in questions_dict.items():
    if len(k.split('_')) > 1:
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
        select_choice_dict[question_key] = {'question':question, 'columns':{}}


multiple_choice_dict={}
multiple_choice = [a for a in questions_dict.keys() if a not in select_choice][1:]
for k in multiple_choice:
    multiple_choice_dict[k] = {'question':questions_dict[k], 'columns':{k:''}}
    
multiple_choice_dict

all_questions_dict = multiple_choice_dict.copy()
all_questions_dict.update(select_choice_dict)
all_questions_dict
# -

# # Identity

# ## Data Prep

identity_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

identity_raw_df = survey_df.loc[:, identity_questions]
identity_raw_df.columns = ['age', 'gender', 'country', 'schooling', 'profession']
identity_raw_df.head()


# +
def str_normalize(x):
    return unidecode.unidecode(x).lower().replace(' ', '_').split(',')[0].strip()

identity_raw_df.loc[:, 'country_key'] = identity_raw_df['country'].apply(lambda x: str_normalize(x))
# -

country_rematch = {'IR': 'Iran', 'TW': 'Taiwan', 'KR':'South Korea', 'KP':'Republic of Korea', 'RU': 'Russia'}

# +
countries_infos = pd.read_csv('./data/countries_info.csv')

countries_infos.loc[:, 'name_adj'] = countries_infos.apply(lambda x: country_rematch[x['alpha-2']] if x['alpha-2'] in country_rematch else x['name'], axis=1)

countries_infos.loc[:, 'country_key'] = countries_infos['name_adj'].apply(lambda x: str_normalize(x))
countries_infos.head()
# -

identity_df = pd.merge(left=identity_raw_df, 
                       right=countries_infos.loc[:, ['region', 'sub-region', 'country_key']],
                       on='country_key',
                       how='left'
                      )

# +
all_ages = identity_df['age'].sort_values().unique().tolist()
all_ages_dict = {all_ages[i]:i for i in range(0, len(all_ages))}

identity_df.loc[:, 'age_adj'] = identity_df['age'].apply(lambda x: all_ages_dict[x])
identity_df.head()

# + [markdown] heading_collapsed=true
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

# + hidden=true
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

all_questions_dict['Q7']['columns']

apparatus_questions = ['Q7', 'Q9', 'Q10', 'Q11', 'Q12', 'Q14', 'Q36']
apparatus_names = ['lang', 'ide', 'notebook', 'platform', 'hardware', 'viz', 'deploy']

# +
cols_per_question = [list(all_questions_dict[a]['columns'].keys()) for a in apparatus_questions]
cols = [item for sublist in cols_per_question for item in sublist]

names_per_question = [all_questions_dict[a]['columns'] for a in apparatus_questions]

cols_name

names_per_question
# -

apparatus_df = survey_df.loc[:, cols]
apparatus_df.columns = ['age', 'gender', 'country', 'schooling', 'profession']
apparatus_df.head()
