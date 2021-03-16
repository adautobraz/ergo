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
import json
import os
from pathlib import Path
from pytimeparse.timeparse import timeparse
import unicodedata
import re
import spacy
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from fuzzywuzzy import fuzz
import operator

stanza_nlp = stanza.Pipeline('pt') # initialize English neural pipeline

nlp = spacy.load("pt_core_news_md")
data_path = Path('./data/')

# + [markdown] heading_collapsed=true
# # Original

# + hidden=true
with open(data_path/'recipes.json', 'r') as f:
    recipes_raw = json.load(f)

# + hidden=true
recipes_raw_df = pd.DataFrame.from_dict(recipes_raw)

recipes_raw_df.loc[:, 'recipe_id'] = recipes_raw_df['_id'].apply(lambda x: x['$oid'])

recipes_df = pd.DataFrame.explode(recipes_raw_df, column='secao').iloc[:, 1:]

recipes_df.loc[:, 'section'] = recipes_df['secao'].apply(lambda x: x['nome']).str.strip()
recipes_df.loc[:, 'section_content'] = recipes_df['secao'].apply(lambda x: x['conteudo'])

recipes_df.loc[:, 'section_full_text'] = recipes_df['section_content'].apply(lambda x: ' '.join(x))


recipes_df.loc[:, 'section_type'] = 'ingredients'
recipes_df.loc[recipes_df['section'] == 'Modo de Preparo', 'section_type'] = 'prepare'
recipes_df.loc[recipes_df['section'] == 'Outras informações', 'section_type'] = 'other_info'

recipes_df.head()

# + hidden=true
ingr_section_df = recipes_df.loc[recipes_df['section_type'] == 'ingredients'].copy()

ingr_section_df.head()

# + hidden=true
from unidecode import unidecode


# + hidden=true
def split_by_quants(all_ingredients, to_print=False):
    
    clean_str = unidecode(all_ingredients)\
                    .replace('⁄','/')\
                    .replace('¹/²', '1/2')\
                    .replace('½', '1/2')
    
    all_caps = re.findall(r'[A-Z]+(?![a-z])', clean_str)
    for a in all_caps:
        clean_str = clean_str.replace(a, '')     
    
    line_start_groups = re.findall('([0-9 ]*[0-9/]+[a-zA-Z]* [Kg]*)|([A-Z]+[a-z-]* )', clean_str)
    
    split_terms = [l[1] if l[1] else l[0] for l in line_start_groups]
     
    if to_print: print(clean_str, split_terms)
        
    ing_list = []

    full_string = clean_str
    found_str = ''
    
    i = 0
    start_index = i
    end_index = start_index + 1
    while start_index < len(split_terms):
        
#         if to_print: print(p)

        search_str = full_string.replace(found_str, '')
        start = split_terms[start_index]
        if to_print: print(start)
            
#                 print(search_str)
#                 print('Start:',start)

        if start_index < len(split_terms) - 1:
            end = split_terms[end_index]
            match_regex = f'{re.escape(start)}(.*?){re.escape(end)}'
#             print(match_regex)
            matches = re.findall(match_regex, search_str)
#             print(matches)
            if matches:
                desc = matches[0]
            else:
                desc = ''   
    
            found_str += start + desc

            aux_dict = {'value': start, 'desc': desc}

        else:
#             print(start)
            desc = search_str.split(f'{start}')[1]
            aux_dict = {'value': start, 'desc': desc}
            
        start_index += 1
        end_index += 1
            
        ing_list.append(aux_dict)

    return ing_list
    
    
ingr_section_df.loc[:, 'ingredient_list'] = ingr_section_df['section_full_text'].apply(lambda x: split_by_quants(x, False))
 
ingredients_df = pd.DataFrame.explode(ingr_section_df, column='ingredient_list').reset_index().iloc[:, 1:]

ingredients_df = pd.concat([ingredients_df, ingredients_df['ingredient_list'].apply(pd.Series)], axis=1)

ingredients_df.head()

# ingredients_df['section_full_text'].value_counts()

# + hidden=true
ingredients_df.loc[ingredients_df['desc'] == '']['value'].value_counts().head(10)

# + hidden=true



# + hidden=true
# ingredients_df.head(15)

ingredients_df.loc[ingredients_df['value'] == 'Massa ']

# + [markdown] heading_collapsed=true
# # Tastemade

# + hidden=true
raw_recipes_path = data_path/'Tastemade/raw_recipes'
recipes_files_json = os.listdir(raw_recipes_path)

all_recipes = []
for r in recipes_files_json:
    with open(raw_recipes_path/r, 'r') as f:
        recipe_dict = json.load(f)
        all_recipes.append(recipe_dict)

# + hidden=true
recipes_df = pd.DataFrame(all_recipes)

recipes_df = pd\
            .DataFrame.explode(recipes_df, column='recipes')\
            .reset_index()\
            .rename(columns={'index':'page_id'})

recipe_dict_expl = recipes_df['recipes'].apply(pd.Series).iloc[:, 1:]
recipes_df = pd.concat([recipes_df, recipe_dict_expl], axis=1)
recipes_df.index.name = 'recipe_id'
# recipe_df.to_csv(data_path/'Tastemade/all_recipes.csv', index=True)

# Remove urls with no recipes
recipes_df = recipe_df.loc[~recipes_df['recipes'].isnull()]
recipes_df.head()

# + hidden=true
recipe_sections = pd.DataFrame.explode(recipes_df, column='sections')\
                    .reset_index()\
                    .rename(columns={'index':'section_id'})

section_dict_expl = recipe_sections['sections'].apply(pd.Series)
sections_df = pd.concat([recipe_sections, section_dict_expl], axis=1)
sections_df.head()

# + hidden=true
# Generate pivoted 
all_dfs = []
for s in sections_df['type'].unique().tolist():
    df = sections_df.loc[sections_df['type'] == s].set_index('recipe_id').loc[:, ['values']]
    df.columns = [s]
    all_dfs.append(df)
    

sections_pvt_df = pd.concat(all_dfs, axis=1) 
sections_pvt_df.head()

# + hidden=true
# Pivot sections of recipes
recipe_info = sections_pvt_df.loc[(sections_pvt_df['instructions'].apply(lambda x: True if x else False)) |
                                  (~sections_pvt_df['ingredients'].isnull())
                                 ]

infos = pd.DataFrame.explode(recipe_info, column='infos').loc[:, ['infos']]
infos.loc[:, 'dim'] = infos['infos'].str.split(':', expand=True).iloc[:, 0]
infos.loc[:, 'dim_adj'] = infos['dim'].map({'Porções':'portions', 'Preparação': 'preparing_time', 'Cozimento':'cooking_time'})

infos.loc[:, 'dim_value'] = infos['infos'].str.split(':', expand=True).iloc[:, 1]

valid_infos = infos.loc[~infos['dim_adj'].isnull()]

pvt_infos = pd.pivot_table(valid_infos.reset_index('recipe_id'), 
                           index=['recipe_id'], 
                           columns='dim_adj', 
                           values='dim_value',
                           aggfunc = lambda x: ' '.join(x)
                          ).fillna('')

pvt_infos.loc[:, 'portions'] = pvt_infos['portions'].apply(lambda x: re.findall(r"([0-9]*)", str(x))[0])
pvt_infos.loc[:, 'cooking_time'] = pvt_infos['cooking_time'].apply(lambda x: timeparse(x)/60 if x else '')
pvt_infos.loc[:, 'preparing_time'] = pvt_infos['preparing_time'].apply(lambda x: timeparse(x)/60 if x else '')

# + hidden=true
all_recipes = pd.concat([recipes_df.loc[:, ['page_id', 'url', 'title', 'tags']], 
                         pvt_infos,
                         sections_pvt_df.loc[:, ['ingredients', 'instructions']]], axis=1)

all_recipes = all_recipes\
                .loc[(~all_recipes['ingredients'].isnull()) & (~all_recipes['instructions'].isnull())]\
                .fillna('')

all_recipes.head()

# + hidden=true
all_recipes.to_csv(data_path/'Tastemade/all_recipes.csv', index=True)

# + hidden=true
import unicodedata
# -

# # Ingredients

# +
all_recipes = pd.read_csv(data_path/'Tastemade/all_recipes.csv').fillna('')

all_recipes.loc[:, 'ingredients'] = all_recipes['ingredients'].apply(lambda x: eval(x))


ingredients_df = pd.DataFrame.explode(all_recipes, column='ingredients').loc[:,['recipe_id', 'title', 
                                                                                'tags', 'cooking_time', 
                                                                                'preparing_time', 'ingredients']].reset_index().iloc[:, 1:]

ingredients_df.index.name = 'ingredient_id'

ingredients_df.reset_index(inplace=True)


# +
def remove_numbers(numbers, text):
    final_text = text
    if numbers:
        for n in numbers:
            final_text = final_text.replace(n, '')
    return final_text.strip()


def extract_ingredient(text):    
    ingredient = ''
    # Xícaras de chá
    match = re.findall(r"xícaras* [de chá]*[de]+ ([^+]*)", str(text))
    if match:
        ingredient = match[0].strip()
    
    # Colher de chá/sopa
    match = re.findall(r"colher[es]* [de]* [chá|sopa]* [de]+ ([^+]*)", str(text))
    if not ingredient and match:
        ingredient = match[0].strip()
        
    # Algo com de, mas letra maiúscula
    match = re.findall(r"(^[A-Z].* [de]+.*)", str(text))
    if not ingredient and match:
        ingredient = match[0].strip() 

    # Algo com de
    match = re.findall(r"[ de ]+ ([^+]*)", str(text))
    if not ingredient and match:
        ingredient = match[0].strip()
    
    # Tirando as quantidades
    adj_text = text
    if ' ml ' in adj_text:
        adj_text = adj_text.replace(' ml ', 'ml ')
    if ' g ' in adj_text:
        adj_text = adj_text.replace(' g ', 'g ')        
        
    match = re.findall(r"[0-9⁄/-]*[g|ml]* *([^\d\+]*)", str(adj_text))
    if not ingredient and match:
        ingredient = match[0].strip()
    
    ing_adj = ' '.join([f for f in re.findall(r'[^\d⁄ ]*', ingredient) if f])\
                .replace('a gosto', '')\
                .replace('à gosto', '')\
                .replace('agosto', '')\
                .strip()

        
    return ing_adj

def remove_details(text, details):
    final_text = text
    for d in details:
        final_text = final_text.replace(d, '')
    return final_text.strip()


ingredients_df.loc[:, 'total_ingredients'] = ingredients_df.groupby(['recipe_id'])['ingredient_id'].transform('count')

ingredients_df.loc[:, 'description_raw'] = ingredients_df['ingredients'].apply(lambda x: unicodedata.normalize('NFKC', x).strip())
ingredients_df.loc[:, 'details'] = ingredients_df['description_raw'].apply(lambda x: re.findall('\(.*\)', x))
ingredients_df.loc[:, 'description'] = ingredients_df.apply(lambda x: remove_details(x['description_raw'], x['details']), axis=1)

ingredients_df.loc[:, 'desc_ing'] = ingredients_df['description'].apply(lambda x: extract_ingredient(x)).str.lower()
ingredients_df.head()

# + code_folding=[]
# ingredients_df.loc[ingredients_df['desc_ing'].str.contains('azeite')]#['desc_ing'].value_counts().head(25)

# +
# # Get lemmas for words used as ingredients

# test_df = ingredients_df

# unique_terms = list(set(test_df['desc_ing'].tolist()))

# lemma_ingredients = {}
# print(len(unique_terms))
# for i in range(0, len(unique_terms)):
#     if i % 100 == 0:
#         print(i)
#     string = unique_terms[i]
#     if string not in lemma_ingredients:
#         doc = stanza_nlp(string)
#         lemmatized = []
#         for sent in doc.sentences:
#             for word in sent.words:
#                 lemmatized.append({'lemma':word.lemma, 'POS':word.pos})
#         lemma_ingredients[string] = lemmatized

# with open(data_path/'Tastemade/ingredients_lemma_dict.json', 'w') as f:
#     json.dump(lemma_ingredients, f)

# +
# Read lemmas, substitute them and get words of each 
with open(data_path/'Tastemade/ingredients_lemma_dict.json', 'r') as f:
    lemma_ingredients = json.load(f)
    
noun_ingredients = {}
for k, v in lemma_ingredients.items():
    noun_ingredients[k] = ' '.join([w['lemma'] for w in v if w['POS'] == 'NOUN'])

# noun_ingredients = {}
# for k, v in lemma_ingredients.items():
#     noun_ingredients[k] = ' '.join([w['lemma'] for w in v])
    
ingredients_df.loc[:, 'ing_lemma'] = ingredients_df['desc_ing'].apply(lambda x: noun_ingredients[x])

# +
pt_stopwords = nltk.corpus.stopwords.words('portuguese')

vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=pt_stopwords, ngram_range=(1,2), use_idf=False)
vectors = vectorizer.fit_transform(ingredients_df['ing_lemma'].tolist())

feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()

merge_df = pd.DataFrame(denselist, columns=feature_names).unstack().to_frame().reset_index()
merge_df.columns = ['word', 'ingredient_id', 'value']
merge_df = merge_df.loc[merge_df['value'] > 0]
# -

top_terms = merge_df\
                .groupby(['word']).agg({'value':'sum'})\
                .sort_values(by='value', ascending=False).iloc[:100].index.tolist()



# +
def top_ingredient_match(text, top_terms):
    best_match={}
    for t in top_terms:
        match = fuzz.partial_ratio(text, t)
        if fuzz.partial_ratio(text, t) >= 70:
            best_match[t] = match
        
    if best_match:
        return max(best_match.items(), key=operator.itemgetter(1))[0]
    
    else:
        return 'other'

ingredients_df.loc[:, 'ingredient_top_match'] = ingredients_df['ing_lemma'].apply(lambda x: top_ingredient_match(x, top_terms))
# -

ingredients_df['ingredient_top_match'].value_counts().head(20)

# +
# %%time
test_df = ingredients_df.sample(1000)
# ingredients_df =pd.DataFrame.explode(ingredients_df, column='description')
test_df.loc[:, 'ing'] = test_df['description'].apply(lambda x: extract_ingredient(x))


# ingredients_df.loc[:, 'amount'] = ingredients_df['description'].apply(lambda x: re.findall(r'^(.*?)[A-Za-z ]', x))

# ingredients_df.loc[:, 'desc_no_number'] = ingredients_df.apply(lambda x: remove_numbers(x['amount'],x['description']), axis=1)

# ingredients_df.loc[:, 'numbers'] = ingredients_df.apply(lambda x: remove_numbers(x['amount'],x['description']), axis=1)


# ingredients_df.loc[:, 'ingredient'] = ingredients_df.apply(lambda x: x['description'].replace(x['amount'], '').split('de')[-1], axis=1)
# -

test_df.head()
