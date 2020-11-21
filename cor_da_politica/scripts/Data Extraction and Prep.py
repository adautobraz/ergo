# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

import pandas as pd
import wikipedia
from bs4 import BeautifulSoup
from unidecode import  unidecode
from urllib.parse import unquote
import requests
import urllib.request
import numpy as np

# ## Extract from Wikipedia

url = 'https://pt.wikipedia.org/wiki/Lista_de_partidos_pol%C3%ADticos_do_Brasil'
s = requests.Session()
response = s.get(url, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')

# +
partidos = soup.find_all('table', {'class':'wikitable sortable'})[0]

posicoes_table = soup.find_all('table', {'class':'wikitable sortable'})[1]
# -

# ### Posicoes infos

# +
posicoes_table

header = [th.text.rstrip() for th in posicoes_table.find_all('th') if th.text.rstrip()]

col_size = len(header)
table_values = posicoes_table.find_all('td')
values_dict = {}
for i in range(0, len(table_values)):
    index = str(np.floor(i/col_size))
    value = table_values[i].text.replace('\n', '')
    if i % col_size == 0:
        link = table_values[i].find('a', href=True)['href']
        values_dict[index] = [link, value]
    else:
        values_dict[index].append(value)

table = list(values_dict.values())

header.insert(0, 'link')

# +
posicoes_df = pd.DataFrame(table, columns=header)

posicoes_df.columns = ['link', 'name', 'fernandes', 'coppedge', 'globo', 'bbc', 'couto', 'folha', 'congresso']

posicoes_df = pd.melt(posicoes_df, id_vars=['name'], value_vars=posicoes_df.iloc[:, 2:].columns.tolist())


posicoes_df.loc[:, 'centro'] = posicoes_df['value'].str.contains('centro').astype(int)
posicoes_df.loc[:, 'direita'] = posicoes_df['value'].str.contains('direita').astype(int)
posicoes_df.loc[:, 'esquerda'] = posicoes_df['value'].str.contains('esquerda').astype(int)

posicoes_df = pd.melt(posicoes_df, id_vars=['name'], value_vars=['centro', 'direita', 'esquerda'], 
                      var_name='position', value_name='mentions')

posicoes_df = posicoes_df\
                .groupby(['name', 'position'], as_index=False)\
                .agg({'mentions':'sum'})\
                .sort_values(by='position', ascending=False)


posicoes_df.loc[:, 'rank'] = posicoes_df.groupby(['name'])['mentions'].rank(method='first', ascending=False)

posicoes_df.loc[:, 'code'] = posicoes_df['name'].str.upper()
posicoes_df.loc[posicoes_df['name'].isin(['PCdoB']), 'code'] = posicoes_df['name']

top_position = posicoes_df.loc[posicoes_df['rank'] == 1, ['code', 'position']]

top_position.head()
# -

# ### Partidos info

# +
header = [th.text.rstrip() for th in partidos.find_all('th') if th.text.rstrip()]

col_size = len(header)
table_values = partidos.find_all('td')
values_dict = {}
for i in range(0, len(table_values)):
    index = str(np.floor(i/col_size))
    value = table_values[i].text.replace('\n', '')
    if i % col_size == 0:
        link = table_values[i].find('a', href=True)['href']
        values_dict[index] = [link, value]
    else:
        values_dict[index].append(value)

table = list(values_dict.values())

header.insert(0, 'link')

# +
partidos_raw_df = pd.DataFrame(table, columns=header)

partidos_raw_df.columns = ['link', 'name', 'initials', 'electoral_num', 'affiliates', 'creation_date', 'register_date', 'current_president']


partidos_raw_df.loc[:, 'affiliates'] = partidos_raw_df['affiliates']\
                                        .str.encode('ascii', 'ignore')\
                                        .str.decode('ascii')

partidos_raw_df.loc[:, 'creation_date'] = partidos_raw_df['creation_date']\
                                            .str.split('[', expand=True).iloc[:, 0]

partidos_raw_df.loc[:, 'register_date'] = partidos_raw_df['register_date']\
                                            .str.split('[', expand=True).iloc[:, 0]

partidos_raw_df.loc[:, 'initials_ok'] = ~partidos_raw_df['initials']\
                                            .str.contains('nenhuma')

partidos_raw_df.loc[:, 'code'] = partidos_raw_df['name'].str.upper()
partidos_raw_df.loc[partidos_raw_df['initials_ok'], 'code'] = partidos_raw_df['initials']
                                        

partidos_raw_df.loc[:, 'page_raw'] = partidos_raw_df['link'].str.split('/', expand=True).iloc[:, 2]

partidos_raw_df.loc[:, 'page'] = partidos_raw_df['page_raw'].apply(lambda x: unquote(x))

partidos_raw_df.head()
# -

# ## Merge info

# +
partidos_df = partidos_raw_df.loc[:, ['name', 'electoral_num', 'affiliates', 'creation_date', 'register_date', 
                                     'current_president', 'code', 'page']]

partidos_df = pd.merge(left=partidos_df, right=top_position, on='code', how='left')
partidos_df.to_csv('./data/partidos_infos.csv', index=False)
partidos_df.head(40)
# -

# ## Get logos

partidos_df = pd.read_csv('./data/partidos_infos.csv')
partidos_df.head()

# +
import wikipedia
import requests
import json

WIKI_REQUEST = 'http://pt.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

def get_wiki_image(search_term):
    try:
        result = wikipedia.search(search_term, results = 1)
        wikipedia.set_lang('pt')
        wkpage = wikipedia.WikipediaPage(title = result[0])
        title = wkpage.title
        response  = requests.get(WIKI_REQUEST+title)
        json_data = json.loads(response.text)
        img_link = list(json_data['query']['pages'].values())[0]['original']['source']
        return img_link        
    except:
        return None
    
get_wiki_image('Partido_da_Social_Democracia_Brasileira')

# +
partidos_dict = partidos_df.to_dict(orient='index')

for k, v in partidos_dict.items():
    key = v['code']
    wiki_image = get_wiki_image(v['page'])
    if wiki_image:
        print(key)   
        file_format = wiki_image.split('.')[-1]
        filename = './data/logos/{}.{}'.format(key, file_format)
        urllib.request.urlretrieve(wiki_image, filename)
        
    else:
        print(key)
# -

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os
from shutil import copyfile

path = './data/logos/'
for file in os.listdir(path + 'raw/'):
    in_file = '{}raw/{}'.format(path, file)
    out_file = '{}png/{}.png'.format(path, file.split('.')[0])
        
    if '.svg' in in_file:
#         cairosvg.svg2png(url=in_file, write_to=out_file)
        drawing = svg2rlg(in_file)
        renderPM.drawToFile(drawing, out_file, fmt="PNG")
        
    else:
        copyfile(in_file, out_file)


# +
partidos = partidos_df['code'].tolist()
files = [f.split('.')[0] for f in os.listdir(path + 'png/')]

not_found = [p for p in partidos if p not in files]
not_found
