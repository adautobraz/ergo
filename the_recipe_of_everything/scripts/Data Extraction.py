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

# # Setup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pickle 
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import json
import os

data_path = Path('./data')

# + [markdown] heading_collapsed=true
# # Crawler dos htmls

# + hidden=true
chrome_options = Options()
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--window-size=1920x1080")

# + hidden=true
driver = webdriver.Chrome('./chromedriver')  # Optional argument, if not specified will search path.

# + hidden=true
driver = webdriver.Chrome(chrome_options=chrome_options, executable_path='./chromedriver')

# + hidden=true
url = "https://www.tastemade.com.br/receitas"
driver.get(url)
time.sleep(2)

# + hidden=true
# Percorrer site, obter urls

# i = 0
# while i <= 500:
#     if i%25 == 0:
#         print(i)
#     try:
#         required_button = driver.find_element_by_css_selector(".cyMhnU")
#         required_button.click()
#         time.sleep(1)
#         i += 1
#     except:
#         i += 500

# + hidden=true
elements = driver.find_elements_by_css_selector(".cDjZPS > a")
storyUrls = list(set([el.get_attribute("href") for el in elements]))

# + hidden=true
# with open(f'./data/craweled_recipes_tastemade', 'wb') as f:
#     pickle.dump(storyUrls, f)
# -

# # Extrair infos das páginas

with open(data_path/'Tastemade/craweled_recipes_tastemade', 'rb') as f:
    recipes = pickle.load(f)


def get_tastemade_page_dict(url):

    # Define url and get page
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    # Find recipes in url
    class_ = 'VideoRecipe__Container-sc-4pl27p-6 bHnvZH VideoRecipe h-recipe'
    recipes = soup.find_all('div', class_=class_)

    recipes_dict = {'url':url,'recipes':[], 'tags':[]}

    for r in recipes:

        title = ''
        title_search = r.find_all(class_="p-name")
        if title_search:
            title = title_search[0].get_text()

            r_dict = {'title':title}

            section_search = r.find_all(class_="VideoRecipe__SubContainer-sc-4pl27p-3 eeTolE")
            # if section_search:
            section_infos = []
            for s in section_search:

                other_info_search = s.find_all('div', class_="VideoRecipe__InfoItem-sc-4pl27p-2")
                ingredients_search = s.find_all('p', class_='p-ingredient')
                instructions_search = s.find_all('ol', class_="VideoRecipe__List-sc-4pl27p-5 dvHwdt recipe-steps-list e-instructions")

                if other_info_search:
                    dict_section = {'type':'infos', 'values':[]}
                    for info in other_info_search:
                        dict_section['values'].append(info.get_text())

                elif ingredients_search:
                    dict_section = {'type':'ingredients', 'values':[]}
                    for i in ingredients_search:
                        dict_section['values'].append(i.get_text())

                elif instructions_search:
                    dict_section = {'type':'instructions', 'values':[]}
                    for c in instructions_search[0].children:
                        dict_section['values'].append(c.text)
                else:
                    dict_section = {'type':'other', 'values':s.get_text()}

                section_infos.append(dict_section)


            r_dict['sections'] = section_infos

            recipes_dict['recipes'].append(r_dict)


    class_="VideoTags__TagContainer-sc-17ye0x5-0 dmdCEC hide-print"
    tags = soup.find_all('div', class_=class_)
    if tags:
        for t in tags[0].children:
                recipes_dict['tags'].append(t.get_text())

    return recipes_dict

# +
downloaded_recipes = os.listdir(data_path/'Tastemade/raw_recipes/')

recipes_to_download = [r for r in recipes if "{}.json".format(r.split('/')[-1]) not in downloaded_recipes]

for url in recipes_to_download:
    
    infos = get_tastemade_page_dict(url)
    key = url.split('/')[-1]
    with open(data_path/f'Tastemade/raw_recipes/{key}.json', 'w') as f:
        json.dump(infos, f)
# -

len(recipes_to_download)


