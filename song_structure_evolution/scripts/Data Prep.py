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

# ## Setup

import pandas as pd
import wikipedia
from bs4 import BeautifulSoup
from unidecode import  unidecode
from urllib.parse import unquote
import requests
import urllib.request
import numpy as np

# ## Links to all year-end lists of billboard

all_years = [f for f in range(1946, 2021)]
charts_df = pd.DataFrame({'year':all_years})
charts_df.loc[charts_df['year'] >= 1959, 'wikipedia_url'] = charts_df['year'].apply(lambda x: f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{x}")
charts_df.loc[charts_df['year'].between(1956, 1958), 'wikipedia_url'] = charts_df['year'].apply(lambda x: f"https://en.wikipedia.org/wiki/Billboard_year-end_top_50_singles_of_{x}")
charts_df.loc[charts_df['year'].between(1949, 1955), 'wikipedia_url'] = charts_df['year'].apply(lambda x: f"https://en.wikipedia.org/wiki/Billboard_year-end_top_30_singles_of_{x}")
charts_df.loc[charts_df['year'] < 1949, 'wikipedia_url'] = charts_df['year'].apply(lambda x: f"https://en.wikipedia.org/wiki/Billboard_year-end_top_singles_of_{x}")

year_end_urls = charts_df.set_index('year')['wikipedia_url'].to_dict()

# ## Extract Billboard year-end charts from Wikipedia

# +
# Extract billboard charts
extract = False

if extract:
    for year, url in year_end_urls.items():
    #     year = 2020
    #     url = year_end_urls[year]
        s = requests.Session()
        response = s.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        song_charts = soup.find_all('table', {'class':'wikitable sortable'})[0]

        if year == 2020: 
            header = ['song', 'artists']
        else:
            header = [f.text.replace('\n', '') for f in song_charts.find_all('th', {'scope':'col'})]


        col_size = len(header)
        table_values = song_charts.find_all('td')
        values_dict = {}
        for i in range(0, len(table_values)):
            index = f"{np.floor(i/col_size) + 1:.0f}"
            value = table_values[i].text.replace('\n', '').replace('"', '')
            pos = i % col_size
            if pos == 0:
                values_dict[index] = {header[pos]: value}
            else:
                values_dict[index][header[pos]] = value

        year_df = pd.DataFrame.from_dict(values_dict, orient='index').reset_index().iloc[:, -3:]
        year_df.columns = ['position', 'song_title', 'artists']

        year_df.to_csv(f"data/billboard_year_end_charts_per_year/{year}.csv", index=False)
