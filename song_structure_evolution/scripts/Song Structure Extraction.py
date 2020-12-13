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

# # Setup

import billboard
import pandas as pd
from datetime import datetime, timedelta, date
import os
import pandas as pd
import plotly.express as px

# # Data Extract

# ## Billboard

billboard.charts()


downloaded_years = [int(f.split('.csv')[0]) for f in os.listdir('./data/per_year')]
downloaded_years = []
fetch_years = [i for i in range(2010, 2021) if i not in downloaded_years]

fetch_years = [2020]

# +
# Extract data from billboard monthly
charts = []

for y in fetch_years:
    if y == 2020:
        date_fetch = date.today()
    else:
        date_fetch = date(year=y, month=12, day=31)
    date_str = date_fetch.strftime('%Y-%m-%d')
    year_charts = []
    while date_str  > '{}-01-01'.format(y):
        print(date_str)
        chart = billboard.ChartData('hot-100', timeout=60, date=date_fetch)
        
        current_date = datetime.strptime(chart.date, '%Y-%m-%d').date()
        date_fetch = current_date - timedelta(weeks=1)
        date_str = date_fetch.strftime('%Y-%m-%d')

        songs = []
        artists = []
        weeks = []
        positions = []

        for i in range(0, len(chart)):
            s = chart[i]
            songs.append(s.title)
            artists.append(s.artist)
            weeks.append(s.weeks)
            positions.append(i+1)

        charts_df = pd.DataFrame({'artist':artists, 'song':songs, 'weeks':weeks, 'positions':positions})
        charts_df.loc[:, 'week'] = chart.date

        year_charts.append(charts_df)
    
    year_chart_df = pd.concat(year_charts, axis=0)
    year_chart_df.to_csv('./data/per_year/{}.csv'.format(y), index=False)

# +
# charts_df.to_csv('2010_2015.csv', index=False)
# -

# ## Prep

all_files = [pd.read_csv('./data/per_year/{}'.format(f)) for f in os.listdir('./data/per_year')]
all_charts_df = pd.concat(all_files).reset_index()
all_charts_df.head()

all_charts_df = pd.read_csv('./data/per_year/2020.csv')
all_charts_df.head()

# +
songs_df = all_charts_df\
            .groupby(['artist', 'song'], as_index=False)\
            .agg({'weeks':['count', 'max'], 'week':['min', 'max'], 'positions':'min'})

songs_df.columns = ['artist', 'song', 'weeks_captured', 'total_weeks', 'first_week', 'last_week', 'peak'] 
# -

songs_df.sort_values(by='total_weeks', ascending=False)


