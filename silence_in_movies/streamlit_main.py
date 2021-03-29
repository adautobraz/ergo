# Setup
import streamlit as st

# Layout definitions
st.set_page_config(layout="wide")

import pandas as pd
from pathlib import Path
from PIL import Image
import string

from sources.data_load_functions import *
from sources.visualization_functions import *

# Data definitions
data_path = Path('./data')


# Load data
data_dict = load_data(data_path)
subs_df = data_dict['subs_df']
silences_df = data_dict['silences_df']
movies_df = data_dict['movies_df']
movies_melt = data_dict['movies_melt_df']
positions_df = data_dict['positions_df']
umap_df = data_dict['umap_df']


# Layout definitons
main_palette = px.colors.qualitative.Safe
sound_types = ['Silence', 'Dialogue', 'Other sounds']
sound_color_map = {sound_types[i]:main_palette[i] for i in range(0, len(sound_types))}
sound_color_map['Mute'] = px.colors.qualitative.Pastel2[-1]

general_config ={'displayModeBar':False}

cols_baseline=12
image_counter = -1
image_ref = string.ascii_uppercase
image = Image.open(data_path/'prep/image.jpg')



###### Title headline, subtitle
# st.markdown("<h1>The Sound of Movies</h1>", unsafe_allow_html=True)
center = pad_cols([cols_baseline])[0]
center.image(image)

# center.title('The (Not So) Silent Companion')
# center.header("A look at the sound of the internet's favorite 150 movies")

text = """
# The (Not So) Silent Companion
## A look at the use of sound in IMDB's top 150 movies
A visual essay by <b>Adauto Braz</b>
"""
center.markdown(text, unsafe_allow_html=True)
space_out(2)


###### Introduction
center = pad_cols([cols_baseline])[0]
text = """
### <b>Movies are just a sonic experience as they are a visual one.</b>

Be it accompanied with live music or lavishly designed soundtracks, 
the impact of sound on how we experience a movie is undeniable, despite our lack of
awareness to it.  

Just think about a horror movie. Though, sometimes, we try to escape from what's unraveling 
during a scene by blocking our vision - Yes, I have watched whole movies with my 
hands on my face and no, I'm not proud of it -, we usually forget to try to escape from sound.
That way, unsurprisingly, even though we're not looking at the screen at all, 
those creeping silences that announce something terrible is about to happen still 
gives us the chills anyway.

When considering this, it struck me just how much sound usage can vary from genre to genre.
We have action movies and their explosions, romantic comedies and their enormous amount of talking,
and even musicals and all their singing. Would there be a way to explore and understand these 
differences?
<br><br>
### <b>The movies and the sounds</b>
Luckily, there is! With audio and subtitle data from IMDB's top 150 movies
(the complete list of movies can be found 
<a href="https://www.imdb.com/chart/top/?ref_=nv_mv_250"> here</a>), we'll 
try to answer some of the most interesting questions about the sound of movies.

To fullfill our task, first we need to categorize sound. 
Due to the type of data we have, we'll separate it in three categories:
* <b>Dialogue</b>: We'll assume that everything that's on the subtitles is dialogue, even if it's song
lyrics or text that is not properly enouncitade on screen (as in the case of silent movies). We treat
the case of Closed Captions (descriptions of sounds between brackets) and remove them.
* <b>Silence</b>: Using the extracted sound files from the movies, we'll categorize as silence anything
that stays below the -40db threshold, for at least one whole second. If there's a clash between silence and subtitle,
we favor the limits of the subtitle, and trim the silence bit accordingly.
* <b>Other sounds</b>: Everything else. 

Here, to help you have a quick glimpse of all the movies we're analyzing, let's check some 
general statistics of how our data is distributed.
"""
center.markdown(text, unsafe_allow_html=True)

image_counter += 1
image_name = image_ref[image_counter]
fig = all_movies_similarity(umap_df)
fig.update_layout(
    legend_orientation='h',
    margin_r=0, 
    margin_b=0,
    height=400,
    title=f'<b>{image_name}</b> - All Top 150 IMDB movies, grouped by similarity')
center.plotly_chart(fig, use_container_width=True, config=general_config)
space_out(2)


text = """
On <b>{}</b>, to group movies, we consider each movie release year, 
sound distribution, associated genres and coloring technique.  
The dimensions are a projection, so the only real meaning is 
that closer points have more similar characteristics, which yields some interesting separations. 
Up left, for example, there seems to be the Animation group, followed, closely on its
right by the Adventure block. What other groups can you find?
""".format(image_name)
center.markdown(text, unsafe_allow_html=True)


pad_l = 3
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

image_counter += 1
image_name = image_ref[image_counter]

text = """
As it's possible to see on <b>{}</b>, on the top 150 of IMDB there's an imbalance, 
with most of the movies on the list being released on the last three decades.
Besides, most films (almost 60%) have a duration between 2 and 3 hours.
""".format(image_name)

left.markdown(text, unsafe_allow_html=True)

fig = all_movies_hist_summary(movies_df, main_palette)
fig.update_layout(
    margin_b=0,
    margin_r=0,
    margin_t=50,
    height=300,
    xaxis_title='',
    title=f'<b>{image_name}</b> - Distribution of movies:<br>duration and release year')

fig.update_xaxes(title='Duration',col = 1)
fig.update_xaxes(title='Year',col = 2)
fig.update_yaxes(range=[0, 70])

fig.for_each_annotation(lambda x: x.update(text=''))

right.plotly_chart(fig, use_container_width=True, config=general_config)

center = pad_cols([cols_baseline])[0]
text = """
To reduce the effects of such peculiarities, most of the following analysis will be 
computed considering relative measures, to make it possible to compare different
absolute values.

So then, regarding the sound of movies, what should one expect from the 'average' movie?
"""
center.markdown(text, unsafe_allow_html=True)
space_out(1)


### Sound per type
center = pad_cols([cols_baseline])[0]
text = """
### <b>What does a movie sound like?</b>
"""
center.markdown(text, unsafe_allow_html=True)

pad_l = 3
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

image_counter += 1
image_name = image_ref[image_counter]

text = """
On <b>{}</b>, we see that Dialogue is the number 1 sound on movies, followed closely by
Other sounds and with Silence at the distant last position.  

But, is there a difference in where these sounds are more prominent during the movie?
""".format(image_name)
left.markdown(text, unsafe_allow_html=True)

fig = sound_type_distribution_bar(movies_melt, sound_color_map)
fig.update_layout(
    margin_r=0, 
    margin_b=0,
    margin_t=120,
    title=f'<b>{image_name}</b> - What is the most common<br>sound in movies?')
right.plotly_chart(fig, use_container_width=True, config=general_config)
space_out(1)


### Sound per position
pad_l = 3
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

image_counter += 1
image_name = image_ref[image_counter]

text = """
On <b>{}</b>, we can see that usually in the beginning of a movie, Silences and 
Other sounds are very relevant. From 3% of the movie up until 80%, 
Dialogue takes the reigns and only loses them back to Other sounds in the final segment. 

To a 2 hour movie, this last 20% segment would represent the last 24 minutes of the movie - 
where usually the credits roll to the sound of some music related to the movie.
""".format(image_name)

left.markdown(text, unsafe_allow_html=True)

fig = sound_type_share_by__position(positions_df, sound_color_map)
fig.update_layout(
    height=400, 
    margin_r=0, 
    legend_orientation='h',
    legend_y=-0.3,
    title=f'<b>{image_name}</b> - Sound type evolution during movie')
right.plotly_chart(fig, use_container_width=True, config=general_config)
space_out(1)



### Top movies

image_counter += 1
image_name = image_ref[image_counter]

center = pad_cols([cols_baseline])[0]
text = """
Despite these results, there are some movies that stay far away from the average sound 
distribution - what we call outliers. Below, let's check the 5 top movies of each 
sound category.
"""

center.markdown(text, unsafe_allow_html=True)

fig = top_movies_by__type(movies_melt, sound_color_map)  
fig.update_layout(
    title=f'<b>{image_name}</b> - Top 5 movies, for each sound type',
    margin_r=20,
    margin_l=0
)
# fig.update_xaxes(range=[0,150], dtick=100)
center.plotly_chart(fig, use_container_width=True, config=general_config)

text = """
On <b>{}</b> we can highlight the top movie of each category.
* <b>Silence</b>: <i>Ran (1985)</i>  
Akira Kurosawa's story about medieval Japan, based on Shakespeare's King Lear has a lot of silence
(37%), with its longest silence running for 84 seconds straight.
* <b>Dialogue</b>: <i>Hamilton (2020)</i>   
The film recording of the Broadway sensation about America's ten-dollar Founding Father, 
has a ridiculously high word-per-minute ratio, which explains why so much of it (80%) is just 
plain dialogue - in this case sang beautifully and rapidly by its amazing cast.
* <b>Other sounds</b>: <i>Modern Times (1936)</i>   
Charlie Chaplin's silent comedy about the struggles of Little Tramp to 
survive the industrialized world is a silent classic, even if "silent" does not mean what its supposed
to, with 93% of it being attributed to Other Sounds, or in this case, the amazing musical score, 
composed by Chaplin himself.
""".format(image_name)

center.markdown(text, unsafe_allow_html=True)

expander = center.beta_expander('Wanna see a specific movie sound type distribution over its duration? Click here!')
with expander:
    all_movies = sorted(movies_df['title'].unique().tolist())
    movie_chosen = expander.selectbox('Movie', all_movies)
    df = positions_df.loc[positions_df['title'] == movie_chosen]
    fig = sound_type_per_position(df, sound_color_map, 5)
    fig.update_layout(legend_orientation='h', legend_y=-0.2, margin_r=0, height=400)
    expander.plotly_chart(fig, use_container_width=True, config=general_config)

space_out(2)



### Color movies, black and white

image_counter += 1
image_name = image_ref[image_counter]

center = pad_cols([cols_baseline])[0]
text = """
### <b>Color x Black and White</b>
As we se, Modern Times relied much more on music score than on dialogue, for example.
Since the rise of movies with color and the integration of sound into the film score, did 
that change?   
"""
center.markdown(text, unsafe_allow_html=True)

pad_l = 4
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

color_num_dict = movies_df['color_type'].value_counts().to_dict()
text = """<br>
As it turns out, it seems that Modern Times is actually the exception in this case. On <b>{}</b>,
comparing all {} black and white movies to all the {} movies in color, we see that
movies in B&W spend more time on dialogue than using other sounds, the opposite of movies with
color. When considering silence, there not seems to exist a significant difference among 
movie types.
""".format(image_name, color_num_dict['Black and White'], color_num_dict['Color'])

left.markdown(text, unsafe_allow_html=True)

fig = sound_share_by__type__color(movies_melt, main_palette)
fig.update_layout(
    height=400,
    margin_b=0,
    margin_r=0, 
    title=f'<b>{image_name}</b> - Color x B&W<br>Is there a difference in sounds?')
right.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})


center = pad_cols([cols_baseline])[0]
text = """<br>
Could this difference just be a sign of the times of when these movies were created?
"""
center.markdown(text, unsafe_allow_html=True)

space_out(1)


### Time evolution

image_counter += 1
image_name = image_ref[image_counter]

center = pad_cols([cols_baseline])[0]
text = """
### <b>A Century of sound</b>
Given that we have movies representing very different periods of time, is there a trend
on how each type of sound has being used from 1920 until 2020?  
"""
center.markdown(text, unsafe_allow_html=True)

pad_l = 4
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

text = """
As it's possible to see in <b>{}</b>, since 1960, the use of each sound category 
has been fairly stable, apart from 2020 - due to <i>Hamilton</i>.

Before 1940, Other sounds were much more prominent than Dialogue,
due to the silent movie era. For out of five movies from this period - <i>The Kid (1921), Metropolis (1927), 
City Lights (1931) and Modern Times(1936)</i> - all have more than 80% of its sounds as Other sounds.
<i> M (1931)</i>, is the only one closer to the current distribution, with 
Dialogue already being the most prominent sound.
""".format(image_name)

left.markdown(text, unsafe_allow_html=True)

fig = sound_share_by__type__year(movies_melt, sound_color_map)
fig.update_layout(
    height=400, 
    margin_b=0, 
    margin_r=0,
    legend_orientation='h',
    legend_y=-0.4,
    title=f'<b>{image_name}</b> - Sound type share evolution<br>1920 - 2020')
right.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

center = pad_cols([cols_baseline])[0]
text = """
The details of how each movie, separated by 20-year periods, contributes to the overall result, 
can be seen expanding this.
"""
expander = center.beta_expander(text)
with expander:
    expander.write('Hover on each point to see which movie it is, and its details.')
    fig = sound_share_strip_by__period(movies_melt, sound_color_map)
    fig.update_layout(title='', margin_r=0, margin_l=0, margin_t=50, height=400)
    expander.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

space_out(2)



### Genres and sound types

image_counter += 1
image_name = image_ref[image_counter]

center = pad_cols([cols_baseline])[0]
text = """
### <b>Sound x Genre</b>

Each movie in IMDB is associated to multiple genres, so a single movie is being considered 
in more than one calculation on the graph below. On <b>{}</b>, we can see not only how sound 
is distributed among sound types to each genre, but also how many movies each genre is considering
(in parenthisis). To help us, we'll highlight the genres with the highest sound share 
of each category.
""".format(image_name)

center.markdown(text, unsafe_allow_html=True)

fig = sound_share_by__type__genre(movies_melt, sound_color_map)
fig.update_layout(
    margin_r=50,
    margin_l=0,
    margin_b=200,
    height=1000,
    title=f'<b>{image_name}</b> - Sound distribution, per genre<br>(highlight to Top 3 of each category)')
center.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

text = """
On <b>{}</b>, we see that the highlighted genres change depending on how we account
every movie contribution. When looking at the Mean of each genre, individual films
have a much larger contribution.

One of such cases is the Sports movies. There are only 3 movies associated to the genre: 
<i> Dangal (2016), Children of Heaven (1997) and Raging Bull (1980)</i>. 
As, we've seen on <b>C</b>, <i>Raging Bull</i> is on the top 10 most silent movies, 
with 30% of it being silences, which considerably raises the average value for the genre.

Independently of how we account for each movie contribution, to each category there
are some genres that are highlighted either way. Those are:
* <b>Dialogue</b>: Musical and Film-Noir;
* <b>Other sounds</b>: Western and Action;
* <b>Silence</b>: War.

And these results seem intuitive in some way, right?
In musicals, the plot is developed through songs - which, since they appear in subtitles,
we're considering as dialogue. In Western and Actions movies, there's a lot of sounds happening all
the time: explosions, shootings, car chases, and so on. And War movies, even though also very 
loud, have those classic post-explosion defeaning moments.
""".format(image_name)
center.markdown(text, unsafe_allow_html=True)

text = """
The details of how each movie of a genre contributes to the overall result, 
can be seen expanding this.
"""
expander = center.beta_expander(text)
with expander:
    expander.write('Hover on each point to see which movie it is, and its details.')
    fig = sound_share_strip_per_genre(movies_melt, sound_color_map)
    fig.update_layout(title='', margin_t=0, height=300)
    expander.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

text = """
<br><br>
The fact that some of these genres have such consistent results may mean that what you hear, just as 
what you see, is known to be a powerful plot development tool. We can try to understand
if that is such the case, by analyzing when each of these sounds is used during the movie.
"""
center.markdown(text, unsafe_allow_html=True)

pad_l = 3
left, right = pad_cols([pad_l,  cols_baseline - pad_l])

text = """
Below, choose a sound type and a genre, to see when in the movie it's used.
"""
left.markdown(text, unsafe_allow_html=True)

chosen_sound = left.selectbox('Sound type', ['Silence', 'Dialogue', 'Other sounds'])
chosen_bins = left.select_slider('Bins', [100, 20, 10, 4], value=20)
smoother = int(100//chosen_bins)

fig = sound_type_by__position_genre(positions_df, smoother, chosen_sound, sound_color_map)
fig.update_layout(title='', margin_t=0, margin_r=0, height=400, margin_l=100)
right.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
space_out(2)

### Closing arguments
center = pad_cols([cols_baseline])[0]
text = """
### <b>The Power of Sound</b>
If you've started this reading oblivious to how important sound is, I really hope you've come to
understand how it plays such an important role on bringing movies to live, making
us feel everything from fear, to joy, sometimes in the space of a second.
"""
center.markdown(text, unsafe_allow_html=True)
space_out(3)


### About
text = """
### <b>Wanna know more?</b>
The complete code of this analysis can be found on the
<a href="https://github.com/adautobraz/ergo/tree/master/silence_in_movies"> github repository </a>.
All of the code is written in Python, data wrangling with Pandas, visualizations with Plotly,
and hosting with Streamlit.
"""
center.markdown(text, unsafe_allow_html=True)

expander = center.beta_expander("About me")
with expander:
    text = """
    I'm a full time Data Scientist, passionate about pop culture, data-driven stories and learning new things.
    I write some things at <a href="https://adautobraz.medium.com/">Medium</a> as well.
    Wanna reach out? Add me on <a href="https://www.linkedin.com/in/adautobraz/">Linkedin</a>
    or email me at adautobraz.neto@gmail.com
    """
    expander.markdown(text, unsafe_allow_html=True)
