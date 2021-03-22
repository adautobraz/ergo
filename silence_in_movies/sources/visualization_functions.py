import pandas as pd
import plotly.express as px

def plot(fig):
    fig.update_layout(
        font_family='Fira Sans', 
        template='plotly_white'
    )

    fig.show()


def facet_prettify(fig_raw):
    fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1].replace('_', ' ').title()))
    return fig