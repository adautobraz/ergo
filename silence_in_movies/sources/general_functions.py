import pandas as pd
import plotly.express as px

def plot(fig):
    fig.update_layout(
        font_family='Fira Sans', 
        template='plotly_white'
    )

    fig.show()


def facet_prettify(fig_raw, capitalize=True):
    if capitalize:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1].replace('_', ' ').title()))
    else:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1]))
    return fig


def break_text(text, limit=15):
    new_text = ''
    words = text.split(' ')
    line = ''
    for w in words:
        if len(line + ' ' + w) <= limit:
            line = f"{line} {w}"
        else:
            new_text += f"{line}<br>"
            line = w
    
    new_text += " " + line
    return new_text.strip()


def vectorize_column(raw_df, col):
    df = raw_df.copy()
    df.loc[:, col] = df[col].apply(lambda x: eval(x))
    return df