import pandas as pd
import plotly.express as px

def plot(fig):
    fig.show()


def format_fig(fig):
    fig.update_layout(
        font_family='Helvetica', 
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
    )
    fig.update_xaxes(fixedrange = True)
    fig.update_yaxes(fixedrange = True)

    return fig


def facet_prettify(fig_raw, capitalize=True):
    if capitalize:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1].replace('_', ' ').title()))
    else:
        fig = fig_raw.for_each_annotation(lambda x: x.update(text = x.text.split('=')[1]))
    return fig


def leave_only_slider(fig):
    fig['layout']['updatemenus']=[]
    fig['layout']['sliders'][0]['x']=0
    fig['layout']['sliders'][0]['len']=1
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