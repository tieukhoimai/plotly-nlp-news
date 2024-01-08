# -*- coding: utf-8 -*-
"""
Module doc string
"""
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
from dateutil import relativedelta
from dash.dependencies import Output, Input, State
import pathlib
import re
import json
from datetime import datetime
import flask
import dash
from dash import dash_table
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

figure = {}

bigram_df = pd.read_csv("data/news_bigram_counts_data.csv")
dedup_df = pd.read_csv("data/dedup_data.csv")


DATA_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

FILENAME_PRECOMPUTED = "data/precomputed.json"
LOGO = "https://raw.githubusercontent.com/tieukhoimai/mia-blog-v3/main/public/static/images/logo.png"

DF = pd.read_json('data/News_Category_Dataset_v3.json', lines=True)

with open(DATA_PATH.joinpath(FILENAME_PRECOMPUTED)) as precomputed_file:
    PRECOMPUTED_LDA = json.load(precomputed_file)

"""
Casting the column to datetime
"""

DF["date"] = pd.to_datetime(
    DF["date"]
)

dedup_df["date"] = pd.to_datetime(
    dedup_df["date"]
)

"""
In order to make the graphs more useful we decided to prevent some words from being included
"""
ADDITIONAL_STOPWORDS = [
    "XXXX",
    "XX",
    "xx",
    "xxxx",
    "n't",
    "s",
    "will",
    "BOA",
    "Citi",
    "account",
]
for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)


def sample_data(dataframe, float_percent):
    """
    Returns a subset of the provided dataframe.
    The sampling is evenly distributed and reproducible
    """
    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)


def get_article_count_by_company(dataframe):
    """ Helper function to get article counts for unique categorys """
    category_counts = dataframe["category"].value_counts()
    # we filter out all categorys with less than 11 articles for now
    category_counts = category_counts[category_counts > 10]
    values = category_counts.keys().tolist()
    counts = category_counts.tolist()
    return values, counts


def calculate_category_sample_data(dataframe, sample_size):
    print(
        "making category_sample_data with sample_size count: %s"
        % (sample_size)
    )

    category_counts = dataframe["category"].value_counts()
    category_counts_sample = category_counts[:sample_size]
    values_sample = category_counts_sample.keys().tolist()
    counts_sample = category_counts_sample.tolist()

    return values_sample, counts_sample


def make_local_df(selected_category, n_selection):
    print("redrawing bank-wordcloud...")
    n_float = float(n_selection / 100)
    print("got n_selection:", str(n_selection), str(n_float))
    # sample the dataset according to the slider
    local_df = sample_data(DF, n_float)
    if selected_category:
        local_df = local_df[local_df["category"] == selected_category]
    return local_df

def make_options_category_drop(values):
    """
    Helper function to generate the data format the dropdown dash component wants
    """
    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret


def plotly_wordcloud(data_frame):
    """returns figure data for three equally plots: wordcloud, frequency histogram and treemap"""
    articles_text = list(data_frame["short_description"].dropna().values)

    if len(articles_text) < 1:
        return {}, {}, {}

    # join all documents in corpus
    text = " ".join(list(articles_text))

    word_cloud = WordCloud(stopwords=STOPWORDS,
                           max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f)
                   for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:10]
    word_list_top.reverse()
    freq_list_top = freq_list[:10]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    return wordcloud_figure_data, frequency_figure_data, treemap_figure


"""
#  Page layout and contents
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand(
                            "News Category Dashboard", className="ml-1")
                    ),
                ],
                align="center",
            ),
            href="https://tieukhoimai.me/",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Select category & data size",
                className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select percentage of dataset", className="lead"),
        dcc.Slider(
            id="n-selection-slider",
            min=1,
            max=100,
            step=1,
            marks={
                0: "0%",
                10: "",
                20: "20%",
                30: "",
                40: "40%",
                50: "",
                60: "60%",
                70: "",
                80: "80%",
                90: "",
                100: "100%",
            },
            value=20,
        ),
        html.Label("Select a category", style={
                   "marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="category-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
    ]
)


WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Most frequently used words in articles")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[
                                                    dcc.Graph(id="category-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(
                                                        id="category-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=8,
                    ),
                ]
            )
        ]
    ),
]

HEADLINE_BOXPLOT = [
    dbc.CardHeader(html.H5("Distribution of article headline's wordcliped")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-categorys-boxplot",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-boxplot-data-alert",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="headline-boxplot"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_CATEGORYS_PLOT = [
    dbc.CardHeader(html.H5("Top 10 categorys by number of articles")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-categorys-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-category",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="category-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_CATEGORYS = [
    dbc.CardHeader(html.H5("Top 10 bigram in Article Headlines")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-headlines",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_category",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.P("Choose categories to see op 10 bigram:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-category",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.category.unique()
                                        ],
                                        value="TOTAL",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-category_plot"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_COMPARISION = [
    dbc.CardHeader(html.H5("Comparison of bigrams for two categories")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.P("Choose two categories to compare:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-comp_1",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.category.unique()
                                        ],
                                        value="POLITICS",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-comp_2",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.category.unique()
                                        ],
                                        value="WELLNESS",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-comps"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        html.H4("DATA EXPLORATION", style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(TOP_CATEGORYS_PLOT), md=8),
            ],
            style={"marginTop": 30},
            ),
        dbc.Row([dbc.Col(dbc.Card(WORDCLOUD_PLOTS)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(HEADLINE_BOXPLOT)),],
                style={"marginTop": 30}),
                
        html.H4("HEADLINE ANALYSIS", style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_CATEGORYS)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_COMPARISION)),],
                style={"marginTop": 30}),
    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

"""
##  DATA EXPLORATION
"""


@app.callback(
    Output("category-drop", "options"),
    [
     Input("n-selection-slider", "value")],
)
def populate_category_dropdown(n_value):
    n_value += 1
    category_names, counts = get_article_count_by_company(DF)
    counts.append(1)
    return make_options_category_drop(category_names)


@app.callback(
    [Output("category-sample", "figure"),Output("no-data-alert-category", "style")],
    [Input("n-selection-slider", "value"),],
)
def update_category_sample_plot(n_value):
    print("redrawing category-sample...")
    print("\tn is:", n_value)
    n_float = float(n_value / 100)
    category_sample_count = 10
    local_df = sample_data(DF, n_float)

    values_sample, counts_sample = calculate_category_sample_data(
        local_df, category_sample_count
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample,
            "text": values_sample,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    print("redrawing category-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(
    [
        Output("category-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("category-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [
        Input("category-drop", "value"),
        
        Input("n-selection-slider", "value"),
    ],
)
def update_wordcloud_plot(value_drop, n_selection):
    """ Callback to rerender wordcloud plot """
    local_df = make_local_df(value_drop, n_selection)
    wordcloud, frequency_figure, treemap = plotly_wordcloud(local_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    print("redrawing category-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)


@app.callback(
        Output("headline-boxplot", "figure"),
    [
        Input("n-selection-slider", "value"),
        
    ],
)
def update_headline_distribution_plot(n_value):

    print("redrawing headline_distribution_plot...")
    print("\tn is:", n_value)

    n_float = float(n_value / 100)
    local_df = sample_data(dedup_df, n_float)

    fig = px.box(local_df,
                 x = 'category',
                 y = 'words_clipped_headline',
                 template='plotly_white')
    
    fig.update_layout(
        yaxis_title='Number of words in headline',
        xaxis_title='Category')
    
    return fig


@app.callback(Output("category-drop", "value"), [Input("category-sample", "clickData")])
def update_category_drop_on_click(value):
    if value is not None:
        selected_category = value["points"][0]["x"]
        return selected_category
    return "POLITICS"


"""
##  HEADLINES ANALYST
"""

@app.callback(
    Output("bigrams-category_plot", "figure"),
    [Input("bigrams-category", "value")],
)
def category_bigram(category_options):

    category_lst = [category_options]
    temp_df = bigram_df[bigram_df.category.isin(category_lst)]
    temp_df = temp_df.sort_values(by=['value'], ascending=False)[:10]

    trace = go.Bar(
        x=temp_df['value'],
        y=temp_df['bigram'],
        text=temp_df['value'],
        textposition='outside',
        orientation='h')
    layout = go.Layout(template="plotly_white",
                       margin=dict(l=20, r=20, t=30, b=20),
                       xaxis=dict(title='Count'), 
                       yaxis=dict(title='Bigram', autorange="reversed"))

    fig = go.Figure(data=[trace], layout=layout)

    return fig


@app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value"), Input("bigrams-comp_2", "value")],
)
def category_bigram_comparisons(category_first, category_second):
    category_list = [category_first, category_second]
    temp_df = bigram_df[bigram_df.category.isin(category_list)]
    temp_df.loc[temp_df.category == category_list[-1], "value"] = -temp_df[
        temp_df.category == category_list[-1]
    ].value.values

    fig = px.bar(
        temp_df,
        title="Comparison: " + category_first + " | " + category_second,
        x="bigram",
        y="value",
        color="category",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"category": "Category:", "bigram": "Bi-Gram"},
        hover_data="category",
    )
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
