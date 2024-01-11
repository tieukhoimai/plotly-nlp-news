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
import pandas as pd
from collections import Counter
import gensim

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots



import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=UserWarning)

from gensim.models.ldamodel import LdaModel

figure = {}

bigram_df = pd.read_csv("data/news_bigram_counts_data.csv")

DATA_PATH = pathlib.Path(__file__).parent.resolve()

LOGO = "https://raw.githubusercontent.com/tieukhoimai/mia-blog-v3/main/public/static/images/logo.png"

DF = pd.read_json('data/News_Category_Dataset_v3.json', lines=True)
DEDUP_DF = pd.read_csv("data/dedup_data.csv")
PROCESSED_DF = pd.read_csv("data/processed_data.csv")
DOMINANT_TOPIC_DF = pd.read_csv('data/lda_data.csv')

LDA_MODEL = LdaModel.load("model/model_lda.model")

"""
Reorder column
"""

DF = DF[['date','headline','short_description','category','authors','link']]
PROCESSED_DF = PROCESSED_DF[['date','headline','short_description','category','authors','link']]

"""
Casting the column to datetime
"""

DF["date"] = pd.to_datetime(
    DF["date"]
)

DEDUP_DF["date"] = pd.to_datetime(
    DEDUP_DF["date"]
)

PROCESSED_DF["date"] = pd.to_datetime(
    PROCESSED_DF["date"]
)

"""
In order to make the graphs more useful we decided to prevent some words from being included
"""
ADDITIONAL_STOPWORDS = [
    "n't",
    "s",
    "will",
    "u",
    "say",
    "says",
    "said",
    "HuffPost Style",
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

def count_words(text):
    if isinstance(text, str):
        words = text.split()
        return len(words)
    else:
        return 0
    
def make_local_df(dataframe, selected_category, n_selection):
    print("redrawing wordcloud...")
    n_float = float(n_selection / 100)
    print("got n_selection:", str(n_selection), str(n_float))
    # sample the dataset according to the slider
    local_df = sample_data(dataframe, n_float)
    if selected_category:
        local_df = local_df[local_df["category"] == selected_category]
    return local_df

def add_word_length_feature(dataframe, column):
    dataframe['word_length'] = dataframe[column].apply(count_words)
    return dataframe

def make_options_category_drop(values):
    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret

def plotly_wordcloud(data_frame, bigram_flag = False):
    if bigram_flag:
        articles_text = data_frame["bigram"].tolist()
    else:
        articles_text = list(data_frame["headline"].dropna().values)

    if len(articles_text) < 1:
        return {}, {}, {}

    text = " ".join(articles_text)

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

    if bigram_flag:
        return wordcloud_figure_data, treemap_figure
    else:
        return wordcloud_figure_data, frequency_figure_data, treemap_figure

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def plot_lda_by_topic():
    topics = LDA_MODEL.show_topics(formatted=False)
    data = list(PROCESSED_DF['headline'].values)

    data_words = list(sent_to_words(data))

    data_flat = [w for w_list in data_words for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    local_df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Create subplot
    fig = make_subplots(rows=1, cols=5, subplot_titles=['Topic: ' + str(i) for i in range(5)],
                        shared_yaxes=True, vertical_spacing=0.2)

    for i in range(5):
        subplot_df = local_df.loc[local_df.topic_id == i, :]
        
        fig.add_trace(
            go.Bar(x=subplot_df['word'], y=subplot_df['word_count'], name='Word Count',
                text=subplot_df.apply(lambda row: f"Word Count: {row['word_count']}, Importance: {row['importance']:.3f}", axis=1),
                hoverinfo='text+x+y',  # Set hover information
                textposition='outside'),  # Set text position outside the bars
            row=1, col=i + 1
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        # title_text='Word Count and Importance of Topic Keywords',
        showlegend=False
    )

    return fig

def create_dominant_topic_groupby_df(dataframe):
    df_dominant_topic_groupby = dataframe.groupby(by=['Dominant_Topic','Topic_Keywords','category'])['headline'].count().reset_index()
    df_dominant_topic_groupby.columns = ['Dominant_Topic','Topic_Keywords','Category', 'Count']

    return df_dominant_topic_groupby

def create_top_category_dominant_topic_df():
    top_category = (
        DF.groupby("category")["link"]
        .count()
        .sort_values(ascending=False)[:20]
        .index
    )

    df_dominant_topic_groupby = create_dominant_topic_groupby_df(DOMINANT_TOPIC_DF)
    top_category_dominant_topic_df = df_dominant_topic_groupby[df_dominant_topic_groupby["Category"].isin(top_category)]

    return top_category_dominant_topic_df

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

ORGINAL_TABLE = [
    dbc.CardHeader(html.H5("Orginal Data")),
    dbc.CardBody(
        [
            html.P(id='table_out'),
            dash_table.DataTable(
                id='table',
                data=DF[:10].to_dict('records'),
                columns=[{"name": i, "id": i} for i in DF.columns],
                style_table={'fontSize':12, 'overflowX': 'auto'},
            ), 
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

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
                            children=[dcc.Graph(id="frequency-figure")],
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

CLEANING_DESCRIPTION = [
    dbc.CardHeader(html.H5("Data Cleaning Preprocessing")),
    dbc.CardBody(
        [
            html.P("1. Excluding Articles with Null or Invalid Published Date: 0 instances"),
            html.P("2. Drop duplicate articles and keep the earliest publication date: 22,505 instances"),
            dcc.Loading(
                dash_table.DataTable(
                id='table_dedup',
                data=DF.loc[DF.short_description.duplicated()][:5].to_dict('records'),
                columns=[{"name": i, "id": i} for i in DF.columns],
                style_table={'fontSize':12, 'overflowX': 'auto'},
                ),
            ),
            html.P(""),
            html.P("3. Eliminate articles which having invalid headline length: 1 instances with word count is 0"),
            dcc.Loading(
                id="loading-headline-length-dist",
                children=[
                    dcc.Graph(id="headline-length-dist"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BAR_CATEGORYS_PLOT = [
    dbc.CardHeader(html.H5("Number of articles by top 10 category and year")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-categorys-bar",
                children=[
                    dcc.Graph(id="date-category-bar-chart"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

HEADLINE_BOXPLOT = [
    dbc.CardHeader(html.H5("Distribution of article headline's length")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-categorys-boxplot",
                children=[
                    dcc.Graph(id="headline-boxplot"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

PREPROCESSING_DESCRIPTION = [
    dbc.CardHeader(html.H5("Textual Preprocessing")),
    dbc.CardBody(
        [
            html.P("1. Remove HTML, Hyperlinks, Newlines, Numbers, Remove Special Characters, Punctuation, Whitespace"),
            html.P("2. Decontracted takes text and convert contractions into natural form."),
            html.P("3. Tokenization: Split the text into sentences and the sentences into words."),
            html.P("4. Remove stop words"),
            html.P("5. Lemmatize: words in third person are changed to first person and verbs in past and future tenses are changed into present."),
            html.P("6. Stemme: words are reduced to their root form"),
            dcc.Loading(
                dash_table.DataTable(
                id='table_preprocess',
                data=PROCESSED_DF[:5].to_dict('records'),
                columns=[{"name": i, "id": i} for i in PROCESSED_DF.columns],
                style_table={'fontSize':12, 'overflowX': 'auto'},
                ),
            ),
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
                                html.P("Choose categories to see Top 10 bigram:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-drops",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.category.unique()
                                        ],
                                        value="POLITICS",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-category-plot"),
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
                                        value="TOTAL",
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

LDA_DESCRIPTION = [
    dbc.CardHeader(html.H5("LDA Preprocessing")),
    dbc.CardBody(
        [
            html.P("Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling with implementations in the Python’s Gensim package. LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion."),
            dcc.Loading(
                dash_table.DataTable(
                id='table_dominant_topic',
                data=DOMINANT_TOPIC_DF.loc[:5].to_dict('records'),
                columns=[{"name": i, "id": i} for i in DOMINANT_TOPIC_DF.columns],
                style_table={'fontSize':12, 'overflowX': 'auto'},
                ),
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

LDA_CHART_BY_TOPIC = [
    dbc.CardHeader(html.H5("Word Count and Importance of Topic Keywords")),
    dbc.CardBody(
        [
            dcc.Graph(id="lda-topic"),
            html.P("This means that Topic 0 is a represented as 0.015*studi + 0.012*poll + 0.010*health + 0.009*idea + 0.009*new + 0.008*network + 0.008*guid + 0.007*hous + 0.006*diy + 0.006*plan"),
            html.P("Generally speaking, the top 10 keywords that contribute to this topic are: 'study', 'poll', 'health',... and so on. The weights reflect how important a keyword is to that topic. The weight of 'study' on topic 0 is 0.015.")
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

LDA_CHART_TOPIC_BY_TOP_CATEGORY = [
    dbc.CardHeader(html.H5("Topic Volum across Top Category")),
    dbc.CardBody(
        [
            dcc.Graph(id="lda-topic-by-top-category"),
            html.P("Topic 0 dominates the 'POLITICS' category with more than 20k articles, evident from the high count in the corresponding bar"),
            html.P("Other topics are more evenly distributed across the top 10 categories, showcasing a balanced distribution."),
            dcc.Graph(id="lda-top-category-by-topic"),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

LDA_CHART_TOPIC_BY_HEATMAP = [
    dbc.CardHeader(html.H5("Dominant Topic-Category Distribution")),
    dbc.CardBody(
        [
            dcc.Graph(id="lda-category-by-topic-heatmap"),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [

        #### PART 1

        html.H4("DATA EXPLORATION", style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(ORGINAL_TABLE)),],
                style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(TOP_CATEGORYS_PLOT), md=8),
            ],
            style={"marginTop": 30},
            ),
        dbc.Row([dbc.Col(dbc.Card(BAR_CATEGORYS_PLOT)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(WORDCLOUD_PLOTS)),],
                style={"marginTop": 30}),

        #### PART 2

        html.H4("DATA CLEANING", style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CLEANING_DESCRIPTION)),],
                        style={"marginTop": 30}),

        #### PART 3

        html.H4("TEXTUAL PREPROCESSING AND ANALYSIS", style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(HEADLINE_BOXPLOT)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(PREPROCESSING_DESCRIPTION)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_CATEGORYS)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_COMPARISION)),],
                style={"marginTop": 30}),

        #### PART 3
        html.H4("LDA TOPIC - MODELLING", style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(LDA_DESCRIPTION)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(LDA_CHART_BY_TOPIC)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(LDA_CHART_TOPIC_BY_TOP_CATEGORY)),],
                style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(LDA_CHART_TOPIC_BY_HEATMAP)),],
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
    Output('table_out', 'children'), 
    Input('table', 'active_cell'))
def update_graphs(active_cell):
    if active_cell:
        cell_data = DF.iloc[active_cell['row']][active_cell['column_id']]
        return f"Data: \"{cell_data}\" from table cell: {active_cell}"
    return ""

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
    Output("headline-length-dist", "figure"),
    [Input("n-selection-slider", "value")],
)
def update_article_distribution(n_value):
    # This step need show full data to count so not using n_values params
    fig = px.histogram(
        DEDUP_DF,
        x='words_clipped_headline',
        template='plotly_white',
        marginal='box',
        labels={"words_clipped_headline": "Words count", "count": "Number of Articles"})
    fig.update_xaxes(categoryorder='total descending', title='Words count')
    fig.update_yaxes(title='Number of article')
    return fig

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
        Output("date-category-bar-chart", "figure"),
        [Input("n-selection-slider", "value"),]
)
def update_bar_plot_by_date_and_category(n_value):

    n_float = float(n_value / 100)
    local_df = sample_data(DF, n_float)

    # Pre-processing data by get top 10 category
    top_category = (
        local_df.groupby("category")["link"]
        .count()
        .sort_values(ascending=False)[:10]
        .index
    )

    top_category_df = local_df[local_df["category"].isin(top_category)]
    top_category_df['date'] = pd.to_datetime(top_category_df["date"])

    # Articles by category & date
    fig = px.histogram(
        top_category_df,
        x="date",
        template="plotly_white",
        color="category",
        nbins=10,
        log_y=True,
        barmode="group",
    )
    fig.update_xaxes(categoryorder="category ascending", title="Year").update_yaxes(
        title="Number of articles"
    )
    return fig


@app.callback(
    [
        Output("category-wordcloud", "figure"),
        Output("frequency-figure", "figure"),
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
    local_df = make_local_df(DF, value_drop, n_selection)
    wordcloud, frequency_figure, treemap = plotly_wordcloud(local_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    print("redrawing category-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)


@app.callback(
        Output("headline-boxplot", "figure"),
    [   Input("n-selection-slider", "value"),  ],
)
def update_headline_distribution_plot(n_value):

    print("redrawing headline_distribution_plot...")
    print("\tn is:", n_value)

    n_float = float(n_value / 100)

    local_df = sample_data(DEDUP_DF[DEDUP_DF['words_clipped_headline'] > 0], n_float)

    fig = px.box(local_df,
                 x = 'category',
                 y = 'words_clipped_headline',
                 template='plotly_white')
    
    fig.update_layout(
        yaxis_title='Number of words in headline',
        xaxis_title='Category')
    
    return fig


@app.callback(Output("category-drop", "value"), 
              [Input("category-sample", "clickData")])
def update_category_drop_on_click(value):
    if value is not None:
        selected_category = value["points"][0]["x"]
        return selected_category
    return "POLITICS"


"""
##  HEADLINES ANALYST
"""

@app.callback(
    Output("bigrams-category-plot", "figure"),
    [Input("bigrams-drops", "value")],
)
def category_bigram(bigrams_drops):

    category_lst = [bigrams_drops]
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
                       yaxis=dict(title='Bi-Gram', autorange="reversed"))

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

@app.callback(
        Output("lda-topic", "figure"),
        [
            Input("n-selection-slider", "value"),
        ],
)
def show_lda_by_topic(n_selection):
    lda_by_topic = plot_lda_by_topic()
    
    return lda_by_topic

@app.callback(
        Output("lda-topic-by-top-category", "figure"),
        [
            Input("n-selection-slider", "value"),
        ],
)
def show_lda_topic_by_top_category(n_selection):
    local_df = create_top_category_dominant_topic_df()

    fig = px.bar(local_df, 
                x='Dominant_Topic', 
                y='Count',
                text='Count',
                hover_data=['Topic_Keywords'],
                color='Category',
                labels={'Count':'Number of Articles','Dominant_Topic':'Dominant Topic'},
                barmode="group",
                )

    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    return fig


@app.callback(
        Output("lda-top-category-by-topic", "figure"),
        [
            Input("n-selection-slider", "value"),
        ],
)
def show_lda_top_category_by_topic(n_selection):
    local_df = create_top_category_dominant_topic_df()

    local_df['Dominant_Topic'] = local_df['Dominant_Topic'].astype(str)

    fig = px.bar(local_df, 
                x='Category', 
                y='Count',
                text='Count',
                hover_data=['Topic_Keywords'],
                color='Dominant_Topic',
                labels={'Count':'Number of Articles'},
                barmode="stack",
                #  facet_col="Dominant_Topic",
                #  category_orders={"Dominant_Topic": ["0", "1", "2", "3", "4"]}
                )

    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    return fig


@app.callback(
        Output("lda-category-by-topic-heatmap", "figure"),
        [
            Input("n-selection-slider", "value"),
        ],
)
def show_lda_category_by_topic_heatmap(n_selection):
    df_dominant_topic_groupby = create_dominant_topic_groupby_df(DOMINANT_TOPIC_DF)

    df_dominant_topic_corr = df_dominant_topic_groupby[['Dominant_Topic','Category','Count']]
    df_dominant_topic_corr = df_dominant_topic_corr[df_dominant_topic_groupby["Category"]!='POLITICS']

    heatmap_data = df_dominant_topic_corr.pivot(index='Dominant_Topic', columns='Category', values='Count')

    fig = px.imshow(heatmap_data, color_continuous_scale='Teal', origin='lower', text_auto=True, aspect="auto")

    fig.update_yaxes(title='Dominate Topic')
    fig.update_xaxes(title='Category (Excluding POLICTICS)')
    
    return fig


"""
##  LDA MODELING
"""

if __name__ == "__main__":
    app.run_server(debug=True)
