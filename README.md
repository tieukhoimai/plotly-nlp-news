# plotly-nlp-news

## [Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/code?datasetId=32526&sortBy=voteCount)

This dataset contains around 210k news headlines from 2012 to 2022 from HuffPost. This is one of the biggest news datasets and can serve as a benchmark for a variety of computational linguistic tasks. HuffPost stopped maintaining an extensive archive of news articles sometime after this dataset was first collected in 2018, so it is not possible to collect such a dataset in the present day. Due to changes in the website, there are about 200k headlines between 2012 and May 2018 and 10k headlines between May 2018 and 2022.

Each record in the dataset consists of the following attributes:

- category: category in which the article was published.
- headline: the headline of the news article.
- authors: list of authors who contributed to the article.
- link: link to the original news article.
- short_description: Abstract of the news article.
- date: publication date of the article.

## Project Description

The project includes 4 phase:
- Part 1: DATA EXPLORATION - data_cleaning.ipynb
- Part 2: DATA CLEANING - data_cleaning.ipynb
- Part 3: TEXTUAL PREPROCESSING AND ANALYSIS - data_cleaning.ipynb
  - 3.1. Textual Preprocessing - data_cleaning.ipynb
  - 3.2. Bigram Analysis - bigram.ipynb
- Part 4: LDA - TOPIC MODELLING - models.ipynb

## Directory Structure
```
plotly-nlp-news/
├── data/ - save data and result
├── model/ - save trained models of LDA
├── assets/ - save public resources as image
├── app.py
├── data_cleaning.ipynyb
├── bigram.ipynyb
├── models.ipynyb
├── requirements.txt
├── README.md
└── .gitignore
```

## Steps to run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

You can run the app on your browser at http://127.0.0.1:8050

## Result

!['screenshot'](assets/screenshot.png 'screenshot')

## Dataset Citation

1. Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
