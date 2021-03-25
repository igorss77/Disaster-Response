import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import re

app = Flask(__name__)

def tokenize(text):
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # case normalization, punctuation removal and tokenization
    words = word_tokenize(re.sub(r'[^a-zA-Z0-9]', " ", text.lower()))

    # removing stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    # stemming
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    return stemmed
            
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('emergency', engine)

model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
        
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_per = round(100*genre_counts/genre_counts.sum(), 2)
    genre_names = list(genre_counts.index)
    
    count_words = pd.Series(' '.join(df.message).lower().split()).value_counts()[:100]
    water_counts = df.groupby('water').count()['message']
    water = ['Yes' if i==1 else 'No' for i in list(water_counts.index)]
    category_names=df.columns.values[4:]
    category_counts=df[category_names].sum()

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        { 'data' : [
                    Bar(x=water,y=water_counts, orientation='v')
                    ],
         'layout' : {
                    'title' : 'Is the message related with water?',
                    'yaxis' : {'title' : 'Count'},
                    'xaxis' : {'title' : 'Water'}
                    },
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.1,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_per,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                   "#9467bd",
                    "#2ca02c",
                    "#17becf"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "Distribution of Messages by Genre"
            }
        },
        { 'data' : [
                    Bar(x=category_names,y=count_words, orientation='v')
                    ],
         'layout' : {
                    'title' : 'Distribution of Category of Disastor',
                    'yaxis' : {'title' : 'Count'},
                    'xaxis' : {'title' : 'Category'}
                    },
        }
        
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# In[9]:
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

# In[10]:
if __name__ == '__main__':
    main()
