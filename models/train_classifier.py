import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import pandas as pd
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from workspace_utils import active_session
import pickle

def load_data(database_filepath):
    '''
    This function load database and split feature and label
    Parameters
    ----------
    datafile_filepath
        Database data.

    Returns
    -------
    X 
      A dataframe with feature 
    y
      A dataframe with targets 
    category_names
      Targets name
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM emergency", engine)
    
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    This function tokenize, remove stop words, lemmitize and stemming text data
    Parameters
    ----------
    text
        A dataframe with messages

    Returns
    -------
    stem 
        cleaned text data
    
    ''' 
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


def build_model():
    '''
    This function build a model with pipeline and grid search.
    Parameters
    ----------
    None
    
    Returns
    -------
    model 
        model parameters
    
    ''' 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])



    parameters = {  'clf__estimator__n_estimators': [300, 400]}

    model = GridSearchCV(pipeline, param_grid=parameters,cv=2, verbose=12, n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function make prediction and generate a model report
    Parameters
    ----------
    model
        A model trained
    X_test
        Dataframe with feature for test set
    y_test
        Dataframe with targets values for test evaluation
    category_names
        Column names for targets
    Returns
    -------
    report _df
        a dataframe with model evaluation metrics
    
    ''' 

    y_pred = model.predict(X_test)
    report= classification_report(y_test,y_pred, target_names=category_names)
    print(classification_report(y_test,y_pred, target_names=category_names))
    temp=[]
          
    for item in report.split("\n"):
        temp.append(item.strip().split('     '))
    clean_list=[x for x in temp if x != ['']]
    report_df=pd.DataFrame(clean_list[1:],columns=['group','precision','recall', 'f1-score','support'])

    
    return report_df

def save_model(model, model_filepath):
    '''
    This function save the trained model
    Parameters
    ----------
    model
        A model trained
    model_filepath
        File path to save the model
    Returns
    -------
    None
    
    '''             
    with open(model_filepath, 'wb') as file:
         pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42,shuffle=True)
        
        print('Building model...')
        model = build_model()
        with active_session():
            print('Training model...')
            model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()