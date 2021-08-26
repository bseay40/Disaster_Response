import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    '''
    Loads the cleaned SQL dataset output from the ETL pipeline
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('clean_message_dataset', engine)
    X = df['message'] # the message is the predictor
    Y = df.iloc[:,4:] # the 36 categories is what we are trying to predict the message falls in.

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizing function to process text data
    '''
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")] # removing stop words
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def build_model():
    '''
    Builds the model pipeline, creates Grid Search parameters to tune model.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,lowercase = True)),
        ('tfidf', TfidfTransformer(sublinear_tf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__strip_accents': ['ascii','unicode', None],
        'vect__lowercase': [True, False],
        'tfidf__sublinear_tf': [True, False],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return model
    #X_train, X_test, y_train, y_test = train_test_split(X, Y)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Outputs model test results for each of the categories.
    '''
    x_pred = model.predict(X_test)
    for col in range(0,36,1):
        print(category_names[col])
        print(classification_report(Y_test.iloc[:,col], x_pred[:,col]))


def save_model(model, model_filepath):
    '''
    Saves the model using Pickle.
    '''
    filename = model_filepath
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

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
