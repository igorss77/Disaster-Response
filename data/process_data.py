import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function load Messages and Categories dataset and  merge them
    Parameters
    ----------
    messages_filepath
        Message dataframe.
    categories_filepath
        Categories dataframe.

    Returns
    -------
    dataframe
        df - merged dataframe.

    
    '''
    # read in file

    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    #merge message_df and categories_df on id column
    df = pd.merge(messages_df, categories_df, on="id")

    return df


def clean_data(df):
    '''
    This function load formatted dataset and clean the data and also label columns
    Split categories columns
    Convert values to 0 and 1
    Remove duplicates
    Parameters
    ----------
    df
        Merged dataframe with message and categories..

    Returns
    -------
    dataframe
        df - cleaned dataframe.
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  [x for x in row.str[:-2]]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = df.categories.str[-1:]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    df= df.drop('categories',axis=1)
    
    df = pd.concat([df,categories],axis=1)
    
    # remove duplicated
    
    df = df.drop_duplicates()
    # replace value '2' for the minor category
    df.loc[df['related']>1,'related'] = 0
    return df
def save_data(df, database_filename):
    '''
    This function save the cleaned dataframe into a sql database
    Parameters
    ---------- 
    df
        Cleaned dataframe with message and categories..

    Returns
    -------
         None
    
    ''' 
  
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('emergency', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()