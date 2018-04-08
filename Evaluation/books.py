import os
import html
import re
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom libraries - must be in same directory
import xml_to_dict
import get_book_tags
import get_bookid_mapper
from util import reduce_matrix

def clean_string(s):
    # often times a book will be missing a feature so we have to return if None
    if not s:
        return s
    
    # clean html
    TAG_RE = re.compile(r'<[^>]+>')
    s = html.unescape(s)
    s = TAG_RE.sub('', s)
    s = s.lower()
    return s

# Read in book metadata and store in a dictionary
def load_books(metadata_directory, goodreads_to_bookid, book_tags):
    books = []
    for file in os.listdir(metadata_directory):
        filename = metadata_directory + '/' + os.fsdecode(file)
        raw_book, popular_shelves = xml_to_dict.dict_from_xml_file(filename)

        book = {}
        goodreads_id = raw_book['book']['id']
        book['id'] = goodreads_to_bookid[goodreads_id]
        book['title'] = raw_book['book']['title']
        book['author'] = raw_book['book']['authors']['author']
        
        # if multiple authors, only use first (main) author
        if isinstance(book['author'], dict):
            book['author'] = book['author']['name']
        else:
            book['author'] = book['author'][0]['name']

        book['description'] = raw_book['book']['description']
        book['description'] = clean_string(book['description'])

        # Turn popular shelves into soup
        book['popular_shelves'] = ''
        normalizing_value = 5
        for key,value in popular_shelves.items():
            for i in range(int(value) // normalizing_value):
                book['popular_shelves'] += ' ' + key
        
        # Turn book tags into soup
        book['tags'] = ''
        tags = book_tags[goodreads_id]
        for key,value in tags.items():
            for i in range(int(value) // normalizing_value):
                book['tags'] += ' ' + key

        books.append(book)
    return books

def get_book_dataframe(data_path):
    # Read in mapper and book tags
    goodreads_to_bookid = get_bookid_mapper.get_mapper(data_path + 'books.csv')
    book_tags = get_book_tags.get_tags(data_path + 'book_tags_with_bookid.csv', data_path + 'tags.csv')
    # Get books as dictionary of all its features
    metadata_directory = data_path + 'books_xml/books_xml'
    books = load_books(metadata_directory, goodreads_to_bookid, book_tags)

    df = pd.DataFrame(books)
    df['id'] = df['id'].astype(int)
    df = df.sort_values(by=['id'])
    df = df.set_index('id')

    #Construct a reverse map of indices and book titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    #Replace NaN with an empty string
    df['description'] = df['description'].fillna('')
    return df

def get_book_features(df):
    """ Returns the sparse feature vector of books.

    The features are the tf-idf values of the book descriptions,
    popular shelves, and book tags.
    """
    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix_description = tfidf.fit_transform(df['description'])
    tfidf_matrix_shelves = tfidf.fit_transform(df['popular_shelves'])
    tfidf_matrix_tags = tfidf.fit_transform(df['tags'])

    return scipy.sparse.hstack([tfidf_matrix_description, tfidf_matrix_shelves, tfidf_matrix_tags])

def get_book_authors(df):
    """ Returns the sparse author counts matrix """
    count_matrix_author = pd.get_dummies(df['author'])
    count_matrix_author = scipy.sparse.csr_matrix(count_matrix_author.values)
    return count_matrix_author

def main():
    """ Sample program to verify the code.

    This method will load in the book features, do some preprocessing,
    and use SVD to reduce it to 100 dimensions. It will then output
    the top 10 singular values.
    """
    # Set this to where you save and load all data
    data_path = '../data/goodbooks-10k/'
    df = get_book_dataframe(data_path)
    fv = get_book_features(df)
    U, S, VT = reduce_matrix(fv, 100, random_state = 42)
    print(S[:10])

if __name__ == '__main__':
    main()
