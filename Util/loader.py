import os
import html
import re
import csv
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom functions
import xml_to_dict
from global_vars import bad_features

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

def get_mapper(filename):
    mapper = {}
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            mapper[line[1]] = line[0]
    return mapper

def get_tags(book_tags, tags):
    tag_defs = {}
    with open(tags, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            tag_defs[line[0]] = line[1]

    books = {}
    with open(book_tags, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            goodreads_book_id = line[1]
            tag_id = line[2]
            count = line[3]
            if goodreads_book_id not in books:
                books[goodreads_book_id] = {}
            
            tag_name = tag_defs[tag_id]
            if tag_name not in bad_features:
                books[goodreads_book_id][tag_name] = count
    return books

# Read in book metadata and store in a dictionary
def get_books(data_path):
    metadata_directory = data_path + 'books_xml/books_xml'
    goodreads_to_bookid = get_mapper(data_path + 'books.csv')
    book_tags = get_tags(data_path + 'book_tags_with_bookid.csv', data_path + 'tags.csv')
    books = []
    
    for file in os.listdir(metadata_directory):
        filename = metadata_directory + '/' + os.fsdecode(file)
        raw_book, popular_shelves = xml_to_dict.dict_from_xml_file(filename)

        book = {}
        goodreads_id = raw_book['book']['id']
        book['id'] = goodreads_to_bookid[goodreads_id]
        book['title'] = raw_book['book']['title']
        book['image_url'] = raw_book['book']['image_url']
        book['url'] = raw_book['book']['url']
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
    # check if dataframe already exists
    try:
        books = pd.read_pickle('../.tmp/books_dataframe')
        print("found books_dataframe in file...")
        return books
    except:
        # Get books as dictionary of all its features
        books = get_books(data_path)

        df = pd.DataFrame(books)
        df['id'] = df['id'].astype(int)
        df = df.sort_values(by=['id'])
        df = df.set_index('id')

        #Replace NaN with an empty string
        df['description'] = df['description'].fillna('')
        print('saving books_dataframe to file')
        df.to_pickle('../.tmp/books_dataframe')
        return df

def get_book_features(df):
    """ Returns the sparse feature vector of books.

    The features are the tf-idf values of the book descriptions,
    popular shelves, and book tags.
    """
    # see if file exists in file
    try:
        feature_matrix = scipy.sparse.load_npz('../.tmp/feature_matrix.npz')
        print('feature_matrix exists in file...')
        return feature_matrix
    except:
        tfidf = TfidfVectorizer(stop_words='english')

        tfidf_matrix_description = tfidf.fit_transform(df['description'])
        tfidf_matrix_shelves = tfidf.fit_transform(df['popular_shelves'])
        tfidf_matrix_tags = tfidf.fit_transform(df['tags'])

        # Weight the smaller matrices bc ration to largest column matrix
        shelves_weight = tfidf_matrix_description.shape[1] / tfidf_matrix_shelves.shape[1]
        tags_weight = tfidf_matrix_description.shape[1] / tfidf_matrix_tags.shape[1]

        tfidf_matrix_shelves = tfidf_matrix_shelves.multiply(shelves_weight)
        tfidf_matrix_tags = tfidf_matrix_tags.multiply(tags_weight)

        feature_matrix = scipy.sparse.hstack([tfidf_matrix_description, tfidf_matrix_shelves, tfidf_matrix_tags])
        print('printing feature_matrix to file')
        scipy.sparse.save_npz('../.tmp/feature_matrix', feature_matrix)
        return feature_matrix

def get_book_authors(df):
    """ Returns the sparse author counts matrix """
    count_matrix_author = pd.get_dummies(df['author'])
    count_matrix_author = scipy.sparse.csr_matrix(count_matrix_author.values)
    return count_matrix_author
