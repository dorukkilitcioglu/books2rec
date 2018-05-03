import requests
from xml.etree import ElementTree
import os
import sys
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

# Custom libraries
import secret # need to make this and add goodreads_api key

def get_id_from_username(username, api_key):
    response = requests.get('https://www.goodreads.com/user/show/?key='+api_key+'&username='+username+'&format=xml')
    tree = ElementTree.fromstring(response.content)
    try:
        user_id = tree.find('user').find('id').text
        return user_id
    except:
        # raise ValueError('Invalid Goodreads username, not id returned')
        return None

def get_user_vector(user_input, books, mapper):
    try:
        sparse_q = scipy.sparse.load_npz('static/data/cached_users/user_'+user_input+'.npz')
        q = sparse_q.toarray()
        q = np.array(q[0].tolist())
        print('found user_vector...')
        return q
    except:
        q = np.zeros((10000), dtype = np.int)
        api_key = secret.API_KEY
        if not user_input.isdigit():
            user_id = get_id_from_username(user_input, api_key)
        else:
            user_id = user_input
        
        if user_id is None:
            return None

        page = 1
        while True:
            response = requests.get('https://www.goodreads.com/review/list/?v=2&id='+user_id+'&shelf=read&format=xml&key='+api_key+'&per_page=200&page=' + str(page))
            tree = ElementTree.fromstring(response.content)
            reviews = tree.find('reviews')
            for review in reviews:
                goodreads_book_id = str(review.find('book').find('id').text)
                if goodreads_book_id in mapper:
                    book_id = int(mapper[goodreads_book_id])
                    rating = int(review.find('rating').text)
                    q[book_id-1] = float(rating)
            page += 1
            
            print(len(reviews))
            if len(reviews) < 1:
                break

        for i in range(len(q)):
            if q[i] != 0:
                title = books.iloc[i]['title']
                print("%s --> %s" % (q[i], title))
        
        # Turn 1-5 rating scale into negative - positive scale
        ratings_mapper = {0:0, 1:-2, 2:-1, 3:1, 4:2, 5:3}
        for i in range(len(q)):
            q[i] = ratings_mapper[q[i]]

        print('saving user_vector...')
        scipy.sparse.save_npz('static/data/cached_users/user_'+user_input+'.npz', scipy.sparse.csr_matrix(q))
        return q

'''

Recommender functions

'''
def chunker(top_books):
    # chunk into groups of 3 to display better in web app
    chunks = []
    current_chunk = []
    for i in range(len(top_books)):
        if len(current_chunk) < 3:
            current_chunk.append(top_books[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [top_books[i]]

    chunks.append(current_chunk)
    return chunks

def get_books_from_indices(top_book_indices, books):
    top_books = []
    for i in range(len(top_book_indices)):
        book_id = top_book_indices[i]
        book = books.iloc[book_id - 1] #index is book_id - 1
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)
    return top_books

def get_top_n_recs(result, books, n, q):
    recs = []
    for i in range(len(result)):
        if q[i] == 0: # book user hasn't already rated
            recs.append((i, result[i]))
        else:
            recs.append((i, float('-inf')))
    recs = sorted(recs, key=lambda tup: tup[1], reverse=True)

    top_books = []
    for i in range(n):
        book_id = recs[i][0]
        book = books.iloc[book_id]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)
    
    return top_books

def most_popular(books, n):
    top_books = []
    for i in range(n):
        book = books.iloc[i]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    return top_books

def map_user(q, V):
    # map new user to concept space by q*V
    user_to_concept = np.matmul(q, V)
    # map user back to itme space with user_to_concept * VT
    result = np.matmul(user_to_concept, V.T)
    return result
