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
from surprise import Reader, Dataset, SVD, evaluate, dump, accuracy
from collections import defaultdict

# Custom libraries
sys.path.append('../Util')
from loader import get_books, get_book_dataframe, get_book_features, get_mapper
from joiner import get_ratings, get_joint
from reduction import reduce_matrix, get_sparse
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

def get_user_vector(user_input):
    try:
        q = np.load('../.tmp/user_'+user_input+'.npy')
        print('found user_vector...')
        return q
    except:
        # Set this to where you save and load all data
        data_path = '../../goodbooks-10k/'

        # Get dataframe from books
        books = get_book_dataframe(data_path)

        mapper = get_mapper(data_path + 'books.csv')

        # make an array for myself
        q = np.zeros((10000), dtype = np.int)

        # username = secret.USERNAME
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
        np.save('../.tmp/user_'+user_input, q)
        
        return q