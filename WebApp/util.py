import requests
from xml.etree import ElementTree
import os
import sys
import random
import math
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

# Custom libraries
import secret  # need to make this and add goodreads_api key

not_found_error_message = "That username doesn't seem to exist on Goodreads, I'm sorry"
private_error_message = "This user account is private, I'm sorry"
no_ratings_error_message = "You don't have any ratings on the books we have access to, I'm sorry"


def get_id_from_username(username, api_key):
    response = requests.get('https://www.goodreads.com/user/show/?key=' + api_key + '&username=' + username + '&format=xml')
    tree = ElementTree.fromstring(response.content)
    try:
        user_id = tree.find('user').find('id').text
        return user_id
    except:
        # raise ValueError('Invalid Goodreads username, not id returned')
        return None


def get_user_vector(user_input, mapper):
    """ Gets the user ratings vector of a user

    Args:
        user_input::str
            username of the user
        mapper::dict
            maps the goodreads book id to our ids

    Returns:
        user_vector::np.array
            an array of 10000 ratings for the given user
        error_message::str
            an error message string, if there is an error
    """
    try:
        sparse_q = scipy.sparse.load_npz('static/data/cached_users/user_' + user_input + '.npz')
        q = sparse_q.toarray()
        q = np.array(q[0].tolist())
        print('found user_vector...')
        return q, None
    except:
        q = np.zeros((10000), dtype = np.float)
        api_key = secret.API_KEY
        if not user_input.isdigit():
            user_id = get_id_from_username(user_input, api_key)
        else:
            user_id = user_input

        if user_id is None:
            return None, not_found_error_message

        page = 1
        total_valid_reviews = 0
        while True:
            response = requests.get('https://www.goodreads.com/review/list/?v=2&id=' + user_id + '&shelf=read&format=xml&key=' + api_key + '&per_page=200&page=' + str(page))
            tree = ElementTree.fromstring(response.content)
            reviews = tree.find('reviews')
            if reviews is None:
                return None, private_error_message
            for review in reviews:
                goodreads_book_id = str(review.find('book').find('id').text)
                if goodreads_book_id in mapper:
                    book_id = int(mapper[goodreads_book_id])
                    rating = int(review.find('rating').text)
                    q[book_id - 1] = float(rating)
                    total_valid_reviews += 1
            page += 1

            print(len(reviews))
            if len(reviews) < 1:
                break

        print("total valid reviews: %s" % (total_valid_reviews))
        if total_valid_reviews < 1:
            return None, no_ratings_error_message

        # TODO: turn off if using partial fit
        # q = feature_scaling(q)

        # Disable this until we find a 'smart' caching solution
        # print('saving user_vector...')
        # scipy.sparse.save_npz('static/data/cached_users/user_'+user_input+'.npz', scipy.sparse.csr_matrix(q))

        return q, None


def feature_scaling(q):
    """ Scales the user features using the mean and the
    standard deviation.
    """
    if q.dtype != np.float:
        q = q.astype(np.float)
    nonzero = np.nonzero(q)
    nonzero_ratings = q[nonzero]
    mean = np.mean(nonzero_ratings)
    std = np.std(nonzero_ratings)
    print('Mean: %s' % (mean))
    print('S.D: %s' % (std))
    q[nonzero] = (1.0 + q[nonzero] - mean) / std
    return q


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
