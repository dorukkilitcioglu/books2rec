import os
import sys
import random
import math
import numpy as np
import pandas as pd
import scipy

def get_books_from_indices(top_book_indices, books):
    top_books = []
    for i in range(len(top_book_indices)):
        book_id = top_book_indices[i]
        book = books.iloc[book_id - 1]  # index is book_id - 1
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)
    return top_books


def most_popular(books, num_results):
    top_books = []
    for i in range(num_results):
        book = books.iloc[i]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    return top_books


def map_user_to_features(p, features):
    p_sparse = scipy.sparse.csr_matrix(p)
    # map new user to concept space by p*features
    user_to_concept = p_sparse.dot(features)
    # map user back to itme space with user_to_concept * featuresT
    result = user_to_concept.dot(features.T).todense()
    return result.T


def get_predictions(p, q, user_bias, item_bias, global_bias):
    pred_ratings = np.zeros(len(q))
    for i in range(len(q)):
        pred = global_bias + user_bias + item_bias[i] + np.dot(p, q[i])
        pred_ratings[i] = pred
    return pred_ratings


def partial_fit(new_user_ratings, q, item_bias, global_bias):

    # create array of indices of books this user has actually rated
    indices = []
    for i in range(len(new_user_ratings)):
        if new_user_ratings[i] != 0:
            indices.append(i)

    # Hyperparams
    learning_rate = 0.07
    user_bias_reg = 0.001
    P_reg = 0.001

    # 5 updates per rated book
    iterations = 5

    # dimensions
    n_factors = q.shape[1]

    # init components
    mu, sigma = 0, 0.1
    p = np.random.normal(mu, (sigma / n_factors), n_factors)
    new_user_bias = 0

    # Compute gradient descent
    for iteration in range(iterations):
        for i in indices:
            rating = new_user_ratings[i]
            pred = global_bias + new_user_bias + item_bias[i] + np.dot(p, q[i])
            error = rating - pred

            # update P
            for f in range(n_factors):
                p_update = learning_rate * (error * q[i][f] - P_reg * p[f])
                p[f] += p_update

            # update user bias
            ub_update = learning_rate * (error - user_bias_reg * new_user_bias)
            new_user_bias += ub_update

    return get_predictions(p, q, new_user_bias, item_bias, global_bias)


def log_rank(predictions_partial_fit, predictions_features, user_ratings, books, weight_feature, num_results):

    # create tuple of book_id and rating for each method, then sort
    partial_fit_ratings = []
    feature_ratings = []
    for i in range(len(books)):
        partial_fit_ratings.append((i, predictions_partial_fit[i]))
        feature_ratings.append((i, predictions_features[i]))
    
    partial_fit_ratings = sorted(partial_fit_ratings, key=lambda x: x[1], reverse=True)
    feature_ratings = sorted(feature_ratings, key=lambda x: x[1], reverse=True)

    # map book_id to the rank for each method
    id_to_rank_partial_fit = {}
    id_to_rank_features = {}
    for i in range(len(books)):
        book_id = partial_fit_ratings[i][0]
        id_to_rank_partial_fit[book_id] = math.log(i+1)

        book_id = feature_ratings[i][0]
        id_to_rank_features[book_id] = math.log(i+1)

    rankings = []
    for i in range(len(books)):
        if user_ratings[i] == 0: # ignore if user has rated this book already
            rank = weight_feature*id_to_rank_features[i] + (1.0-weight_feature)*id_to_rank_partial_fit[i]
            rankings.append((rank, i))
    rankings = sorted(rankings, key=lambda x: x[0])
    print("Number of non-rated books: {}".format(len(rankings)))

    top_books = []
    for i in range(num_results):
        book_id = rankings[i][1]
        book = books.iloc[book_id] # index is book_id - 1
        book['rank'] = i + 1
        top_books.append(book)
    return top_books