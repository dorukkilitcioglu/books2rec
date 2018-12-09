from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import scipy
import logging
import pickle
import csv

# Custom libraries
from util import get_user_vector, chunker, not_found_error_message
from recommendations import get_top_n_recs, map_user, map_user_to_features, most_popular, get_books_from_indices, partial_fit, log_rank

app = Flask(__name__)


''' GLOBALS
'''
bookid_to_title = None
title_to_bookid = None
mapper_id = None
feature_matrix = None
Q = None  # item to concept matrix from SVD
item_bias = None
top_recs_each_book_item_matrix = None
top_recs_each_book_feature_matrix = None
books = None
titles = None


''' DATA LOADING FUNCTIONS
'''


def load_books():
    """ Loads in the books and titles from the pickled dataframe """
    global books, titles
    if books is None or titles is None:
        titles = []
        books = pd.read_pickle('static/data/books_dataframe')
        for index, row in books.iterrows():
            titles.append(row['title'])
        titles.sort()
        print('books loaded')


def load_title_mappers():
    """ Loads in the title mappers using books.csv """
    global bookid_to_title, title_to_bookid
    if bookid_to_title is None or title_to_bookid is None:
        bookid_to_title = {}
        title_to_bookid = {}
        filename = 'static/data/books.csv'
        with open(filename, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                bookid = line[0]
                title = line[10].strip()
                bookid_to_title[bookid] = title
                title_to_bookid[title] = bookid
        print('books mapper loaded')


def load_id_mapper():
    """ Loads in the id mapper using books.csv.

    This maps goodreads book ids to our ids.
    """
    global mapper_id
    if mapper_id is None:
        mapper_id = {}
        filename = 'static/data/books.csv'
        with open(filename, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                mapper_id[line[1]] = line[0]
        print('mapper_id loaded')


def load_feature_matrix():
    """ Loads in the item to feature matrix """
    global feature_matrix
    if feature_matrix is None:
        feature_matrix = scipy.sparse.load_npz('static/data/feature_matrix.npz')
        print('feature matrix loaded')


def load_top_recs_each_book():
    global top_recs_each_book_item_matrix, top_recs_each_book_feature_matrix
    if top_recs_each_book_feature_matrix is None or top_recs_each_book_item_matrix is None:
        f = open('static/data/top_recs_each_book_item_matrix.pkl', "rb")
        top_recs_each_book_item_matrix = pickle.load(f)
        f.close()

        f = open('static/data/top_recs_each_book_feature_matrix.pkl', "rb")
        top_recs_each_book_feature_matrix = pickle.load(f)
        f.close()
        print('top recs for each book loaded')


def load_Q_matrix():
    global Q
    if Q is None:
        Q = np.load('static/data/Q.npy')
        print('Trained Q matrix loaded')


def load_item_bias():
    global item_bias
    if item_bias is None:
        item_bias = np.load('static/data/item_bias.npy')
        print('Trained item_bias vector loaded')


def load_data():
    global titles
    load_title_mappers()
    load_id_mapper()
    load_Q_matrix()
    load_item_bias()
    load_books()
    load_feature_matrix()
    load_top_recs_each_book()
    return render_template('book_list.html', titles=titles)


''' HOME PAGE
'''


@app.route('/')
def home():
    return render_template('start.html')


@app.route('/', methods=['POST'])
def home_post():
    if 'load' in request.form:
        return redirect(url_for('recommender'))


''' SETUP PAGE
'''


@app.route('/setup')
def setup():
    return render_template('setup.html')


@app.route('/setup', methods=['POST'])
def setup_post():
    if 'load' in request.form:
        return redirect(url_for('recommender'))


''' RECOMMENDER PAGE
'''


@app.route('/recommender')
def recommender():
    return load_data()


@app.route('/recommender', methods=['POST'])
def recommender_post():
    global item_matrix, books, title_to_bookid, cosine_sim_item_matrix, cosine_sim_feature_matrix
    if 'book_recs' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                                   response=not_found_error_message,
                                   titles=titles)

        book_id = int(title_to_bookid[text])
        top_book_indices = top_recs_each_book_item_matrix[book_id]
        top_books = get_books_from_indices(top_book_indices, books)
        chunks = chunker(top_books)
        return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               response='Showing Recommendations for: ' + text)

    if 'book_similar' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                                   response=not_found_error_message,
                                   titles=titles)

        book_id = int(title_to_bookid[text])
        top_book_indices = top_recs_each_book_feature_matrix[book_id]
        top_books = get_books_from_indices(top_book_indices, books)
        chunks = chunker(top_books)
        return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               response='Showing Similar Books to: ' + text)

    if 'user_recs' in request.form:
        text = request.form['text']

        user_ratings, error_message = get_user_vector(text, mapper_id)
        if error_message:
            return render_template('book_list.html',
                                   response=error_message,
                                   titles=titles)


        # Combine partial fit results with content based recs
        global_bias = 3.919866
        predictions_partial_fit = partial_fit(user_ratings, Q, item_bias, global_bias)
        predictions_features = map_user_to_features(user_ratings, feature_matrix)
        top_books = log_rank(predictions_partial_fit, predictions_features, user_ratings, books, weight_feature=0.5, num_books=99)

        chunks = chunker(top_books)

        return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               response='Showing Recommendations for: ' + text)

    if 'most_popular' in request.form:
        top_books = most_popular(books, 99)
        chunks = chunker(top_books)
        return render_template('book_list.html',
                               toPass=chunks,
                               titles=titles,
                               response='99 Most Popular Books')
    else:
        return 'ERROR'


if __name__ == '__main__':
    app.run()
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
