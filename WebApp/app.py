from flask import Flask, request, render_template, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate, dump, accuracy
import numpy as np
import pandas as pd
import pickle
import csv
import sys

# Custom libraries
from util import get_user_vector, chunker, get_top_n_recs, map_user, most_popular, get_books_from_indices

app = Flask(__name__)


''' GLOBALS
'''
data_loaded = False
bookid_to_title = None
title_to_bookid = None
mapper_id = None
item_matrix = None
qi = None #item to concept matrix from SVD
top_recs_each_book_item_matrix = None
top_recs_each_book_feature_matrix = None
books = None
titles = None
error_message = "I can't seem to find anything with what you gave me, I'm sorry"


''' DATA LOADING FUNCTIONS
'''
def load_books():
    global books, titles
    titles = []
    books = pd.read_pickle('static/data/books_dataframe')
    for index, row in books.iterrows():
        titles.append(row['title'])
    titles.sort()
    print('books loaded')

def load_title_mappers():
    global bookid_to_title, title_to_bookid
    bookid_to_title = {}
    title_to_bookid = {}
    filename = 'static/data/books.csv'
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            bookid_to_title[line[0]] = line[10]
            title_to_bookid[line[10]] = line[0]
    print('books mapper loaded')

def load_id_mapper():
    global mapper_id
    mapper_id = {}
    filename = 'static/data/books.csv'
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            mapper_id[line[1]] = line[0]
    print('mapper_id loaded')

def load_item_matrix():
    global item_matrix
    item_matrix = np.load('static/data/item_matrix.npy')
    print('item matrix loaded')

def load_top_recs_each_book():
    global top_recs_each_book_item_matrix, top_recs_each_book_feature_matrix
    f = open('static/data/top_recs_each_book_item_matrix.pkl',"rb")
    top_recs_each_book_item_matrix = pickle.load(f)
    f.close()

    f = open('static/data/top_recs_each_book_feature_matrix.pkl',"rb")
    top_recs_each_book_feature_matrix = pickle.load(f)
    f.close()
    print('top recs for each book loaded')

def load_svd_matrix():
    global qi
    qi = np.load('static/data/svd.npy')
    print('svd matrix loaded')

def load_data():
    global books, titles, data_loaded
    load_title_mappers()
    load_id_mapper()
    load_svd_matrix()
    load_books()
    load_item_matrix()
    load_top_recs_each_book()
    data_loaded = True
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


''' RECOMMENDER PAGE
'''
@app.route('/recommender')
def recommender():
    global books, titles, data_loaded
    if not data_loaded:
        print('loading data')
        return load_data()
    else:
        print('data already loded in server')
        return render_template('book_list.html', titles=titles)

@app.route('/recommender', methods=['POST'])
def recommender_post():
    global item_matrix, books, error_message, title_to_bookid, cosine_sim_item_matrix, cosine_sim_feature_matrix
    if 'book_recs' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                                    response=error_message,
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
                                    response=error_message,
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

        q = get_user_vector(text, books, mapper_id)
        if q is None:
            return render_template('book_list.html',
                                    response=error_message,
                                    titles=titles)

        # Get recs using item_matrix
        top_books = get_top_n_recs(map_user(q, item_matrix), books, 99, q)
        chunks = chunker(top_books)

        # Get recs using svd
        # top_books = get_top_n_recs(map_user(q, qi), books, 99, q)

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