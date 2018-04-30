from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate, dump, accuracy
import numpy as np
import csv
import sys

# Custom libraries
sys.path.append('../Util')
from loader import get_book_dataframe, get_book_features
from recommender import get_recommendations, get_top_n_recs, map_user, map_user_sparse
from goodreads_data import get_user_vector

app = Flask(__name__)

data_path = '../../goodbooks-10k/'
bookid_to_title = {}
title_to_bookid = {}
item_matrix = None
svd = None
cosine_sim_item_matrix = None
cosine_sim_svd = None
books = None
error_message = "I can't seem to find anything with what you gave me, I'm sorry"

def load_books_mappers():
    global bookid_to_title, title_to_bookid
    filename = data_path + 'books.csv'
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            bookid_to_title[line[0]] = line[10]
            title_to_bookid[line[10]] = line[0]

def load_item_matrix():
    global item_matrix, cosine_sim_item_matrix

    filename = '../.tmp/item_matrix.npy'
    item_matrix = np.load(filename)
    cosine_sim_item_matrix = cosine_similarity(item_matrix)

def load_svd_matrix():
    global svd, cosine_sim_svd

    filename = '../.tmp/svd_50_10'
    svd = dump.load(filename)[1]
    cosine_sim_svd = cosine_similarity(svd.qi)

@app.route('/')
def my_form():
    global books

    load_books_mappers()
    print('book title mapper loaded')

    # load_svd_matrix()
    # print('svd loaded')

    books = get_book_dataframe(data_path)
    print('books loaded')

    load_item_matrix()
    print('item_matrix loaded')
    return render_template('book_list.html')

@app.route('/', methods=['POST', 'POST2'])
def my_form_post():
    global item_matrix, books, error_message, title_to_bookid, cosine_sim_item_matrix

    if 'book_recs' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                            error=error_message)

        recs = get_recommendations(books, bookid_to_title, title_to_bookid, text, [cosine_sim_item_matrix], [1])
        return render_template('book_list.html', 
                        toPass=recs)

    if 'user_recs' in request.form:
        text = request.form['text']

        q = get_user_vector(text)
        if q is None:
            return render_template('book_list.html',
                            error=error_message)

        # Get recs using item_matrix
        top_books = get_top_n_recs(map_user(q, item_matrix), books, 30, q)

        # Get recs using svd
        # top_books = get_top_n_recs(map_user(q, svd.qi), books, 30, q)

        return render_template('book_list.html', 
                        toPass=top_books)
    else:
        return 'ERROR'

if __name__ == '__main__':
    app.run()