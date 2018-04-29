from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
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
cosine_sim_item_matrix = None
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

def get_recommendation(title):
    global books, item_matrix, cosine_sim_item_matrix, bookid_to_title, title_to_bookid

    similarities= [cosine_sim_item_matrix]
    weights = [1]
    recs = get_recommendations(books, bookid_to_title, title_to_bookid, title, similarities, weights)
    return recs

@app.route('/')
def my_form():
    global books

    load_books_mappers()
    books = get_book_dataframe(data_path)
    load_item_matrix()
    return render_template('book_list.html')

@app.route('/', methods=['POST', 'POST2'])
def my_form_post():
    global item_matrix, books, error_message, title_to_bookid

    if 'book_recs' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                            error=error_message)

        recs = get_recommendation(text)
        return render_template('book_list.html', 
                        toPass=recs)

    if 'user_recs' in request.form:
        text = request.form['text']

        q = get_user_vector(text)
        if q is None:
            return render_template('book_list.html',
                            error=error_message)

        top_books = get_top_n_recs(map_user(q, item_matrix), books, 30, q)
        return render_template('book_list.html', 
                        toPass=top_books)
    else:
        return 'ERROR'

if __name__ == '__main__':
    app.run()