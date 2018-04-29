from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys

# Custom libraries
sys.path.append('../Util')
from loader import get_book_dataframe, get_book_features
from recommender import get_recommendations, get_top_n_recs, map_user, map_user_sparse

app = Flask(__name__)

data_path = '../../goodbooks-10k/'
bookid_to_title = {}
title_to_bookid = {}
item_matrix = None
cosine_sim_item_matrix = None
books = None

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

    filename = data_path + 'item_matrix.npy'
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
    global item_matrix, books
    if 'book_recs' in request.form:
        print('getting recs')

        text = request.form['books']

        print('got title')
        recs = get_recommendation(text)
        res = render_template('book_list.html', 
                        toPass=recs)
        return res

    if 'user_recs' in request.form:
        text = request.form['text']

        q = np.load('../.tmp/user_vector.npy')
        # Turn 1-5 rating scale into negative - positive scale
        ratings_mapper = {0:0, 1:-2, 2:-1, 3:1, 4:2, 5:3}
        for i in range(len(q)):
            q[i] = ratings_mapper[q[i]]

        top_books = get_top_n_recs(map_user(q, item_matrix), books, 25, q)

        res = render_template('book_list.html', 
                        toPass=top_books)
        return res
    else:
        return 'ERROR'

if __name__ == '__main__':
    app.run()