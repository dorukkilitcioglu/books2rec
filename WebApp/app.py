from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate, dump, accuracy
import numpy as np
import csv
import sys

# Custom libraries
sys.path.append('../Util')
from loader import get_book_dataframe, get_book_features
from recommender import get_recommendations, get_top_n_recs, map_user, map_user_sparse, most_popular
from goodreads_data import get_user_vector

app = Flask(__name__)

data_path = '../../goodbooks-10k/'
data_loaded = False
bookid_to_title = None
title_to_bookid = None
item_matrix = None
qi = None
cosine_sim_item_matrix = None
cosine_sim_feature_matrix = None
books = None
titles = None
error_message = "I can't seem to find anything with what you gave me, I'm sorry"

def load_books():
    global books, titles
    titles = []
    books = get_book_dataframe(data_path)
    for index, row in books.iterrows():
        titles.append(row['title'])
    titles.sort()
    print('books loaded')

def load_books_mappers():
    global bookid_to_title, title_to_bookid
    bookid_to_title = {}
    title_to_bookid = {}
    filename = data_path + 'books.csv'
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            bookid_to_title[line[0]] = line[10]
            title_to_bookid[line[10]] = line[0]
    print('books mapper loaded')

def load_item_matrix():
    global item_matrix, cosine_sim_item_matrix, cosine_sim_feature_matrix
    item_matrix = np.load('../.tmp/item_matrix.npy')
    print('item matrix loaded')
    
    cosine_sim_item_matrix = np.load('../.tmp/cosine_sim_item_matrix.npy')
    cosine_sim_feature_matrix = np.load('../.tmp/cosine_sim_feature_matrix.npy')
    print('similarity matrices loaded')

def load_svd_matrix():
    global qi
    qi = np.load('../.tmp/svd_100_300.npy')
    print('svd matrix loaded')

def load_data():
    global books, titles, data_loaded
    load_books_mappers()
    # load_svd_matrix()
    load_books()
    load_item_matrix()
    data_loaded = True
    return render_template('book_list.html', titles=titles)

@app.route('/')
def my_form():
    global books, titles, data_loaded
    if not data_loaded:
        return render_template('start.html')
    else:
        return render_template('book_list.html', titles=titles)

@app.route('/', methods=['POST', 'POST2'])
def my_form_post():
    global item_matrix, books, error_message, title_to_bookid, cosine_sim_item_matrix, cosine_sim_feature_matrix
    if 'load' in request.form:
        print('loadin data')
        return load_data()

    if 'book_recs' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                                    response=error_message,
                                    titles=titles)

        recs = get_recommendations(books, bookid_to_title, title_to_bookid, text, [cosine_sim_item_matrix], [1])
        return render_template('book_list.html', 
                                toPass=recs,
                                titles=titles,
                                response='Showing Recommendations for: ' + text)
    
    if 'book_similar' in request.form:
        text = request.form['books']

        if text not in title_to_bookid:
            return render_template('book_list.html',
                                    response=error_message,
                                    titles=titles)

        recs = get_recommendations(books, bookid_to_title, title_to_bookid, text, [cosine_sim_feature_matrix], [1])
        return render_template('book_list.html', 
                                toPass=recs,
                                titles=titles,
                                response='Showing Similar Books to: ' + text)

    if 'user_recs' in request.form:
        text = request.form['text']

        q = get_user_vector(text)
        if q is None:
            return render_template('book_list.html',
                                    response=error_message,
                                    titles=titles)

        # Get recs using item_matrix
        top_books = get_top_n_recs(map_user(q, item_matrix), books, 30, q)

        # Get recs using svd
        # top_books = get_top_n_recs(map_user(q, qi), books, 30, q)

        return render_template('book_list.html', 
                                toPass=top_books,
                                titles=titles,
                                response='Showing Recommendations for: ' + text)
    if 'most_popular' in request.form:
        top_books = most_popular(books, 100)
        return render_template('book_list.html', 
                                toPass=top_books,
                                titles=titles,
                                response='100 Most Popular Books')
    else:
        return 'ERROR'

if __name__ == '__main__':
    app.run()