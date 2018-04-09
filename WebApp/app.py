from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

# Custom files
import recommender

app = Flask(__name__)

data_path = '../../goodbooks-10k/'
bookid_to_title = {}
title_to_bookid = {}
item_matrix = None
cosine_sim_item_matrix = None

def load_books():
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
    global item_matrix, cosine_sim_item_matrix

    similarities= [cosine_sim_item_matrix]
    weights = [1]
    recs = recommender.get_recommendations(bookid_to_title, title_to_bookid, title, similarities, weights)
    return recs

@app.route('/')
def my_form():
    load_books()
    load_item_matrix()
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    rec = get_recommendation(text)
    print(rec)

    rec_string = ''
    for r in rec:
        rec_string += r + ', '

    return rec_string

if __name__ == '__main__':
    app.run()