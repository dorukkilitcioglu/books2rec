import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Custom libraries
sys.path.append('../Util')
from loader import get_books, get_book_dataframe, get_book_features, get_book_authors
from reduction import reduce_matrix, get_sparse
from joiner import get_ratings, get_reduced_joint
from pipeline import rmse, mae, evaluate, print_evaluation
from cross_validation import ColumnwiseKFold

def main(ratings_components = 100, features_components = 100, print_scores = False):
    #data_path = '../data/goodbooks-10k/'
    data_path = '../../goodbooks-10k/'
    book_features = get_book_features(get_book_dataframe(data_path))
    reduced_item_features, _, _ = reduce_matrix(book_features, n_components = features_components)

    goodreads_path = data_path + 'ratings.csv'
    amazon_path = data_path + 'ratings_amazon.csv'
    spr = get_ratings(goodreads_path, amazon_path, min_amazon_items = 6)

    n_folds = 5
    scores = np.zeros((n_folds, 2))
    kf = ColumnwiseKFold(n_folds, random_seed = 30)
    for i, (X, (user_incides, item_indices)) in enumerate(kf.split(spr)):
        _, _, rating_VT = reduce_matrix(X, n_components = ratings_components)
        reduced_item_ratings = rating_VT.T
        items = get_reduced_joint(reduced_item_ratings, reduced_item_features)
        sim = (cosine_similarity(items) + 1) / 2
        scores[i,:] = evaluate(spr, X, sim, user_incides, item_indices)
        if print_scores:
            print_evaluation(scores[i,0], scores[i,1])

    scores = np.mean(scores, axis = 0)
    if print_scores:
        print('{0:d}-Fold Scores:')
        print_evaluation(scores[0], scores[1])

    return scores

if __name__ == '__main__':
    main(print_scores = True)
