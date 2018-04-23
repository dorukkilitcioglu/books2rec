import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from cross_validation import ColumnwiseKFold
from books import get_book_dataframe, get_book_features
from util import get_sparse, reduce_matrix
from joiner import get_ratings, get_reduced_joint

def rmse(y_true, y_pred):
    """ Returns the root mean squared error """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def mae(y_true, y_pred):
    """ Returns the mean absolute error """
    return np.mean(np.absolute(y_true - y_pred))

def evaluate(X_orig, X_held_out, sim, user_indices, item_indices):
    held_outs = X_orig[(user_indices, item_indices)]
    predictions = []
    for u, i in zip(user_indices, item_indices):
        user_arr = np.asarray(X_held_out[u,:].todense())[0]
        norm_const = np.sum((user_arr != 0) * (sim[i,:]))
        pred_rating = np.sum(user_arr * sim[i,:])/norm_const
        predictions.append(pred_rating)
    rmse_ = rmse(held_outs, np.array(predictions))
    mae_ = mae(held_outs, np.array(predictions))
    return np.array([rmse_, mae_])

def print_evaluation(rmse_, mae_):
    print('RMSE: {0:4.3f} MAE: {1:4.3f}'.format(rmse_, mae_))

def main(ratings_components = 100, features_components = 100, print_scores = False):
    data_path = '../data/goodbooks-10k/'
    book_features = get_book_features(get_book_dataframe(data_path))
    reduced_item_features, _, _ = reduce_matrix(book_features, n_components = features_components)

    goodreads_path = '../data/goodbooks-10k/ratings.csv'
    amazon_path = '../data/amazon/ratings_amazon.csv'
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
