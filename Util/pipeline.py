import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Custom libraries
from loader import get_books, get_book_dataframe, get_book_features, get_book_authors
from reduction import reduce_matrix, get_sparse
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