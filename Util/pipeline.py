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
        user_arr = np.asarray(X_held_out[u, :].todense())[0]
        norm_const = np.sum((user_arr != 0) * (sim[i,:]))
        pred_rating = np.sum(user_arr * sim[i,:])/norm_const
        predictions.append(pred_rating)
    rmse_ = rmse(held_outs, np.array(predictions))
    mae_ = mae(held_outs, np.array(predictions))
    return np.array([rmse_, mae_])


def feature_scaling(q, copy=False):
    """ Scales the user features using the mean and the
    standard deviation.

    TODO: merge this with the one from WebApp
    """
    if q.dtype != np.float:
        q = q.astype(np.float)
    if copy:
        q = q.copy()
    nonzero = np.nonzero(q)
    nonzero_ratings = q[nonzero]
    mean = np.mean(nonzero_ratings)
    std = np.std(nonzero_ratings)
    print('Mean: %s' % (mean))
    print('S.D: %s' % (std))
    q[nonzero] = (1.0 + q[nonzero] - mean) / std
    return q


def map_user(q, V):
    """ Maps a user to item space.

    Args:
        q::np.array
            the user vector
        V::np.array
            the item features vector

    Returns:
        res::np.array
            the mapped vector in the item-space
            in our case, this is a 10000 dim.
            array where each element is the projected
            rating for each item
    """
    # map new user to concept space by q*V
    user_to_concept = np.matmul(q, V)
    # map user back to item space with user_to_concept * VT
    result = np.matmul(user_to_concept, V.T)
    return result


def get_top_n_recs(results, q, n):
    """ Gets the top n results that the user has not rated

    Args:
        results::np.array
            the mapped vector in the item-space
            see map_user function
        q::np.array
            the original user vector
        n::int
            the # of items to recommend

    Returns:
        recs::np.array
            the recommended item ids in order
    """
    candidates = np.where(q == 0)[0]
    estimates = results[candidates]
    top_estimates = np.argsort(estimates)[-n:][::-1]
    return candidates[top_estimates]


def evaluate_rankings(X_held_out, item_concept_mat, user_indices, item_indices):
    user_ind_prev = -1
    user_arr = None
    user_recs = None
    reciprocal_ranks = []
    for user_ind, item_ind in zip(user_indices, item_indices):
        if user_ind != user_ind_prev:
            user_arr = np.asarray(X_held_out[user_ind, :].todense())[0]
            user_recs = get_top_n_recs(map_user(user_arr, item_concept_mat), user_arr, 10000)
        rank = np.where(user_recs == item_ind)[0][0] + 1
        reciprocal_ranks.append(1 / rank)
    mrr = np.mean(reciprocal_ranks)
    return np.array([mrr])


def print_evaluation(rmse_, mae_, mrr_):
    print('RMSE: {0:4.3f} MAE: {1:4.3f} MRR:{2:4.3f}'.format(rmse_, mae_, mrr_))
