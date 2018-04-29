
import sys
import numpy as np
import pandas as pd
import scipy

# Custom libraries
from loader import get_books, get_book_dataframe, get_book_features, get_book_authors
from reduction import reduce_matrix, get_sparse

def load_goodreads(data_path):
    # Read in goodreads data
    ratings_goodreads = pd.read_csv(data_path)
    ratings_goodreads['book_id'] = ratings_goodreads['book_id'].astype(int)
    return ratings_goodreads

def load_amazon(data_path, min_items = 5):
    # Load in amazon ratings
    ratings_amazon = pd.read_csv(data_path)
    ratings_amazon['book_id'] = ratings_amazon['book_id'].astype(int)
    ratings_amazon = ratings_amazon.drop_duplicates(subset = ['book_id', 'user_id'])
    # Get a set of users that rated 5 or more items
    user_counts = ratings_amazon['user_id'].value_counts() >= min_items
    to_drop = set()
    for key,value in user_counts.items():
        if not value:
            to_drop.add(key)
    # drop users from df that don't have 5 or more ratings
    return ratings_amazon[~ratings_amazon['user_id'].isin(to_drop)]

def combine_ratings(goodreads, amazon):
    # combine goodreads ratings with amazon ratings (the 10k books)
    df = goodreads.append(amazon)
    # only need to sort by book_id bc we use the transpose of the User-Item matrix in SVD
    df = df.sort_values(by=['book_id'])
    return df

def get_ratings(goodreads_path, amazon_path, min_amazon_items = 5):
    try:
        ratings_combined = scipy.sparse.load_npz('../.tmp/ratings_combined.npz')
        print('ratings_combined found in file...')
        return ratings_combined
    except:
        goodreads = load_goodreads(goodreads_path)
        amazon = load_amazon(amazon_path, min_items = min_amazon_items)
        ratings_combined = get_sparse(combine_ratings(goodreads, amazon))
        scipy.sparse.save_npz('../.tmp/ratings_combined', ratings_combined)
        return ratings_combined

def get_ratings_pickle(data_path):
    try:
        # Load ratings DF
        ratings = pd.read_pickle('../.tmp/ratings_pickle')
        print('ratings_pickle existed as file...')
        return ratings
    except:
        ratings_goodreads = load_goodreads(data_path + 'ratings.csv')
        ratings_amazon = load_amazon(data_path + 'ratings_amazon.csv')
        # Map amazon userid's to unique ints
        seen = {}
        next_uid = 53424 + 1 # 53424 is last user in goodreads
        for index, row in ratings_amazon.iterrows():
            username = row['user_id']
            if username not in seen:
                seen[username] = next_uid
                next_uid += 1
                
            ratings_amazon.set_value(index,'user_id',seen[username])

        # Join Goodreads and Amazon
        ratings = pd.concat([ratings_goodreads, ratings_amazon])

        # prepare to be read in to Reader
        ratings = ratings.sort_values(by=['user_id','book_id'])
        ratings = ratings.reset_index(drop=True)
        ratings = ratings.rename(index=str, columns={"user_id": "userID", "book_id": "itemID", "rating": "rating"})

        ratings.to_pickle('../.tmp/ratings_pickle')
        return ratings

def get_joint(item_ratings, item_features, ratings_components = 1000, features_components = 1000, weight=False):
    ratings_U, _, _ = reduce_matrix(item_ratings, n_components = ratings_components)
    features_U, _, _ = reduce_matrix(item_features, n_components = features_components)

    # Normalize each
    ratings_U = normalize(ratings_U)
    features_U = normalize(features_U)

    if weight:
        max_num_components = max(ratings_components, features_components)
        ratings_U *= (max_num_components / ratings_components)
        features_U *= (max_num_components / features_components)

    return np.hstack((ratings_U, features_U))

def get_reduced_joint(reduced_item_ratings, reduced_item_features):
    return np.hstack((reduced_item_ratings, reduced_item_features))

def normalize(df):
    # we should improve this using variance but for now divide by largest value seen
    max_value = np.max(df)
    df_normalized = df / max_value
    return df_normalized