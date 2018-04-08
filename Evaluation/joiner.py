import numpy as np
import pandas as pd
from util import reduce_matrix, get_sparse
from books import get_book_dataframe, get_book_features

def load_goodreads(data_path):
    # Read in goodreads data
    ratings_goodreads = pd.read_csv(data_path)
    ratings_goodreads['book_id'] = ratings_goodreads['book_id'].astype(int)
    return ratings_goodreads

def load_amazon(data_path, min_items = 5):
    # Load in amazon ratings
    ratings_amazon = pd.read_csv(data_path)
    ratings_amazon['book_id'] = ratings_amazon['book_id'].astype(int)
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
    df = df.sort_values(by=['book_id'])
    return df

def get_ratings(goodreads_path, amazon_path, min_amazon_items = 5):
    goodreads = load_goodreads(goodreads_path)
    amazon = load_amazon(amazon_path, min_items = min_amazon_items)
    return get_sparse(combine_ratings(goodreads, amazon))

def get_joint(item_ratings, item_features, ratings_components = 1000, features_components = 1000):
    _, _, rating_VT = reduce_matrix(item_ratings, n_components = ratings_components)
    features_U, _, _ = reduce_matrix(item_features, n_components = features_components)
    return np.hstack((rating_VT.T, features_U))

def main():
    # Set this to where you save and load all data
    goodreads_path = '../data/goodbooks-10k/ratings.csv'
    amazon_path = '../data/amazon/ratings_amazon.csv'
    ratings = get_ratings(goodreads_path, amazon_path)

    data_path = '../data/goodbooks-10k/'
    book_features = get_book_features(get_book_dataframe(data_path))

    joint = get_joint(ratings, book_features, 30, 30)
    print(joint.shape)

if __name__ == '__main__':
    main()
