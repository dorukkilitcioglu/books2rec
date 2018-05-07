
import sys
import numpy as np
import pandas as pd

# Custom libraries
sys.path.append('../Util')
from loader import get_books, get_book_dataframe, get_book_features, get_book_authors
from reduction import reduce_matrix, get_sparse
from joiner import get_ratings, get_joint

def main(): 
    """ 
    Sample program to verify the code.
    This method will load and join ratings
    """
    
    # Set this to where you save and load all data
    # data_path = '../data/goodbooks-10k/'
    data_path = '../../goodbooks-10k/'
    goodreads_path = data_path + 'ratings.csv'
    amazon_path = data_path + 'ratings_amazon.csv'
    ratings = get_ratings(goodreads_path, amazon_path)

    book_features = get_book_features(get_book_dataframe(data_path))

    joint = get_joint(ratings.T, book_features, 30, 30)
    print(joint.shape)

if __name__ == '__main__':
    main()
