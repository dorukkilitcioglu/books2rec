import os
import sys
import html
import re
import numpy as np
import pandas as pd

# Custom libraries
sys.path.append('../Util')
from loader import get_books, get_book_dataframe, get_book_features, get_book_authors
from reduction import reduce_matrix

def main():
    """ Sample program to verify the code.

    This method will load in the book features, do some preprocessing,
    and use SVD to reduce it to 100 dimensions. It will then output
    the top 10 singular values.
    """
    # Set this to where you save and load all data
    # data_path = '../data/goodbooks-10k/'
    data_path = '../../goodbooks-10k/'
    df = get_book_dataframe(data_path)
    fv = get_book_features(df)
    U, S, VT = reduce_matrix(fv, 100, random_state = 42)
    print(S[:10])

if __name__ == '__main__':
    main()
