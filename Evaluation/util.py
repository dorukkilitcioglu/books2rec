import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

bad_features = {
    'to-read',
    'currently-reading',
    'books-i-own',
    'owned',
    'owned-books',
    'read',
    'favourites',
    'default',
    'kindle',
    'my-books',
    'to-buy',
    'all-time-favorites',
    're-read',
    'i-own',
    'ebook',
    'on-hold',
    'favorite',
    'favorites'
}

def reduce_matrix(X, n_components = 1000, n_iter = 7, random_state = None):
    """ Uses SVD to reduce a matrix into its components
    
    Args:
        X:              the matrix to reduce
        n_components:   number of singular values to limit to
        n_iter:         the number of iterations for SVD
        random_state:   the random initial state SVD

    Returns:
        U: the user representations
        S: the singular values
        V: the item representations
    """
    svd = TruncatedSVD(n_components = n_components, n_iter = n_iter, random_state = random_state)
    reduced_matrix = svd.fit_transform(X)
    return reduced_matrix, svd.singular_values_, svd.components_

def get_sparse(ratings):
    users = list(ratings.user_id.unique())
    books = list(ratings.book_id.unique())
    data = ratings['rating'].tolist()
    row = ratings.user_id.astype('category', categories=users).cat.codes
    col = ratings.book_id.astype('category', categories=books).cat.codes
    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(books)), dtype = np.dtype('u1'))
    return sparse_matrix