import numpy as np
from scipy.sparse import csr_matrix

class ColumnwiseKFold:
    def __init__(self, n_folds, shuffle = True, random_seed = None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, X):
        if self.random_seed:
            np.random.seed(self.random_seed)
        splits = np.array([self._get_splits(X, i) for i in range(X.shape[0])])
        for i in range(self.n_folds):
            X_c = X.copy()
            u_inds = np.repeat(np.arange(X_c.shape[0]), [a.shape[0] for a in splits[:,i]])
            i_inds = np.hstack(splits[:,i])
            X_c[(u_inds, i_inds)] = 0
            yield X_c, (u_inds, i_inds)
            
    def _get_splits(self, X, i):
        d = X[i,:].todense()
        items = d.nonzero()[1]
        if self.shuffle:
            np.random.shuffle(items)
        splits = np.array_split(items, self.n_folds)
        counts = [a.shape[0] for a in splits]
        for c in counts:
            if c == 0:
                print(items)
                print(splits)
        return splits