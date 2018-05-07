import sys
import numpy as np
from scipy.sparse import csr_matrix

# Custom libraries
sys.path.append('../Util')
from cross_validation import ColumnwiseKFold

def main():
    n_folds = 5
    a = csr_matrix(np.random.uniform(size = (100,20)))
    kf = ColumnwiseKFold(n_folds, random_seed = 30)
    for i, (X, (user_incides, item_indices)) in enumerate(kf.split(a)):
        print(i)
        print(X)
        print(user_incides)
        print(item_indices)
        break

if __name__ == '__main__':
    main()
