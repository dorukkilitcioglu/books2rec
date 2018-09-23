import pickle
import sys

from hpfrec import HPF
import numpy as np

sys.path.append('Util')
from joiner import get_ratings_pickle

n_factors = 300

recommender = HPF(
    k=n_factors, a=0.3, a_prime=0.3, b_prime=1.0,
    c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
    stop_crit='train-llk', check_every=10, stop_thr=1e-3,
    users_per_batch=None, items_per_batch=None, step_size=lambda x: 1 / np.sqrt(x + 2),
    maxiter=100, reindex=False, verbose=True,
    random_seed = None, allow_inconsistent_math=False, full_llk=False,
    alloc_full_phi=False, keep_data=True, save_folder=None,
    produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
)

ratings = get_ratings_pickle('../data/goodbooks-10k/')

recommender.fit(ratings)

recommender.step_size = None

with open('recommender.pkl', 'wb') as fp:
    pickle.dump(recommender, fp)
