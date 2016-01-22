"""A module containing convenient methods for general machine learning"""

__author__ = 'wittawat'

import numpy as np


def meddistance(X, subsample=None):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array

    Return
    ------
    median distance
    """
    if subsample is None:
        s = np.sum(X**2, 1)
        D2 =  s[:, np.newaxis] - 2.0*X.dot(X.T) + s[np.newaxis, :] 
        # to prevent numerical errors from taking sqrt of negative numbers
        D2[D2 < 0] = 0
        D = np.sqrt(D2)
        return np.median(D.flatten())
    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None)


def is_real_num(x):
    """return true if x is a real number"""
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False
    

def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind

