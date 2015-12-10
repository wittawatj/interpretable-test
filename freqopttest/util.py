"""A module containing convenient methods for general machine learning"""

__author__ = 'wittawat'

import numpy as np

class Str(object):
    """Class containing static methods for string processing"""

    @staticmethod
    def pretty_numpy_array(precision=3):
        """Print the numpy array""" 
        pass


def meddistance(X):
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
    s = np.sum(X**2, 1)
    D = np.sqrt( s[:, np.newaxis] - 2.0*X.dot(X.T) + s[np.newaxis, :] )
    return np.median(D.flatten())


def is_real_num(x):
    """return true if x there is no error when converting x to a float"""
    try:
        float(x)
        return True
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

