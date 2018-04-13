"""
Module for testing tst module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import freqopttest.data as data
import freqopttest.util as util
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import freqopttest.glo as glo
import scipy.stats as stats

import unittest


class TestMETest(unittest.TestCase):
    """
    Test independent functions in tst module.
    """

    def setUp(self):
        pass

    def test_perform_test(self):
        # Full sample size
        n = 200

        # mean shift
        my = 0.1
        dim = 3
        ss = data.SSGaussMeanDiff(dim, my=my)
        # Consider two dimensions here
        for s in [2,8, 9]:
            with util.NumpySeedContext(seed=s):
                tst_data = ss.sample(n, seed=s)
                locs = np.random.randn(2, dim)
                k = kernel.KGauss(1)

                me1 = tst.METest(locs[[0],:], k, alpha=0.01)
                result1 = me1.perform_test(tst_data)
                self.assertGreaterEqual(result1['pvalue'],  0)
                self.assertGreaterEqual(result1['test_stat'],  0)

                me2 = tst.METest(locs, k, alpha=0.01)
                result2 = me2.perform_test(tst_data)
                self.assertGreaterEqual(result2['pvalue'],  0)
                self.assertGreaterEqual(result2['test_stat'],  0)

    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

