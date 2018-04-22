"""
Module for testing tst module.
"""

__author__ = 'wittawat'

import autograd.numpy as np
import matplotlib.pyplot as plt
import freqopttest.data as data
import freqopttest.util as util
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import freqopttest.glo as glo
import scipy.stats as stats

import unittest

class TestUMETest(unittest.TestCase):
    """
    Test the class UMETest.
    """

    def setUp(self):
        pass

    def test_basic_H1(self):
        """
        Nothing special. Just test basic things.
        """
        seed = 12
        # sample
        n = 271
        alpha = 0.01
        for d in [1, 4]:
            # h1 is true
            ss = data.SSGaussMeanDiff(d=d, my=2.0)
            dat = ss.sample(n, seed=seed)
            xy = dat.stack_xy()
            
            sig2 = util.meddistance(xy, subsample=1000)**2
            k = kernel.KGauss(sig2)

            # Test
            for J in [1, 6]:
                # random test locations
                V = util.fit_gaussian_draw(xy, J, seed=seed+1)
                ume = tst.UMETest(V, k, n_simulate=2000, alpha=alpha)
                tresult = ume.perform_test(dat)

                # assertions
                self.assertGreaterEqual(tresult['pvalue'], 0.0)
                # H1 is true. Should reject with a small p-value
                self.assertLessEqual(tresult['pvalue'], 0.1)

    def test_optimize_locs_width(self):
        """
        Test the function optimize_locs_width(..). Make sure it does not return 
        unusual results.
        """
        # sample source 
        n = 600
        dim = 2
        seed = 17

        ss = data.SSGaussMeanDiff(dim, my=1.0)
        #ss = data.SSGaussVarDiff(dim)
        #ss = data.SSSameGauss(dim)
        # ss = data.SSBlobs()
        dim = ss.dim()

        dat = ss.sample(n, seed=seed)
        tr, te = dat.split_tr_te(tr_proportion=0.5, seed=10)
        xy_tr = tr.stack_xy()

        # initialize test_locs by drawing the a Gaussian fitted to the data
        # number of test locations
        J = 3
        V0 = util.fit_gaussian_draw(xy_tr, J, seed=seed+1)
        med = util.meddistance(xy_tr, subsample=1000)
        gwidth0 = med**2
        assert gwidth0 > 0

        # optimize
        V_opt, gw2_opt, opt_info = tst.GaussUMETest.optimize_locs_width(tr, V0, gwidth0, reg=1e-2,
            max_iter=100,  tol_fun=1e-5, disp=False, locs_bounds_frac=100,
            gwidth_lb=None, gwidth_ub=None)

        # perform the test using the optimized parameters on the test set
        alpha = 0.01
        ume_opt = tst.GaussUMETest(V_opt, gw2_opt, n_simulate=2000, alpha=alpha)
        test_result = ume_opt.perform_test(te)

        assert test_result['h0_rejected']
        assert util.is_real_num(gw2_opt)
        assert gw2_opt > 0
        assert np.all(np.logical_not((np.isnan(V_opt))))
        assert np.all(np.logical_not((np.isinf(V_opt))))


    def tearDown(self):
        pass


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

