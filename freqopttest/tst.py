"""Module containing many types of two sample test algorithms"""

__author__ = "wittawat"

from abc import ABCMeta, abstractmethod
from freqopttest.data import TSTData
import matplotlib.pyplot as plt
import numpy as np
import freqopttest.util as util
import freqopttest.kernel as kernel

import scipy.stats as stats
import theano
import theano.tensor as tensor
import theano.tensor.nlinalg as nlinalg
import theano.tensor.slinalg as slinalg

class TwoSampleTest(object):
    """Abstract class for two sample tests."""
    __metaclass__ = ABCMeta

    def __init__(self, alpha=0.01):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        raise NotImplementedError()

    #@abstractmethod
    #def visual_test(self, tst_data):
    #    """Perform the test and plot the results. This is suitable for use 
    #    with IPython."""
    #    raise NotImplementedError()
    ##@abstractmethod
    #def pvalue(self):
    #   """Compute and return the p-value of the test"""
    #   raise NotImplementedError()

    #def h0_rejected(self):
    #    """Return true if the null hypothesis is rejected"""
    #    return self.pvalue() < self.alpha


class HotellingT2Test(TwoSampleTest):
    """Two-sample test with Hotelling T-squared statistic.
    Techinical details follow "Applied Multivariate Analysis" of Neil H. Timm.
    See page 156.
    
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha 

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        chi2_stat = self.compute_stat(tst_data)
        pvalue = stats.chi2.sf(chi2_stat, d)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': chi2_stat,
                'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        X, Y = tst_data.xy()
        #if X.shape[0] != Y.shape[0]:
        #    raise ValueError('Require nx = ny for now. Will improve if needed.')
        nx = X.shape[0]
        ny = Y.shape[0]
        mx = np.mean(X, 0)
        my = np.mean(Y, 0)
        mdiff = mx-my
        sx = np.cov(X.T)
        sy = np.cov(Y.T)
        s = sx/nx + sy/ny
        chi2_stat = np.linalg.solve(s, mdiff).dot(mdiff)
        return chi2_stat

class LinearMMDTest(TwoSampleTest):
    """Two-sample test with linear MMD^2 statistic. """
    
    def __init__(self, kernel, alpha=0.01):
        """
        kernel: an instance of Kernel 
        """
        self.kernel = kernel
        self.alpha = alpha 

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(X, Y, self.kernel)
        var = 2.0*snd
        pval = stats.norm.sf(stat, loc=0, scale=(var/n)**0.5)
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                'h0_rejected': pval < self.alpha}
        return results

    def compute_stat(self, tst_data):
        """Compute unbiased linear mmd estimator."""
        X, Y = tst_data.xy()
        return LinearMMDTest.linear_mmd(X, Y, self.kernel)

    @staticmethod
    def linear_mmd(X, Y, kernel):
        """Compute linear mmd estimator. O(n)"""
        lin_mmd, _ = LinearMMDTest.two_moments(X, Y, kernel)
        return lin_mmd

    @staticmethod
    def two_moments(X, Y, kernel):
        """Compute linear mmd estimator and a linear estimate of 
        the uncentred 2nd moment of h(z, z'). Total cost: O(n).

        return: (linear mmd, linear 2nd moment)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n%2 == 1:
            # make it even by removing the last row 
            X = np.delete(X, -1, axis=0)
            Y = np.delete(Y, -1, axis=0)

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]
        # linear mmd. O(n) 
        xx = kernel.pair_eval(Xodd, Xeven)
        yy = kernel.pair_eval(Yodd, Yeven)
        xo_ye = kernel.pair_eval(Xodd, Yeven)
        xe_yo = kernel.pair_eval(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        lin_mmd = np.mean(h)
        """
        Compute a linear-time estimate of the 2nd moment of h = E_z,z' h(z, z')^2.
        Note that MMD = E_z,z' h(z, z').
        This is derived by Wittawat. Did not see it proposed anywhere.
        Require O(n). Same trick as used in linear MMD to get O(n).
        """
        lin_2nd = np.mean(h**2) 
        return lin_mmd, lin_2nd


    @staticmethod
    def variance(X, Y, kernel, lin_mmd=None):
        """Compute an estimate of the variance of the linear MMD.
        Require O(n^2)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if lin_mmd is None:
            lin_mmd = LinearMMDTest.linear_mmd(X, Y, kernel)
        # compute uncentred 2nd moment of h(z, z')
        K = kernel.eval(X, X)
        L = kernel.eval(Y, Y)
        KL = kernel.eval(X, Y)
        snd_moment = np.sum( (K+L-KL-KL.T)**2 )/(n*(n-1))
        var_mmd = 2.0*(snd_moment - lin_mmd**2)
        return var_mmd

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha):
        """
        Return from the list the best kernel that maximizes the test power.
        The test power of the linear mmd is given by the CDF of a Gaussian. 

        return: (best kernel index, list of test powers)
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        powers = np.zeros(len(list_kernels))
        for ki, kernel in enumerate(list_kernels):
            lin_mmd, snd_moment = LinearMMDTest.two_moments(X, Y, kernel)
            var_lin_mmd = 2.0*(snd_moment - lin_mmd**2)
            # test threshold from N(0, var)
            thresh = stats.norm.isf(alpha, loc=0, scale=(2.0*snd_moment/n)**0.5)
            power = stats.norm.sf(thresh, loc=lin_mmd, scale=(var_lin_mmd/n)**0.5)
            powers[ki] = power
        best_ind = np.argmax(powers)
        return best_ind, powers

class GammaMMDKGaussTest(TwoSampleTest):
    """MMD test by fitting a Gamma distribution to the test statistic (MMD^2).
    This class is specific to Gaussian kernel.
    The implementation follows Arthur Gretton's Matlab code at 
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    - Has O(n^2) memory and runtime complexity
    """
    def __init__(self, gwidth2, alpha=0.01):
        """
        gwidth2: Gaussian width squared. Kernel is exp(|x-y|^2/(2*width^2))
        """
        self.alpha = alpha 
        self.gwidth2 = gwidth2

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        """

        meanMMD, varMMD, test_stat = \
            GammaMMDKGaussTest.compute_mean_variance_stat(tst_data, self.gwidth2)
        # parameters of the fitted Gamma distribution
        X, _ = tst_data.xy()
        n = X.shape[0]
        al = meanMMD**2 / varMMD
        bet = varMMD*n / meanMMD
        pval = stats.gamma.sf(test_stat, al, scale=bet)
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': test_stat,
                'h0_rejected': pval < self.alpha}
        return results


    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        raise NotImplementedError()

    def visual_test(self, tst_data):
        """Perform the test and plot the results. This is suitable for use 
        with IPython."""
        raise NotImplementedError()

    @staticmethod
    def compute_mean_variance_stat(tst_data, gwidth2):
        """Compute the mean and variance of the MMD^2, and the test statistic
        :return: (mean, variance)
        """

        X, Y = tst_data.xy()
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')

        ker = kernel.KGauss(gwidth2)
        K = ker.eval(X, X)
        L = ker.eval(Y, Y)
        KL = ker.eval(X, Y)

        n = X.shape[0]
        # computing meanMMD is only correct for Gaussian kernels.
        meanMMD = 2.0/n * (1.0 - 1.0/n*np.sum(np.diag(KL)))

        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)
        np.fill_diagonal(KL, 0.0)

        varMMD = 2.0/n/(n-1) * 1.0/n/(n-1) * np.sum((K + L - KL - KL.T)**2)
        # test statistic
        test_stat = 1.0/n * np.sum(K + L - KL - KL.T)
        return meanMMD, varMMD, test_stat

    @staticmethod
    def grid_search_gwidth2(tst_data, list_gwidth2, alpha):
        """
        Return the Gaussian width squared in the list that maximizes the test power. 
        The test power p(test_stat > alpha) is computed based on the distribution
        of the MMD^2 under H_1, which is a Gaussian.

        - list_gwidth2: a list of squared Gaussian width candidates

        :return: best width^2, list of test powers
        """
        raise NotImplementedError('Not implemented yet')
        pass
        #X, Y = tst_data.xy()
        #gwidth2_powers = np.zeros(len(list_gwidth2))
        #n = X.shape[0]
        #for i, gwidth2_i in enumerate(list_gwidth2):
        #    meanMMD, varMMD, test_stat = \
        #        GammaMMDKGaussTest.compute_mean_variance_stat(tst_data, gwidth2_i)
        #    # x_alpha = location corresponding to alpha under H0
        #    al = meanMMD**2 / varMMD
        #    bet = varMMD*n / meanMMD
        #    x_alpha = stats.gamma.ppf(1.0-alpha, al, scale=bet)

        #    # Distribution of MMD under H1 is a Gaussian 
        #    power = stats.norm.sf(x_alpha, loc=meanMMD, scale=(varMMD/n)**0.5)
        #    gwidth2_powers[i] = power
        #    print 'al: %.3g, bet: %.3g, gw2: %.2g, m_mmd: %.3g, v_mmd: %.3g'%(al, bet,
        #            gwidth2_i, meanMMD, varMMD)
        #    print 'x_alpha: %.3g'%x_alpha
        #    print ''
        #best_i = np.argmax(gwidth2_powers)
        #return list_gwidth2[best_i], gwidth2_powers


#-------------------------------------------------
class SmoothCFTest(TwoSampleTest):
    """Class for two-sample test using smooth characteristic functions.
    Use Gaussian kernel."""
    def __init__(self, test_freqs, gaussian_width, alpha=0.01):
        """
        :param test_freqs: J x d numpy array of J frequencies to test the difference
        gaussian_width: The width is used to divide the data. The test will be 
            equivalent if the data is divided beforehand and gaussian_width=1.
        """
        super(SmoothCFTest, self).__init__(alpha)
        self.test_freqs = test_freqs
        self.gaussian_width = gaussian_width

    @property
    def gaussian_width(self):
        # Gaussian width. Positive number.
        return self._gaussian_width
    
    @gaussian_width.setter
    def gaussian_width(self, width):
        if util.is_real_num(width) and float(width) > 0:
            self._gaussian_width = float(width)
        else:
            raise ValueError('gaussian_width must be a float > 0. Was %s'%(str(width)))

    def compute_stat(self, tst_data):
        # test freqs or Gaussian width undefined 
        if self.test_freqs is None: 
            raise ValueError('test_freqs must be specified.')

        X, Y = tst_data.xy()
        test_freqs = self.test_freqs
        gamma = self.gaussian_width
        s = SmoothCFTest.compute_nc_parameter(X, Y, test_freqs, gamma)
        return s

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        stat = self.compute_stat(tst_data)
        J, d = self.test_freqs.shape
        # 2J degrees of freedom because of sin and cos
        pvalue = stats.chi2.sf(stat, 2*J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha}
        return results

    #---------------------------------
    @staticmethod
    def compute_nc_parameter(X, Y, T, gwidth, reg=0.0):
        """
        Compute the non-centrality parameter of the non-central Chi-squared 
        which is the distribution of the test statistic under the H_1 (and H_0).
        The nc parameter is also the test statistic. 
        """
        if gwidth is None or gwidth <= 0:
            raise ValueError('require gaussian_width > 0. Was %s'%(str(gwidth)))

        Z = SmoothCFTest.construct_z(X, Y, T, gwidth)
        s = generic_nc_parameter(Z, reg)
        return s

    @staticmethod
    def grid_search_gwidth(tst_data, T, list_gwidth, alpha):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power, fixing the test locations ot T. 
        The test power is given by the CDF of a non-central Chi-squared 
        distribution.
        return: (best width index, list of test powers)
        """
        func_nc_param = SmoothCFTest.compute_nc_parameter
        J = T.shape[0]
        return generic_grid_search_gwidth(tst_data, T, 2*J, list_gwidth, alpha,
                func_nc_param)
            

    @staticmethod
    def create_randn(tst_data, J, alpha=0.01, seed=19):
        """Create a SmoothCFTest whose test frequencies are drawn from 
        the standard Gaussian """

        rand_state = np.random.get_state()
        np.random.seed(seed)

        gamma = tst_data.mean_std()*tst_data.dim()**0.5

        d = tst_data.dim()
        T = np.random.randn(J, d)
        np.random.set_state(rand_state)
        scf_randn = SmoothCFTest(T, gamma, alpha=alpha)
        return scf_randn

    @staticmethod 
    def construct_z(X, Y, test_freqs, gaussian_width):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.14 of Chwialkovski et al., 2015 (NIPS). 

        test_freqs: J x d test frequencies
        
        Return a n x 2J numpy array. 2J because of sin and cos for each frequency.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Sample size n must be the same for X and Y.')
        X = X/gaussian_width
        Y = Y/gaussian_width 
        n, d = X.shape
        J = test_freqs.shape[0]
        # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
        fx = np.exp(-np.sum(X**2, 1)/2)[:, np.newaxis]
        fy = np.exp(-np.sum(Y**2, 1)/2)[:, np.newaxis]
        # n x J
        x_freq = X.dot(test_freqs.T)
        y_freq = Y.dot(test_freqs.T)
        # zx: n x 2J
        zx = np.hstack((np.sin(x_freq)*fx, np.cos(x_freq)*fx))
        zy = np.hstack((np.sin(y_freq)*fy, np.cos(y_freq)*fy))
        z = zx-zy
        assert z.shape == (n, 2*J)
        return z

    @staticmethod 
    def construct_z_theano(Xth, Yth, Tth, gwidth_th):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.14 of Chwialkovski et al., 2015 (NIPS). 
        Theano version.
        
        Return a n x 2J numpy array. 2J because of sin and cos for each frequency.
        """
        Xth = Xth/gwidth_th
        Yth = Yth/gwidth_th 
        # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
        fx = tensor.exp(-(Xth**2).sum(1)/2).reshape((-1, 1))
        fy = tensor.exp(-(Yth**2).sum(1)/2).reshape((-1, 1))
        # n x J
        x_freq = Xth.dot(Tth.T)
        y_freq = Yth.dot(Tth.T)
        # zx: n x 2J
        zx = tensor.concatenate([tensor.sin(x_freq)*fx, tensor.cos(x_freq)*fx], axis=1)
        zy = tensor.concatenate([tensor.sin(y_freq)*fy, tensor.cos(y_freq)*fy], axis=1)
        z = zx-zy
        return z

    @staticmethod
    def optimize_freqs_width(tst_data, alpha, n_test_freqs=10, max_iter=400,
            freqs_step_size=0.2, gwidth_step_size=0.01, batch_proportion=1.0,
            tol_fun=1e-3, seed=1):
        """Optimize the test frequencies and the Gaussian kernel width by 
        maximizing the test power. X, Y should not be the same data as used 
        in the actual test (i.e., should be a held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (test_freqs, gaussian_width, info)
        """
        J = n_test_freqs
        """
        Optimize the empirical version of Lambda(T) i.e., the criterion used 
        to optimize the test locations, for the test based 
        on difference of mean embeddings with Gaussian kernel. 
        Also optimize the Gaussian width.

        :return a theano function T |-> Lambda(T)
        """
        d = tst_data.dim()
        # set the seed
        rand_state = np.random.get_state()
        np.random.seed(seed)

        # draw frequencies randomly from the standard Gaussian. 
        # TODO: Can we do better?
        T0 = np.random.randn(J, d)
        # reset the seed back to the original
        np.random.set_state(rand_state)

        # grid search to determine the initial gwidth
        mean_sd = tst_data.mean_std()
        scales = 2.0**np.linspace(-4, 4, 20)
        list_gwidth = np.hstack( (mean_sd*scales*(d**0.5), 2**np.linspace(-20, 10, 20) ))
        list_gwidth.sort()
        besti, powers = SmoothCFTest.grid_search_gwidth(tst_data, T0,
                list_gwidth, alpha)
        # initialize with the best width from the grid search
        gwidth0 = list_gwidth[besti]
        assert util.is_real_num(gwidth0), 'gwidth0 not real. Was %s'%str(gwidth0)
        assert gwidth0 > 0, 'gwidth0 not positive. Was %.3g'%gwidth0

        func_z = SmoothCFTest.construct_z_theano
        # info = optimization info 
        T, gamma, info = optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, 
                max_iter=max_iter, T_step_size=freqs_step_size, 
                gwidth_step_size=gwidth_step_size, batch_proportion=batch_proportion,
                tol_fun=tol_fun)
        assert util.is_real_num(gamma), 'gamma is not real. Was %s' % str(gamma)

        ninfo = {'test_freqs': info['Ts'], 'test_freqs0': info['T0'], 
                'gwidths': info['gwidths'], 'obj_values': info['obj_values'],
                'gwidth0': gwidth0, 'gwidth0_powers': powers}
        return (T, gamma, ninfo  )

    @staticmethod
    def optimize_gwidth(tst_data, T, gwidth0, max_iter=400, 
            gwidth_step_size=0.1, batch_proportion=1.0, tol_fun=1e-3):
        """Optimize the Gaussian kernel width by 
        maximizing the test power, fixing the test frequencies to T. X, Y should
        not be the same data as used in the actual test (i.e., should be a
        held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (gaussian_width, info)
        """

        func_z = SmoothCFTest.construct_z_theano
        # info = optimization info 
        gamma, info = optimize_gaussian_width(tst_data, T, gwidth0, func_z, 
                max_iter=max_iter, gwidth_step_size=gwidth_step_size,
                batch_proportion=batch_proportion, tol_fun=tol_fun)

        ninfo = {'test_freqs': T, 'gwidths': info['gwidths'], 'obj_values':
                info['obj_values']}
        return ( gamma, ninfo  )

#-------------------------------------------------
class MeanEmbeddingTest(TwoSampleTest):
    """Class for two-sample test using squared difference of mean embeddings. 
    Use Gaussian kernel."""

    def __init__(self, test_locs, gaussian_width, alpha=0.01):
        """
        :param test_locs: J x d numpy array of J locations to test the difference
        gaussian_width: The width is used to divide the data. The test will be 
            equivalent if the data is divided beforehand and gaussian_width=1.
        """
        super(MeanEmbeddingTest, self).__init__(alpha)

        self.test_locs = test_locs
        self.gaussian_width = gaussian_width

    @property
    def gaussian_width(self):
        # Gaussian width. Positive number.
        return self._gaussian_width
    
    @gaussian_width.setter
    def gaussian_width(self, width):
        if util.is_real_num(width) and float(width) > 0:
            self._gaussian_width = float(width)
        else:
            raise ValueError('gaussian_width must be a float > 0. Was %s'%(str(width)))

    def perform_test(self, tst_data):
        stat = self.compute_stat(tst_data)
        J, d = self.test_locs.shape
        pvalue = stats.chi2.sf(stat, J)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, tst_data):
        # test locations or Gaussian width undefined 
        if self.test_locs is None: 
            raise ValueError('test_locs must be specified.')

        X, Y = tst_data.xy()
        test_locs = self.test_locs
        gamma = self.gaussian_width
        stat = MeanEmbeddingTest.compute_nc_parameter(X, Y, test_locs, gamma)
        return stat

    def visual_test(self, tst_data):
        results = self.perform_test(tst_data)
        s = results['test_stat']
        pval = results['pvalue']
        J = self.test_locs.shape[0]
        domain = np.linspace(stats.chi2.ppf(0.001, J), stats.chi2.ppf(0.9999, J), 200)
        plt.plot(domain, stats.chi2.pdf(domain, J), label='$\chi^2$ (df=%d)'%J)
        plt.stem([s], [stats.chi2.pdf(J, J)/2], 'or-', label='test stat')
        plt.legend(loc='best', frameon=True)
        plt.title('%s. p-val: %.3g. stat: %.3g'%(type(self).__name__, pval, s))
        plt.show()

    #===============================
    @staticmethod
    def compute_nc_parameter(X, Y, T, gwidth, reg=0.0):
        """
        Compute the non-centrality parameter of the non-central Chi-squared 
        which is the distribution of the test statistic under the H_1 (and H_0).
        The nc parameter is also the test statistic. 
        """
        if gwidth is None or gwidth <= 0:
            raise ValueError('require gaussian_width > 0. Was %s.'%(str(gwidth)))
        n = X.shape[0]
        #g = MeanEmbeddingTest.asym_gauss_kernel(X, T, gwidth)
        #h = MeanEmbeddingTest.asym_gauss_kernel(Y, T, gwidth)
        g = MeanEmbeddingTest.gauss_kernel(X, T, gwidth)
        h = MeanEmbeddingTest.gauss_kernel(Y, T, gwidth)
        Z = g-h
        s = generic_nc_parameter(Z, reg)
        return s


    @staticmethod 
    def construct_z_theano(Xth, Yth, T, gaussian_width):
        """Construct the features Z to be used for testing with T^2 statistics.
        Z is defined in Eq.12 of Chwialkovski et al., 2015 (NIPS). 

        T: J x d test locations
        
        Return a n x J numpy array. 
        """
        g = MeanEmbeddingTest.gauss_kernel_theano(Xth, T, gaussian_width)
        h = MeanEmbeddingTest.gauss_kernel_theano(Yth, T, gaussian_width)
        # Z: nx x J
        Z = g-h
        return Z


    #@staticmethod
    #def asym_gauss_kernel(X, test_locs, gamma):
    #    """Compute a X.shape[0] x test_locs.shape[0] Gaussian kernel matrix where
    #    the Gaussian width gamma will divide only the data X (not test_locs).
    #    This is defined as in Chwialkovski, 2015 (NIPS)
    #    """
    #    n, d = X.shape
    #    X = X/gamma
    #    D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*X.dot(test_locs.T) + np.sum(test_locs**2, 1)
    #    K = np.exp(-D2)
    #    raise ValueError('deprecated.')
    #    return K

    #@staticmethod
    #def asym_gauss_kernel_theano(X, test_locs, gamma):
    #    """Asymmetric kernel for the two sample test. Theano version.
    #    :return kernel matrix X.shape[0] x test_locs.shape[0]
    #    """
    #    T = test_locs
    #    n, d = X.shape
    #    X = X/gamma

    #    D2 = (X**2).sum(1).reshape((-1, 1)) - 2*X.dot(T.T) + tensor.sum(T**2, 1).reshape((1, -1))
    #    K = tensor.exp(-D2)
    #    raise ValueError('deprecated.')
    #    return K

    @staticmethod
    def gauss_kernel(X, test_locs, gwidth2):
        """Compute a X.shape[0] x test_locs.shape[0] Gaussian kernel matrix 
        """
        n, d = X.shape
        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*X.dot(test_locs.T) + np.sum(test_locs**2, 1)
        K = np.exp(-D2/(2.0*gwidth2))
        return K

    @staticmethod
    def gauss_kernel_theano(X, test_locs, gwidth2):
        """Gaussian kernel for the two sample test. Theano version.
        :return kernel matrix X.shape[0] x test_locs.shape[0]
        """
        T = test_locs
        n, d = X.shape

        D2 = (X**2).sum(1).reshape((-1, 1)) - 2*X.dot(T.T) + tensor.sum(T**2, 1).reshape((1, -1))
        K = tensor.exp(-D2/(2.0*gwidth2))
        return K

    @staticmethod
    def create_fit_gauss_heuristic(tst_data, n_test_locs, alpha=0.01, seed=1):
        """Construct a MeanEmbeddingTest where test_locs are drawn from  Gaussians
        fitted to the data x, y.          
        """
        #if cov_xy.ndim == 0:
        #    # 1d dataset. 
        #    cov_xy = np.array([[cov_xy]])
        X, Y = tst_data.xy()
        T = MeanEmbeddingTest.init_locs_2randn(tst_data, n_test_locs, seed)

        # Gaussian (asymmetric) kernel width is set to the average standard
        # deviations of x, y
        #gamma = tst_data.mean_std()*(tst_data.dim()**0.5)
        gwidth2 = util.meddistance(tst_data.stack_xy(), 1000)
        
        met = MeanEmbeddingTest(test_locs=T, gaussian_width=gwidth2, alpha=alpha)
        return met

    @staticmethod
    def optimize_locs_width(tst_data, alpha, n_test_locs=10, max_iter=400, 
            locs_step_size=0.1, gwidth_step_size=0.01, batch_proportion=1.0, 
            tol_fun=1e-3, seed=1):
        """Optimize the test locations and the Gaussian kernel width by 
        maximizing the test power. X, Y should not be the same data as used 
        in the actual test (i.e., should be a held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (test_locs, gaussian_width, info)
        """
        J = n_test_locs
        """
        Optimize the empirical version of Lambda(T) i.e., the criterion used 
        to optimize the test locations, for the test based 
        on difference of mean embeddings with Gaussian kernel. 
        Also optimize the Gaussian width.

        :return a theano function T |-> Lambda(T)
        """

        T0 = MeanEmbeddingTest.init_locs_2randn(tst_data, n_test_locs, seed=seed)
        func_z = MeanEmbeddingTest.construct_z_theano
        # Use grid search to initialize the gwidth
        med = util.meddistance(tst_data.stack_xy(), 1000)
        list_gwidth2 = np.hstack( ( (med**2) *(2.0**np.linspace(-5, 5, 40) ) ) )
        list_gwidth2.sort()
        besti, powers = MeanEmbeddingTest.grid_search_gwidth(tst_data, T0,
                list_gwidth2, alpha)
        gwidth0 = list_gwidth2[besti]
        assert util.is_real_num(gwidth0), 'gwidth0 not real. Was %s'%str(gwidth0)
        assert gwidth0 > 0, 'gwidth0 not positive. Was %.3g'%gwidth0

        # info = optimization info 
        T, gamma, info = optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, 
                max_iter=max_iter, T_step_size=locs_step_size, 
                gwidth_step_size=gwidth_step_size, batch_proportion=batch_proportion,
                tol_fun=tol_fun)
        assert util.is_real_num(gamma), 'gamma is not real. Was %s' % str(gamma)

        ninfo = {'test_locs': info['Ts'], 'test_locs0': info['T0'], 
                'gwidths': info['gwidths'], 'obj_values': info['obj_values'],
                'gwidth0': gwidth0, 'gwidth0_powers': powers}
        return (T, gamma, ninfo  )

    @staticmethod 
    def init_locs_randn(tst_data, n_test_locs, seed=1):
        """Fit a Gaussian to the merged data of the two samples and draw 
        n_test_locs points from the Gaussian"""
        # set the seed
        rand_state = np.random.get_state()
        np.random.seed(seed)

        X, Y = tst_data.xy()
        # fit a Gaussian in the middle of X, Y and draw sample to initialize T
        xy = np.vstack((X, Y))
        mean_xy = np.mean(xy, 0)
        cov_xy = np.cov(xy.T)
        T0 = np.random.multivariate_normal(mean_xy, cov_xy, n_test_locs)
        # reset the seed back to the original
        np.random.set_state(rand_state)
        return T0

    @staticmethod 
    def init_locs_2randn(tst_data, n_test_locs, seed=1):
        """Fit a Gaussian to each dataset and draw half of n_test_locs from 
        each """
        if n_test_locs == 1:
            return MeanEmbeddingTest.init_locs_randn(tst_data, n_test_locs, seed)

        # set the seed 
        rand_state = np.random.get_state()
        np.random.seed(seed)

        X, Y = tst_data.xy()
        # fit a Gaussian to each of X, Y
        mean_x = np.mean(X, 0)
        mean_y = np.mean(Y, 0)
        cov_x = np.cov(X.T)
        [Dx, Vx] = np.linalg.eig(cov_x)
        # shrink the covariance so that the drawn samples will not be so 
        # far away from the data
        eig_pow = 0.9 # 1.0 = not shrink
        reduced_cov_x = Vx.dot(np.diag(Dx**eig_pow)).dot(Vx.T)
        cov_y = np.cov(Y.T)
        [Dy, Vy] = np.linalg.eig(cov_y)
        reduced_cov_y = Vy.dot(np.diag(Dy**eig_pow).dot(Vy.T))
        # integer division
        Jx = n_test_locs/2
        Jy = n_test_locs - Jx
        assert Jx+Jy==n_test_locs, 'total test locations is not n_test_locs'
        Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
        Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
        T0 = np.vstack((Tx, Ty))

        # reset the seed back 
        np.random.set_state(rand_state)
        return T0

    @staticmethod
    def grid_search_gwidth(tst_data, T, list_gwidth, alpha):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power, fixing the test locations ot T. 
        The test power is given by the CDF of a non-central Chi-squared 
        distribution.
        return: (best width index, list of test powers)
        """
        func_nc_param = MeanEmbeddingTest.compute_nc_parameter
        J = T.shape[0]
        return generic_grid_search_gwidth(tst_data, T, J, list_gwidth, alpha,
                func_nc_param)
            

    @staticmethod
    def optimize_gwidth(tst_data, T, gwidth0, max_iter=400, 
            gwidth_step_size=0.1, batch_proportion=1.0, tol_fun=1e-3):
        """Optimize the Gaussian kernel width by 
        maximizing the test power, fixing the test locations to T. X, Y should
        not be the same data as used in the actual test (i.e., should be a
        held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        
        Return (gaussian_width, info)
        """

        func_z = MeanEmbeddingTest.construct_z_theano
        # info = optimization info 
        gamma, info = optimize_gaussian_width(tst_data, T, gwidth0, func_z, 
                max_iter=max_iter, gwidth_step_size=gwidth_step_size,
                batch_proportion=batch_proportion, tol_fun=tol_fun)

        ninfo = {'test_locs': T, 'gwidths': info['gwidths'], 'obj_values':
                info['obj_values']}
        return ( gamma, ninfo  )


    @staticmethod
    def heu_3gauss_test_locs(J):
        """Draw J test locations with a heuristic.
        The heuristic is to fit a Gaussian to each sample, and another Gaussian 
        in the center of the two. Then, draw test locations from the three Gaussians.
        """
        pass


# ///////////// global functions ///////////////

def generic_nc_parameter(Z, reg=0.0):
    """
    Compute the non-centrality parameter of the non-central Chi-squared 
    which is the distribution of the test statistic under the H_1 (and H_0).
    The nc parameter is also the test statistic. 
    """
    n = Z.shape[0]
    Sig = np.cov(Z.T)
    W = np.mean(Z, 0)
    n_features = len(W)
    # test statistic
    try:
        s = n*np.linalg.solve(Sig + reg*np.eye(Sig.shape[0]), W).dot(W)
    except np.linalg.LinAlgError:
        print('LinAlgError. Return -1 as the nc_parameter.')
        s = -1 
    return s

def generic_grid_search_gwidth(tst_data, T, df, list_gwidth, alpha, func_nc_param):
    """
    Linear search for the best Gaussian width in the list that maximizes 
    the test power, fixing the test locations ot T. 
    The test power is given by the CDF of a non-central Chi-squared 
    distribution.
    return: (best width index, list of test powers)
    """
    # number of test locations
    X, Y = tst_data.xy()
    powers = np.zeros(len(list_gwidth))
    lambs = np.zeros(len(list_gwidth))
    thresh = stats.chi2.isf(alpha, df=df)
    print('thresh: %.3g'% thresh)
    for wi, gwidth in enumerate(list_gwidth):
        # non-centrality parameter
        try:
            #import pdb; pdb.set_trace()
            lamb = func_nc_param(X, Y, T, gwidth, reg=0)
            if lamb <= 0:
                # This can happen when Z, Sig are ill-conditioned. 
                #print('negative lamb: %.3g'%lamb)
                raise np.linalg.LinAlgError
            power = stats.ncx2.sf(thresh, df=df, nc=lamb)
            powers[wi] = power
            lambs[wi] = lamb
            print('i: %2d, lamb: %5.3g, gwidth: %5.3g, power: %.4f'
                   %(wi, lamb, gwidth, power))
        except np.linalg.LinAlgError:
            # probably matrix inverse failed. 
            print('LinAlgError. skip width (%d, %.3g)'%(wi, gwidth))
            powers[wi] = np.NINF
            lambs[wi] = np.NINF
    # to prevent the gain of test power from numerical instability, 
    # consider upto 3 decimal places. Widths that come early in the list 
    # are preferred if test powers are equal.
    besti = np.argmax(np.around(powers, 3))
    return besti, powers


# Used by SmoothCFTest and MeanEmbeddingTest
def optimize_gaussian_width(tst_data, T, gwidth0, func_z, max_iter=400, 
        gwidth_step_size=0.1, batch_proportion=1.0, 
        tol_fun=1e-3 ):
    """Optimize the Gaussian kernel width by gradient ascent 
    by maximizing the test power.
    This does the same thing as optimize_T_gaussian_width() without optimizing 
    T (T = test locations / test frequencies).

    Return (optimized Gaussian width, info)
    """

    X, Y = tst_data.xy()
    nx, d = X.shape
    # initialize Theano variables
    Tth = theano.shared(T, name='T')
    Xth = tensor.dmatrix('X')
    Yth = tensor.dmatrix('Y')
    it = theano.shared(1, name='iter')
    # square root of the Gaussian width. Use square root to handle the 
    # positivity constraint by squaring it later.
    gamma_sq_init = gwidth0**0.5
    gamma_sq_th = theano.shared(gamma_sq_init, name='gamma')

    #sqr(x) = x^2
    Z = func_z(Xth, Yth, Tth, tensor.sqr(gamma_sq_th))
    W = Z.sum(axis=0)/nx
    # covariance 
    Z0 = Z - W
    Sig = Z0.T.dot(Z0)/nx

    # gradient computation does not support solve()
    #s = slinalg.solve(Sig, W).dot(nx*W)
    s = nlinalg.matrix_inverse(Sig).dot(W).dot(W)*nx
    gra_gamma_sq = tensor.grad(s, gamma_sq_th)
    step_pow = 0.5
    max_gam_sq_step = 1.0
    func = theano.function(inputs=[Xth, Yth], outputs=s, 
           updates=[
              (it, it+1), 
              #(gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq\
              #        /it**step_pow/tensor.sum(gra_gamma_sq**2)**0.5 ) 
              (gamma_sq_th, gamma_sq_th+gwidth_step_size*tensor.sgn(gra_gamma_sq) \
                      *tensor.minimum(tensor.abs_(gra_gamma_sq), max_gam_sq_step) \
                      /it**step_pow) 
              ] 
           )
    # //////// run gradient ascent //////////////
    S = np.zeros(max_iter)
    gams = np.zeros(max_iter)
    for t in range(max_iter):
        # stochastic gradient ascent
        ind = np.random.choice(nx, min(int(batch_proportion*nx), nx), replace=False)
        # record objective values 
        S[t] = func(X[ind, :], Y[ind, :])
        gams[t] = gamma_sq_th.get_value()**2

        # check the change of the objective values 
        if t >= 2 and abs(S[t]-S[t-1]) <= tol_fun:
            break

    S = S[:t]
    gams = gams[:t]

    # optimization info 
    info = {'T': T, 'gwidths': gams, 'obj_values': S}
    return (gams[-1], info  )




# Used by SmoothCFTest and MeanEmbeddingTest
def optimize_T_gaussian_width(tst_data, T0, gwidth0, func_z, max_iter=400, 
        T_step_size=0.05, gwidth_step_size=0.01, batch_proportion=1.0, 
        tol_fun=1e-3 ):
    """Optimize the T (test locations for MeanEmbeddingTest, frequencies for 
    SmoothCFTest) and the Gaussian kernel width by 
    maximizing the test power. X, Y should not be the same data as used 
    in the actual test (i.e., should be a held-out set). 
    Optimize the empirical version of Lambda(T) i.e., the criterion used 
    to optimize the test locations.

    - T0: Jxd numpy array. initial value of T,  where
      J = the number of test locations/frequencies
    - gwidth0: initial Gaussian width (width squared for the MeanEmbeddingTest)
    - func_z: function that works on Theano variables 
        to construct features to be used for the T^2 test. 
        (X, Y, T, gaussian_width) |-> n x J'
    - max_iter: #gradient descent iterations
    - batch_proportion: (0,1] value to be multipled with nx giving the batch 
        size in stochastic gradient. 1 = full gradient ascent.
    - tol_fun: termination tolerance of the objective value
    
    Return (test_locs, gaussian_width, info)
    """

    print 'T0: '
    print(T0)
    X, Y = tst_data.xy()
    nx, d = X.shape
    # initialize Theano variables
    T = theano.shared(T0, name='T')
    Xth = tensor.dmatrix('X')
    Yth = tensor.dmatrix('Y')
    it = theano.shared(1, name='iter')
    # square root of the Gaussian width. Use square root to handle the 
    # positivity constraint by squaring it later.
    gamma_sq_init = gwidth0**0.5
    gamma_sq_th = theano.shared(gamma_sq_init, name='gamma')

    #sqr(x) = x^2
    Z = func_z(Xth, Yth, T, tensor.sqr(gamma_sq_th))
    W = Z.sum(axis=0)/nx
    # covariance 
    Z0 = Z - W
    Sig = Z0.T.dot(Z0)/nx

    # gradient computation does not support solve()
    #s = slinalg.solve(Sig, W).dot(nx*W)
    s = nlinalg.matrix_inverse(Sig).dot(W).dot(W)*nx
    gra_T, gra_gamma_sq = tensor.grad(s, [T, gamma_sq_th])
    step_pow = 0.5
    max_gam_sq_step = 1.0
    func = theano.function(inputs=[Xth, Yth], outputs=s, 
           updates=[
              (T, T+T_step_size*gra_T/it**step_pow/tensor.sum(gra_T**2)**0.5 ), 
              (it, it+1), 
              #(gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq\
              #        /it**step_pow/tensor.sum(gra_gamma_sq**2)**0.5 ) 
              (gamma_sq_th, gamma_sq_th+gwidth_step_size*tensor.sgn(gra_gamma_sq) \
                      *tensor.minimum(tensor.abs_(gra_gamma_sq), max_gam_sq_step) \
                      /it**step_pow) 
              ] 
           )
           #updates=[(T, T+T_step_size*gra_T), (it, it+1), 
           #    (gamma_sq_th, gamma_sq_th+gwidth_step_size*gra_gamma_sq) ] )
                           #updates=[(T, T+0.1*gra_T), (it, it+1) ] )

    # //////// run gradient ascent //////////////
    S = np.zeros(max_iter)
    J = T0.shape[0]
    Ts = np.zeros((max_iter, J, d))
    gams = np.zeros(max_iter)
    for t in range(max_iter):
        # stochastic gradient ascent
        ind = np.random.choice(nx, min(int(batch_proportion*nx), nx), replace=False)
        # record objective values 
        try:
            S[t] = func(X[ind, :], Y[ind, :])
        except: 
            print('Exception occurred during gradient descent. Stop optimization.')
            print('Return the value from previous iter. ')
            import traceback as tb 
            tb.print_exc()
            t = t -1
            break

        Ts[t] = T.get_value()
        gams[t] = gamma_sq_th.get_value()**2

        # check the change of the objective values 
        if t >= 2 and abs(S[t]-S[t-1]) <= tol_fun:
            break

    S = S[:t+1]
    Ts = Ts[:t+1]
    gams = gams[:t+1]

    # optimization info 
    info = {'Ts': Ts, 'T0':T0, 'gwidths': gams, 'obj_values': S, 'gwidth0':
            gwidth0}

    if t >= 0:
        opt_T = Ts[-1]
        # for some reason, optimization can give a non-numerical result
        opt_gwidth = gams[-1] if util.is_real_num(gams[-1]) else gwidth0
    else:
        # Probably an error occurred in the first iter.
        opt_T = T0
        opt_gwidth = gwidth0
    return (opt_T, opt_gwidth, info  )


