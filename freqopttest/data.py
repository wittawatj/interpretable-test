__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import math
import matplotlib.pyplot as plt
import numpy as np
import freqopttest.util as util
import matplotlib.pyplot as plt

class SampleSource(object):
    """A data source where it is possible to resample. Subclasses may prefix 
    class names with SS"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, n, seed):
        """Return a TSTData. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """Return the dimension of the problem"""
        raise NotImplementedError()

    def visualize(self, n=400):
        """Visualize the data, assuming 2d. If not possible graphically,
        subclasses should print to the console instead."""
        data = self.sample(n, seed=1)
        x, y = data.xy()
        d = x.shape[1]

        if d==2:
            plt.plot(x[:, 0], x[:, 1], '.r', label='X')
            plt.plot(y[:, 0], y[:, 1], '.b', label='Y')
            plt.legend(loc='best')
        else:
            # not 2d. Print stats to the console.
            print(data)

class SSResample(SampleSource):
    """
    A SampleSource which subsamples without replacement from two samples independently 
    through the specified TSTData.
    """

    def __init__(self, tst_data):
        self.tst_data = tst_data

    def dim(self):
        return self.tst_data.dim()

    def sample(self, n, seed=900):
        tst_sub = self.tst_data.subsample(n, seed)
        return tst_sub

class SSNullResample(SampleSource):
    """
    A SampleSource which subsamples without replacement from only one sample.
    Randomly partition the one sample into two samples. 
    This is meant to simulate the case where H0: P=Q is true.
    """

    def __init__(self, X):
        """
        X: nxd numpy array
        """
        self.X = X

    def dim(self):
        return self.X.shape[1]

    def sample(self, n, seed=981):
        if 2*n > self.X.shape[0]:
            raise ValueError('2*n=%d exceeds the size of X = %d'%(2*n, self.X.shape[0]))
        ind = util.subsample_ind(self.X.shape[0], 2*n, seed)
        #print ind
        x = self.X[ind[:n]]
        y = self.X[ind[n:]]
        assert(x.shape[0]==y.shape[0])
        return TSTData(x, y)


class SSBlobs(SampleSource):
    """Mixture of 2d Gaussians arranged in a 2d grid. This dataset is used 
    in Chwialkovski et al., 2015 as well as Gretton et al., 2012. 
    Part of the code taken from Dino Sejdinovic and Kacper Chwialkovski's code."""

    def __init__(self, blob_distance=5, num_blobs=4, stretch=2, angle=math.pi/4.0):
        self.blob_distance = blob_distance
        self.num_blobs = num_blobs
        self.stretch = stretch
        self.angle = angle

    def dim(self):
        return 2

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        x = gen_blobs(stretch=1, angle=0, blob_distance=self.blob_distance,
                num_blobs=self.num_blobs, num_samples=n)

        y = gen_blobs(stretch=self.stretch, angle=self.angle,
                blob_distance=self.blob_distance, num_blobs=self.num_blobs,
                num_samples=n)

        np.random.set_state(rstate)
        return TSTData(x, y, label='blobs_s%d'%seed)



def gen_blobs(stretch, angle, blob_distance, num_blobs, num_samples):
    """Generate 2d blobs dataset """

    # rotation matrix
    r = np.array( [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]] )
    eigenvalues = np.diag(np.array([np.sqrt(stretch), 1]))
    mod_matix = r.dot(eigenvalues)
    mean = float(blob_distance * (num_blobs-1)) / 2
    mu = np.random.randint(0, num_blobs,(num_samples, 2))*blob_distance - mean
    return np.random.randn(num_samples,2).dot(mod_matix) + mu

class SSSameGauss(SampleSource):
    """Two same standard Gaussians for P, Q. The null hypothesis 
    H0: P=Q is true."""
    def __init__(self, d):
        """
        d: dimension of the data 
        """
        self.d = d

    def dim(self):
        return self.d

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        d = self.d
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) 
        np.random.set_state(rstate)
        return TSTData(X, Y, label='sg_d%d'%self.d)

class SSGaussMeanDiff(SampleSource):
    """Toy dataset one in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N( (my,0,0, 000), I). Only the first dimension of the means 
    differ."""
    def __init__(self, d, my=1.0):
        """
        d: dimension of the data 
        """
        self.d = d
        self.my = my

    def dim(self):
        return self.d

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        d = self.d
        mean_y = np.hstack((self.my, np.zeros(d-1) ))
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) + mean_y
        np.random.set_state(rstate)
        return TSTData(X, Y, label='gmd_d%d'%self.d)

class SSGaussVarDiff(SampleSource):
    """Toy dataset two in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N(0, diag((2, 1, 1, ...))). Only the variances of the first 
    dimension differ."""

    def __init__(self, d):
        """
        d: dimension of the data 
        """
        self.d = d

    def dim(self):
        return self.d

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        d = self.d
        std_y = np.diag(np.hstack((np.sqrt(2.0), np.ones(d-1) )))
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d).dot(std_y)
        np.random.set_state(rstate)
        return TSTData(X, Y, label='gvd')

class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        #if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        #Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = (stdx + stdy)/2.0
        return mstd
        #xy = self.stack_xy()
        #return np.mean(np.std(xy, 0)**2.0)**0.5
    
    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same 
        for both X, Y. 
        
        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = util.subsample_ind( self.X.shape[0], n, seed )
        ind_y = util.subsample_ind( self.Y.shape[0], n, seed )
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    ### end TSTData class        


def toy_2d_gauss_mean_diff(n_each, shift_2d=[0, 0], seed=1):
    """Generate a 2d toy data X, Y. 2 unit-variance Gaussians with different means."""

    rand_state = np.random.get_state()
    np.random.seed(seed)
    d = 2
    mean = [0, 00]
    X = np.random.randn(n_each, d) + mean
    Y = np.random.randn(n_each, d) + mean + shift_2d
    tst_data = TSTData(X, Y, '2D-N. shift: %s'%(str(shift_2d)) )

    np.random.set_state(rand_state)
    return tst_data

def toy_2d_gauss_variance_diff(n_each, std_factor=2, seed=1):
    """Generate a 2d toy data X, Y. 2 zero-mean Gaussians with different variances."""

    rand_state = np.random.get_state()
    np.random.seed(seed)
    d = 2
    X = np.random.randn(n_each, d)
    Y = np.random.randn(n_each, d)*std_factor
    tst_data = TSTData(X, Y, '2D-N. std_factor: %.3g'%(std_factor) )

    np.random.set_state(rand_state)
    return tst_data

def plot_2d_data(tst_data):
    """
    tst_data: an instance of TSTData
    Return a figure handle
    """
    X, Y = tst_data.xy()
    n, d = X.shape 
    if d != 2:
        raise ValueError('d must be 2 to plot.') 
    # plot
    fig = plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'xb', label='X')
    plt.plot(Y[:, 0], Y[:, 1], 'xr', label='Y')
    plt.title(tst_data.label)
    plt.legend()
    return fig

