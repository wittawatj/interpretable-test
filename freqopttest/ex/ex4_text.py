"""
Experiment on real text data.
"""
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
import freqopttest.data as data
import freqopttest.tst as tst
import freqopttest.glo as glo
import freqopttest.util as util 
import freqopttest.kernel as kernel 
from . import exglobal
try:
   import pickle as pickle 
except:
   import pickle

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and freqopttest have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import math
import autograd.numpy as np
import os
import sys 

def job_met_opt(sample_source, tr, te, r):
    """MeanEmbeddingTest with test locations optimzied."""
    # MeanEmbeddingTest. optimize the test locations
    with util.ContextTimer() as t:
        met_opt_options = {'n_test_locs': J, 'max_iter': 200, 
                'locs_step_size': 500.0, 'gwidth_step_size': 0.2, 'seed': r+92856,
                'tol_fun': 1e-4}
        test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        met_opt_test  = met_opt.perform_test(te)

    result = {'test_method': met_opt, 'test_result': met_opt_test, 'time_secs': t.secs}
    return result

def job_met_gwgrid(sample_source, tr, te, r):
    """MeanEmbeddingTest. Optimize only the Gaussian width with grid search
    Fix the test locations."""

    with util.ContextTimer() as t: 
        # optimize on the training set
        T_randn = tst.MeanEmbeddingTest.init_locs_2randn(tr, J, seed=r+92856)
        med = util.meddistance(tr.stack_xy(), 1000)
        list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-5, 5, 40) ) ) )
        list_gwidth.sort()
        besti, powers = tst.MeanEmbeddingTest.grid_search_gwidth(tr, T_randn,
                list_gwidth, alpha)

        best_width2 = list_gwidth[besti]
        met_grid = tst.MeanEmbeddingTest(T_randn, best_width2, alpha)
        test_result = met_grid.perform_test(te)
    result = {'test_method': met_grid, 'test_result': test_result, 'time_secs': t.secs}
    return result

def job_scf_opt(sample_source, tr, te, r):
    """SmoothCFTest with frequencies optimized."""
    with util.ContextTimer() as t:
        op = {'n_test_freqs': J, 'max_iter': 500, 'freqs_step_size': 1.0, 
                'gwidth_step_size': 0.1, 'seed': r+92856, 'tol_fun': 1e-3}
        test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
        scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
        scf_opt_test = scf_opt.perform_test(te)
    
    result = {'test_method': scf_opt, 'test_result': scf_opt_test, 'time_secs': t.secs}
    return result

def job_scf_gwgrid(sample_source, tr, te, r):

    rand_state = np.random.get_state()
    np.random.seed(r+92856)

    with util.ContextTimer() as t:
        d = tr.dim()
        T_randn = np.random.randn(J, d)
        np.random.set_state(rand_state)

        # grid search to determine the initial gwidth
        mean_sd = tr.mean_std()
        scales = 2.0**np.linspace(-4, 4, 20)
        list_gwidth = np.hstack( (mean_sd*scales*(d**0.5), 2**np.linspace(-8, 8, 20) ))
        list_gwidth.sort()
        besti, powers = tst.SmoothCFTest.grid_search_gwidth(tr, T_randn,
                list_gwidth, alpha)
        # initialize with the best width from the grid search
        best_width = list_gwidth[besti]
        scf_gwgrid = tst.SmoothCFTest(T_randn, best_width, alpha)
        test_result = scf_gwgrid.perform_test(te)
    result = {'test_method': scf_gwgrid, 'test_result': test_result, 'time_secs': t.secs}
    return result

def job_quad_mmd(sample_source, tr, te, r):
    """Quadratic mmd with grid search to choose the best Gaussian width.
    One-sample U-statistic. This should NOT be used anymore."""
    # If n is too large, pairwise meddian computation can cause a memory error. 
            
    with util.ContextTimer() as t:
        med = util.meddistance(tr.stack_xy(), 1000)
        list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 40) ) ) )
        list_gwidth.sort()
        list_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

        # grid search to choose the best Gaussian width
        besti, powers = tst.QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
        # perform test 
        best_ker = list_kernels[besti]
        mmd_test = tst.QuadMMDTest(best_ker, n_permute=1000, alpha=alpha, 
                use_1sample_U=True)
        test_result = mmd_test.perform_test(te)
    result = {'test_method': mmd_test, 'test_result': test_result, 'time_secs': t.secs}
    return result


def job_quad_mmd_2U(sample_source, tr, te, r):
    """Quadratic mmd with grid search to choose the best Gaussian width.
    Use two-sample U statistics to compute k(X,Y).
    """
    # If n is too large, pairwise meddian computation can cause a memory error. 
            
    with util.ContextTimer() as t:
        med = util.meddistance(tr.stack_xy(), 1000)
        list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 40) ) ) )
        list_gwidth.sort()
        list_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

        # grid search to choose the best Gaussian width
        besti, powers = tst.QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
        # perform test 
        best_ker = list_kernels[besti]
        mmd_test = tst.QuadMMDTest(best_ker, n_permute=1000, alpha=alpha,
                use_1sample_U=False)
        test_result = mmd_test.perform_test(te)
    result = {'test_method': mmd_test, 'test_result': test_result, 'time_secs': t.secs}
    return result

def job_lin_mmd(sample_source, tr, te, r):
    """Linear mmd with grid search to choose the best Gaussian width."""
    # should be completely deterministic

    # If n is too large, pairwise meddian computation can cause a memory error. 
    with util.ContextTimer() as t:
        X, Y = tr.xy()
        Xr = X[:min(X.shape[0], 1000), :]
        Yr = Y[:min(Y.shape[0], 1000), :]
        
        med = util.meddistance(np.vstack((Xr, Yr)) )
        widths = [ (med*f) for f in 2.0**np.linspace(-1, 4, 40)]
        list_kernels = [kernel.KGauss( w**2 ) for w in widths]
        # grid search to choose the best Gaussian width
        besti, powers = tst.LinearMMDTest.grid_search_kernel(tr, list_kernels, alpha)
        # perform test 
        best_ker = list_kernels[besti]
        lin_mmd_test = tst.LinearMMDTest(best_ker, alpha)
        test_result = lin_mmd_test.perform_test(te)

    result = {'test_method': lin_mmd_test, 'test_result': test_result, 'time_secs': t.secs}
    return result

def job_hotelling(sample_source, tr, te, r):
    """Hotelling T-squared test"""
    # Since text data are high-d, T-test will likely cause a LinAlgError because 
    # of the singular covariance matrix.
    with util.ContextTimer() as t:
        htest = tst.HotellingT2Test(alpha=alpha)
        try:
            test_result = htest.perform_test(te)
        except np.linalg.linalg.LinAlgError:
            test_result = {'alpha': alpha, 'pvalue': 1.0, 'test_stat': 1e-5,
                    'h0_rejected':  False}
    result = {'test_method': htest, 'test_result': test_result, 'time_secs': t.secs}
    return result

# Define our custom Job, which inherits from base class IndependentJob
class Ex4Job(IndependentJob):
   
    def __init__(self, aggregator, prob_label, rep, n, job_func):
        walltime = 60*59*24 
        memory = int(tr_proportion*n*1e-1) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        self.prob_label = prob_label
        self.rep = rep
        self.n = n
        self.job_func = job_func

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        
        r = self.rep
        sample_source, nmax = get_sample_source(self.prob_label)
        d = sample_source.dim()
        job_func = self.job_func
        logger.info("computing. %s. r=%d "%(job_func.__name__, r ))

        tst_data = sample_source.sample(self.n, seed=r)
        tr, te = tst_data.split_tr_te(tr_proportion=tr_proportion, seed=r+20 )
        prob_label = self.prob_label
        job_result = job_func(sample_source, tr, te, r)

        # create ScalarResult instance
        result = SingleResult(job_result)
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done. ex4: %s, r=%d "%(job_func.__name__, r))

        # save result
        func_name = job_func.__name__
        fname = '%s-%s-J%d_r%d_d%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, J, r, d, alpha, tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from freqopttest.ex.ex4_text import job_met_opt
from freqopttest.ex.ex4_text import job_met_gwgrid
from freqopttest.ex.ex4_text import job_scf_opt
from freqopttest.ex.ex4_text import job_scf_gwgrid
from freqopttest.ex.ex4_text import job_quad_mmd
from freqopttest.ex.ex4_text import job_quad_mmd_2U
from freqopttest.ex.ex4_text import job_lin_mmd
from freqopttest.ex.ex4_text import job_hotelling
from freqopttest.ex.ex4_text import Ex4Job

#--- experimental setting -----
ex = 4

# number of test locations / test frequencies J
J = 1
alpha = 0.01
tr_proportion = 0.5
# repetitions 
reps = 500
#method_job_funcs = [ job_met_opt, job_scf_opt, job_lin_mmd, job_hotelling]
#method_job_funcs = [ job_met_opt, job_scf_opt, job_quad_mmd_2U,
#        job_lin_mmd]
method_job_funcs = [ job_met_opt, job_met_gwgrid, job_scf_opt, job_scf_gwgrid,
       job_quad_mmd_2U, job_lin_mmd]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (ni, r) already exists.
is_rerun = False
#---------------------------

label2fname = {'bayes_neuro_d2000_rnoun':'bayes_neuro_np794_nq788_d2000_random_noun.p',
        'bayes_learning_d2000_rnoun': 'bayes_learning_np821_nq276_d2000.p',
        'bayes_bayes_d2000_rnoun': 'bayes_bayes_np430_nq432_d2000.p',
        #'bayes_neuro_d800_rverb': 'bayes_neuro_np794_nq788_d800_random_verb.p',
        #'bayes_neuro_d300_rnoun': 'bayes_neuro_np794_nq788_d300_random_noun.p',
        #'deep_learning_d1000_rnoun': 'deep_learning_np427_nq339_d1000_random_noun.p',
        'deep_learning_d2000_rnoun': 'deep_learning_np431_nq299_d2000_random_noun.p',
        'bayes_deep_d2000_rnoun': 'bayes_deep_np846_nq433_d2000_random_noun.p',
        #'deep_neuro_d2000_rnoun': 'deep_neuro_np105_nq512_d2000.p',
        'neuro_learning_d2000_rnoun': 'neuro_learning_np832_nq293_d2000.p',
        }

cache_loaded = {}

def load_nips_TSTData(fname):
    if fname in cache_loaded:
        return cache_loaded[fname]

    fpath = glo.data_file(fname)
    with open(fpath, 'r') as f:
        loaded = pickle.load(f)

    X = loaded['P']
    Y = loaded['Q']
    n_min = min(X.shape[0], Y.shape[0])
    X = X[:n_min, :]
    Y = Y[:n_min, :]
    assert(X.shape[0] == Y.shape[0])
    tst_data = data.TSTData(X, Y) 
    cache_loaded[fname] = (tst_data, n_min)
    return tst_data,  n_min

def get_sample_source(prob_label):
    """Return a (SampleSource, n) representing the problem"""

    if prob_label not in label2fname:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(label2fname.keys())) )
    fname = label2fname[prob_label]
    tst_data, n = load_nips_TSTData(fname)
    ss = data.SSResample(tst_data)
    return ss, n

def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_dataset(prob_label)

def run_dataset(prob_label):
    """Run the experiment"""
    sample_source, n = get_sample_source(prob_label)

    # ///////  submit jobs //////////
    # create folder name string
    home = os.path.expanduser("~")
    foldername = os.path.join(home, "freqopttest_slurm", 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x  #methods
    aggregators = np.empty((reps, n_methods ), dtype=object)
    d = sample_source.dim()
    for r in range(reps):
        for mi, f in enumerate(method_job_funcs):
            # name used to save the result
            func_name = f.__name__
            fname = '%s-%s-J%d_r%d_d%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, J, r, d, alpha, tr_proportion)
            if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                logger.info('%s exists. Load and return.'%fname)
                test_result = glo.ex_load_result(ex, prob_label, fname)

                sra = SingleResultAggregator()
                if test_result is SingleResult:
                    sra.submit_result(test_result)
                else:
                    sra.submit_result(SingleResult(test_result))

                aggregators[r, mi] = sra
            else:
                # result not exists or rerun
                job = Ex4Job(SingleResultAggregator(), prob_label, r, n, f)
                agg = engine.submit_job(job)
                aggregators[r, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    test_results = np.empty((reps, n_methods), dtype=object)
    for r in range(reps):
        for mi, f in enumerate(method_job_funcs):
            logger.info("Collecting result (%s, r=%d)" % (f.__name__, r ))
            # let the aggregator finalize things
            aggregators[r, mi].finalize()

            # aggregators[i].get_final_result() returns a SingleResult instance,
            # which we need to extract the actual result
            test_result = aggregators[r, mi].get_final_result().result
            if isinstance(test_result, SingleResult):
                test_result = test_result.result
            if isinstance(test_result, SingleResult):
                test_result = test_result.result
            if isinstance(test_result, SingleResult):
                test_result = test_result.result
            test_results[r, mi] = test_result

            func_name = f.__name__
            fname = '%s-%s-J%d_r%d_d%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, J, r, d, alpha, tr_proportion)
            glo.ex_save_result(ex, test_result, prob_label, fname)

    func_names = [f.__name__ for f in method_job_funcs]
    func2labels = exglobal.get_func2label_map()
    method_labels = [func2labels[f] for f in func_names if f in func2labels]
    # save results 
    results = {'results': test_results, 'n': n, 'data_fname':label2fname[prob_label],
            'alpha': alpha, 'J': J, 'sample_source': sample_source, 
            'tr_proportion': tr_proportion, 'method_job_funcs': method_job_funcs, 
            'prob_label': prob_label, 'method_labels': method_labels}
    
    # class name 
    fname = 'ex%d-%s-me%d_J%d_rs%d_nma%d_d%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, J, reps, n, d, alpha, tr_proportion)
    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


if __name__ == '__main__':
    main()



