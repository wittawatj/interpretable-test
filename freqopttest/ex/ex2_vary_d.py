"""Simulation to examine the type-1 error or test power as the dimension 
increases"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range
__author__ = 'wittawat'

import freqopttest.ex.ex1_power_vs_n as ex1
import freqopttest.data as data
import freqopttest.tst as tst
import freqopttest.glo as glo
import freqopttest.util as util 
import freqopttest.kernel as kernel 
from . import exglobal

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and freqopttest have to be in the globl search path (.bashrc)
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
    """MeanEmbeddingTest with test locations optimzied.
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    with util.ContextTimer() as t:
        met_opt_options = {'n_test_locs': J, 'max_iter': 200, 
                'locs_step_size': 0.1, 'gwidth_step_size': 0.1, 'seed': r+92856,
                'tol_fun': 1e-3}
        test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        met_opt_test  = met_opt.perform_test(te)
    return {
            #'test_method': met_opt, 
            'test_result': met_opt_test,
            'time_secs': t.secs}

def job_met_opt5(sample_source, tr, te, r):
    """MeanEmbeddingTest with test locations optimzied.
    Large step size
    Return results from calling perform_test()"""
    with util.ContextTimer() as t:
        # MeanEmbeddingTest. optimize the test locations
        met_opt_options = {'n_test_locs': J, 'max_iter': 200, 
                'locs_step_size': 0.5, 'gwidth_step_size': 0.1, 'seed': r+92856,
                'tol_fun': 1e-3}
        test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        met_opt_test  = met_opt.perform_test(te)
    return {
            #'test_method': met_opt, 
            'test_result': met_opt_test,
            'time_secs': t.secs}

def job_met_opt10(sample_source, tr, te, r):
    """MeanEmbeddingTest with test locations optimzied.
    Large step size
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    with util.ContextTimer() as t:
        met_opt_options = {'n_test_locs': J, 'max_iter': 100, 
                'locs_step_size': 10.0, 'gwidth_step_size': 0.2, 'seed': r+92856,
                'tol_fun': 1e-3}
        test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        met_opt_test  = met_opt.perform_test(te)
    return {
            #'test_method': met_opt, 
            'test_result': met_opt_test,
            'time_secs': t.secs}


def job_met_gwopt(sample_source, tr, te, r):
    """MeanEmbeddingTest. Optimize only the Gaussian width. 
    Fix the test locations."""
    raise ValueError('Use job_met_gwgrid instead')
    op_gwidth = {'max_iter': 200, 'gwidth_step_size': 0.1,  
                 'batch_proportion': 1.0, 'tol_fun': 1e-3}
    # optimize on the training set
    T_randn = tst.MeanEmbeddingTest.init_locs_2randn(tr, J, seed=r+92856)
    gwidth, info = tst.MeanEmbeddingTest.optimize_gwidth(tr, T_randn, **op_gwidth)
    met_gwopt = tst.MeanEmbeddingTest(T_randn, gwidth, alpha)
    return met_gwopt.perform_test(te)

def job_met_gwgrid(sample_source, tr, te, r):
    """MeanEmbeddingTest. Optimize only the Gaussian width with grid search
    Fix the test locations."""
    return ex1.job_met_gwgrid('', tr, te, r, -1, -1)

def job_scf_opt(sample_source, tr, te, r):
    """SmoothCFTest with frequencies optimized."""
    with util.ContextTimer() as t:
        op = {'n_test_freqs': J, 'max_iter': 200, 'freqs_step_size': 0.1, 
                'gwidth_step_size': 0.1, 'seed': r+92856, 'tol_fun': 1e-3}
        test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
        scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
        scf_opt_test = scf_opt.perform_test(te)
    return {
            #'test_method': scf_opt, 
            'test_result': scf_opt_test,
            'time_secs': t.secs}

def job_scf_opt10(sample_source, tr, te, r):
    """SmoothCFTest with frequencies optimized."""
    with util.ContextTimer() as t:
        op = {'n_test_freqs': J, 'max_iter': 100, 'freqs_step_size': 1.0, 
                'gwidth_step_size': 0.1, 'seed': r+92856, 'tol_fun': 1e-3}
        test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
        scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
        scf_opt_test = scf_opt.perform_test(te)
    return {
            #'test_method': scf_opt, 
            'test_result': scf_opt_test,
            'time_secs': t.secs}

def job_scf_gwopt(sample_source, tr, te, r):
    """SmoothCFTest. Optimize only the Gaussian width. 
    Fix the test frequencies"""
    raise ValueError('Use job_scf_gwgrid instead')
    op_gwidth = {'max_iter': 200, 'gwidth_step_size': 0.1,  
                 'batch_proportion': 1.0, 'tol_fun': 1e-3}
    # optimize on the training set
    rand_state = np.random.get_state()
    np.random.seed(seed=r+92856)
    T_randn = np.random.randn(J, sample_source.dim())
    np.random.set_state(rand_state)

    gwidth, info = tst.SmoothCFTest.optimize_gwidth(tr, T_randn, **op_gwidth)
    scf_gwopt = tst.SmoothCFTest(T_randn, gwidth, alpha)
    return scf_gwopt.perform_test(te)

def job_scf_gwgrid(sample_source, tr, te, r):
    return ex1.job_scf_gwgrid('', tr, te, r, -1 , -1)

def job_quad_mmd(sample_source, tr, te, r):
    """Quadratic mmd with grid search to choose the best Gaussian width."""
    # If n is too large, pairwise meddian computation can cause a memory error. 

    with util.ContextTimer() as t:
        med = util.meddistance(tr.stack_xy(), 1000)
        list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 30) ) ) )
        list_gwidth.sort()
        list_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

        # grid search to choose the best Gaussian width
        besti, powers = tst.QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
        # perform test 
        best_ker = list_kernels[besti]
        mmd_test = tst.QuadMMDTest(best_ker, n_permute=400, alpha=alpha)
        test_result = mmd_test.perform_test(te)
    return {
            #'test_method': mmd_test, 
            'test_result': test_result,
            'time_secs': t.secs}

def job_lin_mmd(sample_source, tr, te, r):
    """Linear mmd with grid search to choose the best Gaussian width."""
    # should be completely deterministic

    with util.ContextTimer() as t:
        # If n is too large, pairwise meddian computation can cause a memory error. 
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
    return {
            #'test_method': lin_mmd_test, 
            'test_result': test_result,
            'time_secs': t.secs}

def job_hotelling(sample_source, tr, te, r):
    """Hotelling T-squared test"""
    with util.ContextTimer() as t:
        htest = tst.HotellingT2Test(alpha=alpha)
        result = htest.perform_test(te)
    return {'test_method': htest, 
            'test_result': result,
            'time_secs': t.secs}


# Define our custom Job, which inherits from base class IndependentJob
class Ex2Job(IndependentJob):
   
    def __init__(self, aggregator, sample_source, prob_label, rep, job_func):
        d = sample_source.dim()
        #walltime = 60*59*24 if d*sample_size*tr_proportion/15 >= 8000 else 60*59
        walltime = 60*59*24 
        memory = int(tr_proportion*sample_size*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        self.sample_source = sample_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        
        sample_source = self.sample_source 
        r = self.rep
        d = sample_source.dim()
        job_func = self.job_func
        logger.info("computing. %s. r=%d, d=%d"%(job_func.__name__, r, d))

        # sample_size is a global variable
        tst_data = sample_source.sample(sample_size, seed=r)
        tr, te = tst_data.split_tr_te(tr_proportion=tr_proportion, seed=r+20 )
        prob_label = self.prob_label
        test_result = job_func(sample_source, tr, te, r)

        # create ScalarResult instance
        result = SingleResult(test_result)
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done. ex2: %s, r=%d, d=%d,  "%(job_func.__name__, r, d))

        # save result
        func_name = job_func.__name__
        fname = '%s-%s-J%d_r%d_n%d_d%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, J, r, sample_size, d, alpha, tr_proportion)
        glo.ex_save_result(ex, test_result, prob_label, fname)



# This import is needed so that pickle knows about the class Ex2Job.
# pickle is used when collecting the results from the submitted jobs.
from freqopttest.ex.ex2_vary_d import job_met_opt
from freqopttest.ex.ex2_vary_d import job_met_opt5
from freqopttest.ex.ex2_vary_d import job_met_opt10
from freqopttest.ex.ex2_vary_d import job_met_gwopt
from freqopttest.ex.ex2_vary_d import job_met_gwgrid
from freqopttest.ex.ex2_vary_d import job_scf_opt
from freqopttest.ex.ex2_vary_d import job_scf_opt10
from freqopttest.ex.ex2_vary_d import job_scf_gwopt
from freqopttest.ex.ex2_vary_d import job_scf_gwgrid
from freqopttest.ex.ex2_vary_d import job_quad_mmd
from freqopttest.ex.ex2_vary_d import job_lin_mmd
from freqopttest.ex.ex2_vary_d import job_hotelling
from freqopttest.ex.ex2_vary_d import Ex2Job


#--- experimental setting -----


ex = 2

# sample size = n (the number training and test sizes are n/2)
sample_size = 20000

# number of test locations / test frequencies J
J = 5
alpha = 0.01
tr_proportion = 0.5
# repetitions for each dimension
reps = 500
#method_job_funcs = [ job_met_opt,  job_met_opt10, job_met_gwgrid,
#         job_scf_opt, job_scf_opt10, job_scf_gwgrid, job_lin_mmd, job_hotelling]
method_job_funcs = [  job_met_opt10, job_met_gwgrid,
        job_scf_opt10, job_scf_gwgrid, 
        #job_quad_mmd, 
        job_lin_mmd, job_hotelling]
#method_job_funcs = [  job_lin_mmd, job_hotelling]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (di, r) already exists.
is_rerun = False
#---------------------------

def get_sample_source_list(prob_label):
    """Return a list of SampleSource's representing the problems, each 
    corresponding to one dimension in the list.
    """
    # map: prob_label -> [sample_source]
    # dimensions to try 
    dimensions = [5] + [100*i for i in range(1, 5+1)] 
    high_dims = [5] + [300*i for i in range(1, 5+1)]
    low_dims = [2*d for d in range(1, 7+1)]
    prob2ss = { 
            'gmd': [data.SSGaussMeanDiff(d=d, my=1.0) for d in high_dims],
            'gvd': [data.SSGaussVarDiff(d=d) for d in dimensions], 
            # The null is true
            'sg': [data.SSSameGauss(d=d) for d in high_dims],
            'sg_low': [data.SSSameGauss(d=d) for d in low_dims]
            }
    if prob_label not in prob2ss:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(prob2ss.keys())) )
    return prob2ss[prob_label]

def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_dataset(prob_label)

def run_dataset(prob_label):
    """Run the experiment"""
    list_ss = get_sample_source_list(prob_label)
    dimensions = [ss.dim() for ss in list_ss]

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
    # repetitions x len(dimensions) x #methods
    aggregators = np.empty((reps, len(dimensions), n_methods ), dtype=object)
    for r in range(reps):
        for di, d in enumerate(dimensions):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-J%d_r%d_n%d_d%d_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, J, r, sample_size, d, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    test_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(test_result))
                    aggregators[r, di, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex2Job(SingleResultAggregator(), list_ss[di],
                            prob_label, r, f)
                    agg = engine.submit_job(job)
                    aggregators[r, di, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    test_results = np.empty((reps, len(dimensions), n_methods), dtype=object)
    for r in range(reps):
        for di, d in enumerate(dimensions):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, d=%d)" % (f.__name__, r, d))
                # let the aggregator finalize things
                aggregators[r, di, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                test_result = aggregators[r, di, mi].get_final_result().result
                test_results[r, di, mi] = test_result

    func_names = [f.__name__ for f in method_job_funcs]
    func2labels = exglobal.get_func2label_map()
    method_labels = [func2labels[f] for f in func_names if f in func2labels]
    # save results 
    results = {'test_results': test_results, 'dimensions': dimensions, 
            'alpha': alpha, 'J': J, 'list_sample_source': list_ss, 
            'tr_proportion': tr_proportion, 'method_job_funcs': method_job_funcs, 
            'prob_label': prob_label, 'sample_size': sample_size, 
            'method_labels': method_labels}
    
    # class name 
    fname = 'ex2-%s-me%d_J%d_rs%d_n%d_dmi%d_dma%d_a%.3f_trp%.2f.p' \
        %(prob_label, n_methods, J, reps, sample_size, min(dimensions),
                max(dimensions), alpha, tr_proportion)
    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


if __name__ == '__main__':
    main()

