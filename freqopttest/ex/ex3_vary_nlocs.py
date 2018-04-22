"""Simulation to examine the type-1 error or test power as J, the number of 
test locations/test frequencies increases"""
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



def job_met_opt10(sample_source, tr, te, r, J):
    """MeanEmbeddingTest with test locations optimzied.
    Large step size
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    met_opt_options = {'n_test_locs': J, 'max_iter': 50, 
            'locs_step_size': 10.0, 'gwidth_step_size': 0.2, 'seed': r+92856,
            'tol_fun': 1e-3}
    test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
    met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
    met_opt_test  = met_opt.perform_test(te)
    return met_opt_test

def job_met_gwgrid(sample_source, tr, te, r, J):
    """MeanEmbeddingTest. Optimize only the Gaussian width with grid search
    Fix the test locations."""
    # optimize on the training set
    T_randn = tst.MeanEmbeddingTest.init_locs_2randn(tr, J, seed=r+92856)
    med = util.meddistance(tr.stack_xy(), 1000)
    list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-5, 5, 40) ) ) )
    list_gwidth.sort()
    besti, powers = tst.MeanEmbeddingTest.grid_search_gwidth(tr, T_randn,
            list_gwidth, alpha)

    best_width2 = list_gwidth[besti]
    met_grid = tst.MeanEmbeddingTest(T_randn, best_width2, alpha)
    return met_grid.perform_test(te)


def job_scf_opt10(sample_source, tr, te, r, J):
    """SmoothCFTest with frequencies optimized."""
    op = {'n_test_freqs': J, 'max_iter': 100, 'freqs_step_size': 5.0, 
            'gwidth_step_size': 0.2, 'seed': r+92856, 'tol_fun': 1e-4}
    test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
    scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
    scf_opt_test = scf_opt.perform_test(te)
    return scf_opt_test


def job_scf_gwgrid(sample_source, tr, te, r, J):
    rand_state = np.random.get_state()
    np.random.seed(r+92856)

    d = tr.dim()
    T_randn = np.random.randn(J, d)
    np.random.set_state(rand_state)

    # grid search to determine the initial gwidth
    mean_sd = tr.mean_std()
    scales = 2.0**np.linspace(-4, 4, 20)
    list_gwidth = np.hstack( (mean_sd*scales*(d**0.5), 2**np.linspace(-10, 10, 20) ))
    list_gwidth.sort()
    besti, powers = tst.SmoothCFTest.grid_search_gwidth(tr, T_randn,
            list_gwidth, alpha)
    # initialize with the best width from the grid search
    best_width = list_gwidth[besti]
    scf_gwgrid = tst.SmoothCFTest(T_randn, best_width, alpha)
    return scf_gwgrid.perform_test(te)


# Define our custom Job, which inherits from base class IndependentJob
class Ex3Job(IndependentJob):
   
    def __init__(self, aggregator, sample_source, prob_label, rep, job_func, n_locs):
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
        self.n_locs = n_locs

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
        test_result = job_func(sample_source, tr, te, r, self.n_locs)

        # create ScalarResult instance
        result = SingleResult(test_result)
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done. ex2: %s, r=%d, d=%d,  "%(job_func.__name__, r, d))

        # save result
        func_name = job_func.__name__
        J = self.n_locs
        fname = '%s-%s-J%d_n%d_r%d_a%.3f_trp%.2f.p' \
            %(prob_label, func_name, J, sample_size, r, alpha, tr_proportion)
        glo.ex_save_result(ex, test_result, prob_label, fname)



# This import is needed so that pickle knows about the class Ex3Job.
# pickle is used when collecting the results from the submitted jobs.
from freqopttest.ex.ex3_vary_nlocs import job_met_opt10
from freqopttest.ex.ex3_vary_nlocs import job_met_gwgrid
from freqopttest.ex.ex3_vary_nlocs import job_scf_opt10
from freqopttest.ex.ex3_vary_nlocs import job_scf_gwgrid
from freqopttest.ex.ex3_vary_nlocs import Ex3Job


#--- experimental setting -----
ex = 3

# sample size = n (the number training and test sizes)
sample_size = 2000

alpha = 0.01
tr_proportion = 0.5
# repetitions for each dimension
reps = 100
# list of number of test locations/frequencies
#Js = [5, 10, 15, 20, 25]
#Js = range(2, 6+1)
Js = [300, 470, 500]

#method_job_funcs = [  job_met_opt10, job_met_gwgrid, job_scf_opt10, job_scf_gwgrid]
method_job_funcs = [  job_met_opt10]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (di, r) already exists.
is_rerun = False

#---------------------------

def get_sample_source(prob_label):
    """Return a SampleSource corresponding to the problem label.
    """
    # map: prob_label -> sample_source
    prob2ss = { 
            'SSBlobs': data.SSBlobs(),
            'gmd_d100': data.SSGaussMeanDiff(d=100, my=1.0),
            'gmd_d2': data.SSGaussMeanDiff(d=2, my=1.0),
            'gvd_d50': data.SSGaussVarDiff(d=50), 
            'gvd_d100': data.SSGaussVarDiff(d=100), 
            # The null is true
            'sg_d50': data.SSSameGauss(d=50)
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
    ss = get_sample_source(prob_label)

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
    # repetitions x #J x #methods
    aggregators = np.empty((reps, len(Js), n_methods ), dtype=object)
    for r in range(reps):
        for ji, J in enumerate(Js):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-J%d_n%d_r%d_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, J, sample_size, r, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    test_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(test_result))
                    aggregators[r, ji, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex3Job(SingleResultAggregator(), ss,
                            prob_label, r, f, J)
                    agg = engine.submit_job(job)
                    aggregators[r, ji, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    test_results = np.empty((reps, len(Js), n_methods), dtype=object)
    for r in range(reps):
        for ji, J in enumerate(Js):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, J=%d)" % (f.__name__, r, J))
                # let the aggregator finalize things
                aggregators[r, ji, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                test_result = aggregators[r, ji, mi].get_final_result().result
                test_results[r, ji, mi] = test_result

    func_names = [f.__name__ for f in method_job_funcs]
    func2labels = exglobal.get_func2label_map()
    method_labels = [func2labels[f] for f in func_names if f in func2labels]
    # save results 
    results = {'test_results': test_results, 'list_J': Js, 
            'alpha': alpha, 'J': J, 'sample_source': ss, 
            'tr_proportion': tr_proportion, 'method_job_funcs': method_job_funcs, 
            'prob_label': prob_label, 'sample_size': sample_size, 
            'method_labels': method_labels}
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_n%d_jmi%d_jma%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, reps, sample_size, min(Js), max(Js), alpha, 
                tr_proportion)
    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


if __name__ == '__main__':
    main()

