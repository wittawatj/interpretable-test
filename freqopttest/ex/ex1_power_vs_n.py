"""Simulation to test the test power vs increasing sample size"""

__author__ = 'wittawat'

import freqopttest.data as data
import freqopttest.tst as tst
import freqopttest.glo as glo
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
import numpy as np
import os


def job_met_heu(tr, te, r, ni, n):
    """MeanEmbeddingTest with test_locs randomized. 
    tr unused."""
    # MeanEmbeddingTest random locations
    met_heu = tst.MeanEmbeddingTest.create_fit_gauss_heuristic(te, J, alpha, seed=180)
    met_heu_test = met_heu.perform_test(te)
    return met_heu_test

def job_met_opt(tr, te, r, ni, n):
    """MeanEmbeddingTest with test locations optimzied.
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    met_opt_options = {'n_test_locs': J, 'max_iter': 300, 
            'locs_step_size': 0.1, 'gwidth_step_size': 0.02, 'seed': r,
            'tol_fun': 1e-4}
    test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, **met_opt_options)
    met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
    met_opt_test  = met_opt.perform_test(te)
    return met_opt_test

def job_met_gwopt(tr, te, r, ni, n):
    """MeanEmbeddingTest. Optimize only the Gaussian width. 
    Fix the test locations."""
    op_gwidth = {'max_iter': 300, 'gwidth_step_size': 0.1,  
                 'batch_proportion': 1.0, 'tol_fun': 1e-4}
    # optimize on the training set
    T_randn = tst.MeanEmbeddingTest.init_locs_randn(tr, J, seed=r)
    gwidth, info = tst.MeanEmbeddingTest.optimize_gwidth(tr, T_randn, **op_gwidth)
    met_gwopt = tst.MeanEmbeddingTest(T_randn, gwidth, alpha)
    return met_gwopt.perform_test(te)

def job_scf_randn(tr, te, r, ni, n):
    """SmoothCFTest with frequencies drawn from randn(). tr unused."""
    scf_randn = tst.SmoothCFTest.create_randn(te, J, alpha, seed=19)
    scf_randn_test = scf_randn.perform_test(te)
    return scf_randn_test

def job_scf_opt(tr, te, r, ni, n):
    """SmoothCFTest with frequencies optimized."""
    op = {'n_test_freqs': J, 'max_iter': 300, 'freqs_step_size': 0.1, 
            'gwidth_step_size': 0.02, 'seed': r, 'tol_fun': 1e-4}
    test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tr, **op)
    scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha)
    scf_opt_test = scf_opt.perform_test(te)
    return scf_opt_test

def job_scf_gwopt(tr, te, r, ni, n):
    """SmoothCFTest. Optimize only the Gaussian width. 
    Fix the test frequencies"""
    op_gwidth = {'max_iter': 300, 'gwidth_step_size': 0.1,  
                 'batch_proportion': 1.0, 'tol_fun': 1e-4}
    # optimize on the training set
    rand_state = np.random.get_state()
    np.random.seed(seed=r)
    ss, _ = get_sample_source()
    T_randn = np.random.randn(J, ss.dim())
    np.random.set_state(rand_state)

    gwidth, info = tst.SmoothCFTest.optimize_gwidth(tr, T_randn, **op_gwidth)
    scf_gwopt = tst.SmoothCFTest(T_randn, gwidth, alpha)
    return scf_gwopt.perform_test(te)

def job_hotelling(tr, te, r, ni, n):
    """Hotelling T-squared test"""
    htest = tst.HotellingT2Test(alpha=alpha)
    return htest.perform_test(te)


# Define our custom Job, which inherits from base class IndependentJob
class Ex1Job(IndependentJob):
   
    def __init__(self, aggregator, sample_source, prob_label, rep, ni, n, job_func):
        d = sample_source.dim()
        ntr = int(n*tr_proportion)
        walltime = 60*59*24 if d*ntr/4 >= 10000 else 60*59
        memory = int(ntr*5e-3) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        self.sample_source = sample_source
        self.prob_label = prob_label
        self.rep = rep
        self.ni = ni
        self.n = n
        self.job_func = job_func

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        logger.info("computing")
        
        sample_source = self.sample_source 
        r = self.rep
        ni = self.ni 
        n = self.n
        job_func = self.job_func

        tst_data = sample_source.sample(n, seed=r)
        tr, te = tst_data.split_tr_te(tr_proportion=tr_proportion, seed=r+20 )
        test_result = job_func(tr, te, r, ni, n)

        # create ScalarResult instance
        result = SingleResult(test_result)
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done. ex1: %s, r=%d, n=%d,  "%(job_func.__name__, r, n))

        # save result
        prob_label = self.prob_label
        func_name = job_func.__name__
        fname = '%s-%s-J%d_r%d_n%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, J, r, n, alpha, tr_proportion)
        glo.ex_save_result(ex, fname, test_result)


# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from freqopttest.ex.ex1_power_vs_n import job_met_heu
from freqopttest.ex.ex1_power_vs_n import job_met_opt
from freqopttest.ex.ex1_power_vs_n import job_met_gwopt
from freqopttest.ex.ex1_power_vs_n import job_scf_randn
from freqopttest.ex.ex1_power_vs_n import job_scf_opt
from freqopttest.ex.ex1_power_vs_n import job_scf_gwopt
from freqopttest.ex.ex1_power_vs_n import job_hotelling
from freqopttest.ex.ex1_power_vs_n import Ex1Job


#--- experimental setting -----
ex = 1
# SSBlobs
#sample_sizes = [i*1000 for i in range(1, 14+1)]

# gmd_d20
sample_sizes = [i*1000 for i in range(1, 8+1)]

# number of test locations / test frequencies J
J = 5
alpha = 0.01
tr_proportion = 0.5
# repetitions for each sample size 
reps = 50
method_job_funcs = [ job_met_opt, job_met_gwopt, 
         job_scf_opt, job_scf_gwopt, job_hotelling]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (ni, r) already exists.
is_rerun = False
#---------------------------

def get_sample_source():
    """Return a SampleSource representing the problem, and a label for file 
    naming in a 2-tuple"""
    #sample_source = data.SSBlobs()
    #label = 'SSBlobs'

    #d = 20
    #sample_source = data.SSGaussMeanDiff(d=d, my=1.0)
    #label = 'gmd_d%d'%d

    # The null is true
    d = 5
    sample_source = data.SSSameGauss(d=d)
    label = 'sg_d%d'%d

    return (sample_source, label)

def main():
    """Run the experiment"""

    # ///////  submit jobs //////////
    # create folder name string
    home = os.path.expanduser("~")
    foldername = os.path.join(home, "freqopttest_slurm")
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e1_", parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(sample_sizes) x #methods
    aggregators = np.empty((reps, len(sample_sizes), n_methods ), dtype=object)
    sample_source, prob_label = get_sample_source()
    for r in range(reps):
        for ni, n in enumerate(sample_sizes):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-J%d_r%d_n%d_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, J, r, n, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    test_result = glo.ex_load_result(ex, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(test_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex1Job(SingleResultAggregator(), sample_source,
                            prob_label, r, ni, n, f)
                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    test_results = np.empty((reps, len(sample_sizes), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(sample_sizes):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, n=%d)" % (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                test_result = aggregators[r, ni, mi].get_final_result().result
                test_results[r, ni, mi] = test_result

    # save results 
    results = {'test_results': test_results, 'sample_sizes': sample_sizes, 
            'alpha': alpha, 'J': J, 'sample_source': sample_source, 
            'tr_proportion': 0.5, 'method_job_funcs': method_job_funcs, 
            'prob_label': prob_label}
    
    # class name 
    fname = 'ex1-%s-me%d_J%d_rs%d_nmi%d_nma%d_a%.3f_trp%.2f.p' \
        %(prob_label, n_methods, J, reps, min(sample_sizes), max(sample_sizes), alpha, 
                tr_proportion)
    glo.ex_save_result(ex, fname, results)


if __name__ == '__main__':
    main()

