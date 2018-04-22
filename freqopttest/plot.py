"""
A module for plotting experimental results
"""

__author__ = 'wittawat'

from builtins import range
import freqopttest.ex.exglobal as exglo
import freqopttest.glo as glo
import matplotlib.pyplot as plt
import autograd.numpy as np


def plot_prob_stat_above_thresh(ex, fname, h1_true, func_xvalues, xlabel,
        func_title=None):
    """
    plot the empirical probability that the statistic is above the theshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - h1_true: True if H1 is true 
    - func_xvalues: function taking results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot

    Return loaded results
    """

    results = glo.ex_load_result(ex, fname)
    f_pval = lambda job_result: job_result['test_result']['h0_rejected']
    #f_pval = lambda job_result: job_result['h0_rejected']
    vf_pval = np.vectorize(f_pval)
    pvals = vf_pval(results['test_results'])
    repeats, _, n_methods = results['test_results'].shape
    mean_rejs = np.mean(pvals, axis=0)
    #std_pvals = np.std(pvals, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = exglo.func_plot_fmt_map()
    method_labels = exglo.get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label)
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Test power' if h1_true else 'Type-I error'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks( np.hstack((xvalues) ))
    
    alpha = results['alpha']
    """
    if not h1_true:
        # plot Wald interval if H0 is true
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        z = stats.norm.isf( (1-confidence)/2.0)
        gap = z*(alpha*(1-alpha)/repeats)**0.5
        lb = alpha-gap
        ub = alpha+gap
        plt.plot(test_sizes, np.repeat(lb, len(test_sizes)), '--', linewidth=2, 
                 label='99%-Conf', color='k')
        plt.plot(test_sizes, np.repeat(ub, len(test_sizes)), '--', linewidth=2, color='k')
        plt.ylim([lb-0.005, ub+0.005])
    """
    
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results
        

def plot_runtime(ex, fname, func_xvalues, xlabel, func_title=None):
    results = glo.ex_load_result(ex, fname)
    value_accessor = lambda job_results: job_results['time_secs']
    vf_pval = np.vectorize(value_accessor)
    # results['test_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    times = vf_pval(results['test_results'])
    repeats, _, n_methods = results['test_results'].shape
    time_avg = np.mean(times, axis=0)
    time_std = np.std(times, axis=0)

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = exglo.func_plot_fmt_map()
    method_labels = exglo.get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plt.errorbar(xvalues, time_avg[:, i], yerr=time_std[:,i], fmt=fmt,
                label=method_label)
            
    ylabel = 'Time (s)'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.gca().set_yscale('log')
    plt.xlim([np.min(xvalues), np.max(xvalues)])
    plt.xticks( xvalues, xvalues)
    plt.legend(loc='best')
    title = '%s. %d trials. '%( results['prob_label'],
            repeats ) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results


