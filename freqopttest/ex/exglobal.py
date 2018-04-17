"""
A module containing functions shared among all experiments 
""" 

from builtins import zip
__author__ = 'wittawat'

def get_func2label_map():
    # map: job_func_name |-> plot label
    func_names = ['job_met_opt', 'job_met_opt5', 'job_met_opt10', 'job_met_gwopt',
            'job_met_gwgrid',
            'job_scf_opt', 'job_scf_opt10', 'job_scf_gwopt', 'job_scf_gwgrid', 
            'job_quad_mmd', 'job_quad_mmd_2U', 'job_lin_mmd', 'job_hotelling']

    labels = ['ME-full', 'ME-opt-0.5', 'ME-full', 'ME-gw-opt', 
            'ME-grid',
            'SCF-full', 'SCF-full', 'SCF-gw-opt', 'SCF-grid',
            'MMD-quad', 'MMD-2U', 'MMD-lin', '$T^2$']
    M = {k:v for (k,v) in zip(func_names, labels)}
    return M

def func_plot_fmt_map():
    """
    Return a map from job function names to matplotlib plot styles 
    """
    # line_styles = ['o-', 'x-',  '*-', '-_', 'D-', 'h-', '+-', 's-', 'v-', 
    #               ',-', '1-']
    M = {}
    M['job_met_opt'] = 'bo-'
    M['job_met_opt10'] = 'bo-'
    M['job_met_gwgrid'] = 'bo--'

    M['job_scf_opt'] = 'r*-'
    M['job_scf_opt10'] = 'r*-'
    M['job_scf_gwgrid'] = 'r*--'

    M['job_quad_mmd'] = 'g-^'
    M['job_lin_mmd'] = 'cD-'
    M['job_hotelling'] = 'yv-'
    return M

