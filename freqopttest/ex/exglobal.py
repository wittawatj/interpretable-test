"""
A module containing functions shared among all experiments 
""" 

__author__ = 'wittawat'

def get_func2label_map():
    # map: job_func_name |-> plot label
    func_names = ['job_met_opt', 'job_met_opt5', 'job_met_opt10', 'job_met_gwopt',
            'job_met_gwgrid',
            'job_scf_opt', 'job_scf_opt10', 'job_scf_gwopt', 'job_scf_gwgrid', 
            'job_lin_mmd', 'job_hotelling']
    labels = ['ME-opt', 'ME-opt-0.5', 'ME-opt-1.0', 'ME-gw-opt', 
            'ME-gw-grid',
            'SCF-opt', 'SCF-opt-1.0', 'SCF-gw-opt', 'SCF-gw-grid',
            'MMD-lin', '$T^2$']
    M = {k:v for (k,v) in zip(func_names, labels)}
    return M
