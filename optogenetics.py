"""
Optogenetics module for pathway agents.
Performs optogenetics experiments for DLS and DMS pathways.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy as copy

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Local imports
from analysis.pathway_analysis import pathway_analysis_v6
from utils import save_sim_data, load_sim_data, check_data_exists

def run_dls_perturbation_experiments(sim_data, tf_m, param, test_data, force_rerun=False):
    """
    Run DLS (A2A) perturbation experiments.
    
    Args:
        sim_data: Training simulation data
        tf_m: Trained agent
        param: Parameters
        test_data: Control test data
        force_rerun: If True, forces rerunning even if saved data exists
        
    Returns:
        dict: DLS perturbation results
    """
    # Check if DLS perturbation data already exists
    if not force_rerun and check_data_exists("dls_perturbation_data"):
        print("Loading existing DLS perturbation data...")
        saved_data = load_sim_data("dls_perturbation_data")
        if saved_data is not None:
            return saved_data
    
    print("Running DLS perturbation experiments...")
    
    # DLS perturbation parameters
    pert_list_a2a = [0.94, 0.93, 0.92]
    
    # Initialize results storage
    results = {
        'pert_list_a2a': pert_list_a2a,
        'metric_a2a_bl': [],
        'pert_a2a_bl': [],
        'metric_a2a_cs': [],
        'pert_a2a_cs': [],
        'metric_a2a_cl': [],
        'pert_a2a_cl': []
    }
    
    tf_a = pathway_analysis_v6(test_data, param)
    
    # DLS-BL perturbation
    print("Running iDLS-BL perturbations...")
    tf_m.perturbation_type = 'iDLS-BL'
    for i, e in enumerate(tqdm(pert_list_a2a, desc="iDLS-BL")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_a2a_bl'].append([pert_data, e])
        results['metric_a2a_bl'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("iDLS-BL metrics:", results['metric_a2a_bl'])

    # DLS-CS perturbation
    print("Running iDLS-CS perturbations...")
    tf_m.perturbation_type = 'iDLS-CS'
    for i, e in enumerate(tqdm(pert_list_a2a, desc="iDLS-CS")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_a2a_cs'].append([pert_data, e])
        results['metric_a2a_cs'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("iDLS-CS metrics:", results['metric_a2a_cs'])

    # DLS-CL perturbation
    print("Running iDLS-CL perturbations...")
    tf_m.perturbation_type = 'iDLS-CL'
    for i, e in enumerate(tqdm(pert_list_a2a, desc="iDLS-CL")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_a2a_cl'].append([pert_data, e])
        results['metric_a2a_cl'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("iDLS-CL metrics:", results['metric_a2a_cl'])
    
    # Save results
    save_sim_data(results, "dls_perturbation_data")
    
    return results

def run_dms_perturbation_experiments(sim_data, tf_m, param, test_data, force_rerun=False):
    """
    Run DMS perturbation experiments.
    
    Args:
        sim_data: Training simulation data
        tf_m: Trained agent
        param: Parameters
        test_data: Control test data
        force_rerun: If True, forces rerunning even if saved data exists
        
    Returns:
        dict: DMS perturbation results
    """
    # Check if DMS perturbation data already exists
    if not force_rerun and check_data_exists("dms_perturbation_data"):
        print("Loading existing DMS perturbation data...")
        saved_data = load_sim_data("dms_perturbation_data")
        if saved_data is not None:
            return saved_data
    
    print("Running DMS perturbation experiments...")
    
    # DMS perturbation parameters (assuming same as DLS for now)
    pert_list_dm = [0.94, 0.93, 0.92]
    
    # Initialize results storage
    results = {
        'pert_list_dm': pert_list_dm,
        'metric_dm_bl': [],
        'pert_dm_bl': [],
        'metric_dm_cs': [],
        'pert_dm_cs': [],
        'metric_dm_cl': [],
        'pert_dm_cl': []
    }
    
    tf_a = pathway_analysis_v6(test_data, param)
    
    # DMS-BL perturbation
    print("Running dDMS-BL perturbations...")
    tf_m.perturbation_type = 'dDMS-BL'
    for i, e in enumerate(tqdm(pert_list_dm, desc="dDMS-BL")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_dm_bl'].append([pert_data, e])
        results['metric_dm_bl'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("dDMS-BL metrics:", results['metric_dm_bl'])

    # DMS-CS perturbation
    print("Running dDMS-CS perturbations...")
    tf_m.perturbation_type = 'dDMS-CS'
    for i, e in enumerate(tqdm(pert_list_dm, desc="dDMS-CS")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_dm_cs'].append([pert_data, e])
        results['metric_dm_cs'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("dDMS-CS metrics:", results['metric_dm_cs'])

    # DMS-CL perturbation
    print("Running dDMS-CL perturbations...")
    tf_m.perturbation_type = 'dDMS-CL'
    for i, e in enumerate(tqdm(pert_list_dm, desc="dDMS-CL")):
        tf_m.pert_mag = e
        pert_data = tf_m.test_agent(sim_data)
        results['pert_dm_cl'].append([pert_data, e])
        results['metric_dm_cl'].append(tf_a.get_behavior_metrics_opto(test_data, pert_data, tf_m))
    print("dDMS-CL metrics:", results['metric_dm_cl'])
    
    # Save results
    save_sim_data(results, "dms_perturbation_data")
    
    return results

def plot_dls_perturbation_results(dls_results, param, tf_m, save_plots=True):
    """
    Plot DLS perturbation results.
    
    Args:
        dls_results: DLS perturbation results
        param: Parameters
        tf_m: Trained agent
        save_plots: Whether to save plots to files
        
    Returns:
        list: List of figures
    """
    f_size = [12, 10]
    init_f = 0.7
    n_bins = 14
    
    tf_a = pathway_analysis_v6({}, param)
    figures = []
    
    # Plot BL perturbations
    for i in range(len(dls_results['pert_list_a2a'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_BL', 
                               dls_results['pert_a2a_bl'][i][0], param, tf_m, 
                               init_f, n_bins, dls_results['pert_a2a_bl'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dls_bl_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    # Plot CS perturbations
    for i in range(len(dls_results['pert_list_a2a'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_CS', 
                               dls_results['pert_a2a_cs'][i][0], param, tf_m, 
                               init_f, n_bins, dls_results['pert_a2a_cs'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dls_cs_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    # Plot CL perturbations
    for i in range(len(dls_results['pert_list_a2a'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_CL', 
                               dls_results['pert_a2a_cl'][i][0], param, tf_m, 
                               init_f, n_bins, dls_results['pert_a2a_cl'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dls_cl_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    if save_plots:
        print("DLS perturbation plots saved to plots/")
    
    return figures

def plot_dms_perturbation_results(dms_results, param, tf_m, save_plots=True):
    """
    Plot DMS perturbation results.
    
    Args:
        dms_results: DMS perturbation results
        param: Parameters
        tf_m: Trained agent
        save_plots: Whether to save plots to files
        
    Returns:
        list: List of figures
    """
    f_size = [12, 10]
    init_f = 0.7
    n_bins = 14
    
    tf_a = pathway_analysis_v6({}, param)
    figures = []
    
    # Plot BL perturbations
    for i in range(len(dms_results['pert_list_dm'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_BL', 
                               dms_results['pert_dm_bl'][i][0], param, tf_m, 
                               init_f, n_bins, dms_results['pert_dm_bl'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dms_bl_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    # Plot CS perturbations
    for i in range(len(dms_results['pert_list_dm'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_CS', 
                               dms_results['pert_dm_cs'][i][0], param, tf_m, 
                               init_f, n_bins, dms_results['pert_dm_cs'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dms_cs_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    # Plot CL perturbations
    for i in range(len(dms_results['pert_list_dm'])):
        f = tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_CL', 
                               dms_results['pert_dm_cl'][i][0], param, tf_m, 
                               init_f, n_bins, dms_results['pert_dm_cl'][i][1], 
                               tf_m.trial_subset)
        figures.append(f)
        if save_plots:
            f.savefig(f'plots/dms_cl_pert_{i}.png', bbox_inches='tight', dpi=300)
    
    if save_plots:
        print("DMS perturbation plots saved to plots/")
    
    return figures

if __name__ == "__main__":
    # This would typically be called from main.py, but can be run independently
    print("Optogenetics module - should be called from main.py with trained agent and test data")
    print("To run independently, first train an agent and run control tests")
