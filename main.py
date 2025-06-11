"""
Main pipeline for pathway agents analysis.
Orchestrates training, testing, optogenetics experiments, and perceptual policy analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
import sys
import os
from os import walk
import pandas as pd
import pickle
import datetime
from IPython.display import clear_output, display
from numpy import linalg as LA

from tqdm import tqdm

from matplotlib import rc

# Local imports
from training import train_agent, plot_training_results, plot_neural_activity
from testing_control import run_control_test, plot_control_test_results, get_test_metrics
from optogenetics import (
    run_dls_perturbation_experiments, 
    run_dms_perturbation_experiments,
    plot_dls_perturbation_results,
    plot_dms_perturbation_results
)
from perceptualpolicy import (
    run_perceptual_policy_analysis,
    plot_perceptual_policy_results,
    print_analysis_summary
)
from utils import save_sim_data, load_sim_data, check_data_exists

# Set matplotlib parameters
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'axes.linewidth': 2}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 14})
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 2

def run_full_pipeline(force_retrain=False, force_retest=False, force_rerun_opto=False, 
                     force_rerun_ppolicy=False, save_plots=True):
    """
    Run the complete analysis pipeline.
    
    Args:
        force_retrain: Force retraining even if saved data exists
        force_retest: Force retesting even if saved data exists
        force_rerun_opto: Force rerunning optogenetics even if saved data exists
        force_rerun_ppolicy: Force rerunning perceptual policy even if saved data exists
        save_plots: Whether to save plots to files
        
    Returns:
        dict: Complete results from all analyses
    """
    print("="*80)
    print("STARTING FULL PATHWAY ANALYSIS PIPELINE")
    print("="*80)
    
    results = {}
    
    # 1. TRAINING
    print("\n1. TRAINING PHASE")
    print("-" * 40)
    sim_data, tf_m, param = train_agent(force_retrain=force_retrain)
    results['training'] = {
        'sim_data': sim_data,
        'tf_m': tf_m,
        'param': param
    }
    
    # Plot training results
    print("Plotting training results...")
    training_fig1 = plot_training_results(sim_data, param, tf_m, save_plots)
    training_fig2 = plot_neural_activity(sim_data, tf_m, save_plots)
    results['training']['figures'] = {
        'training_results': training_fig1,
        'neural_activity': training_fig2
    }
    
    # 2. CONTROL TESTING
    print("\n2. CONTROL TESTING PHASE")
    print("-" * 40)
    test_data = run_control_test(sim_data, tf_m, param, force_retest=force_retest)
    results['control_test'] = test_data
    
    # Plot control test results
    print("Plotting control test results...")
    control_fig = plot_control_test_results(test_data, param, tf_m, save_plots)
    test_metrics = get_test_metrics(test_data, tf_m)
    results['control_test']['figure'] = control_fig
    results['control_test']['metrics'] = test_metrics
    
    # 3. OPTOGENETICS EXPERIMENTS
    print("\n3. OPTOGENETICS EXPERIMENTS")
    print("-" * 40)
    
    # DLS perturbations
    print("Running DLS perturbation experiments...")
    dls_results = run_dls_perturbation_experiments(sim_data, tf_m, param, test_data, 
                                                  force_rerun=force_rerun_opto)
    results['dls_perturbations'] = dls_results
    
    # Plot DLS results
    print("Plotting DLS perturbation results...")
    dls_figures = plot_dls_perturbation_results(dls_results, param, tf_m, save_plots)
    results['dls_perturbations']['figures'] = dls_figures
    
    # DMS perturbations
    print("Running DMS perturbation experiments...")
    dms_results = run_dms_perturbation_experiments(sim_data, tf_m, param, test_data,
                                                  force_rerun=force_rerun_opto)
    results['dms_perturbations'] = dms_results
    
    # Plot DMS results
    print("Plotting DMS perturbation results...")
    dms_figures = plot_dms_perturbation_results(dms_results, param, tf_m, save_plots)
    results['dms_perturbations']['figures'] = dms_figures
    
    # 4. PERCEPTUAL POLICY ANALYSIS
    print("\n4. PERCEPTUAL POLICY ANALYSIS")
    print("-" * 40)
    ppolicy_results = run_perceptual_policy_analysis(sim_data, tf_m, param,
                                                    force_rerun=force_rerun_ppolicy)
    results['perceptual_policy'] = ppolicy_results
    
    # Plot perceptual policy results
    print("Plotting perceptual policy results...")
    ppolicy_figures = plot_perceptual_policy_results(ppolicy_results, save_plots)
    results['perceptual_policy']['figures'] = ppolicy_figures
    
    # Print analysis summary
    print_analysis_summary(ppolicy_results)
    
    # 5. SAVE COMPLETE RESULTS
    print("\n5. SAVING COMPLETE RESULTS")
    print("-" * 40)
    
    # Create a summary of all results (without the large data structures for storage efficiency)
    summary_results = {
        'training_metrics': {
            'n_episodes': param['n_episodes'],
            'n_states': param['n_states'],
            'beta': param['beta'],
            'gamma_v': param['gamma_v'],
            'gamma_dm': param['gamma_dm']
        },
        'control_test_metrics': test_metrics,
        'dls_perturbation_metrics': {
            'pert_list': dls_results['pert_list_a2a'],
            'bl_metrics': dls_results['metric_a2a_bl'],
            'cs_metrics': dls_results['metric_a2a_cs'],
            'cl_metrics': dls_results['metric_a2a_cl']
        },
        'dms_perturbation_metrics': {
            'pert_list': dms_results['pert_list_dm'],
            'bl_metrics': dms_results['metric_dm_bl'],
            'cs_metrics': dms_results['metric_dm_cs'],
            'cl_metrics': dms_results['metric_dm_cl']
        },
        'perceptual_policy_summary': {
            'transition_matrix_shape': ppolicy_results['T_test'].shape,
            'successor_rep_shape': ppolicy_results['M1_act'].shape,
            'n_test_episodes': ppolicy_results['tf_m_params']['n_test_episodes']
        }
    }
    
    save_sim_data(summary_results, "pipeline_summary")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Training performance: {test_metrics['performance']:.3f}")
    print(f"Breaking probability: {test_metrics['p_break']:.3f}")
    print(f"All plots saved to: plots/")
    print(f"All data saved to: simulation_data/")
    print("="*80)
    
    return results

def run_training_only(force_retrain=False, save_plots=True):
    """
    Run only the training phase.
    
    Args:
        force_retrain: Force retraining even if saved data exists
        save_plots: Whether to save plots to files
        
    Returns:
        dict: Training results
    """
    print("Running training only...")
    sim_data, tf_m, param = train_agent(force_retrain=force_retrain)
    
    # Plot results
    training_fig1 = plot_training_results(sim_data, param, tf_m, save_plots)
    training_fig2 = plot_neural_activity(sim_data, tf_m, save_plots)
    
    results = {
        'sim_data': sim_data,
        'tf_m': tf_m,
        'param': param,
        'figures': {
            'training_results': training_fig1,
            'neural_activity': training_fig2
        }
    }
    
    return results

def run_analysis_only(save_plots=True):
    """
    Run analysis on existing training data.
    
    Args:
        save_plots: Whether to save plots to files
        
    Returns:
        dict: Analysis results
    """
    print("Loading existing training data for analysis...")
    
    # Load training data
    training_data = load_sim_data("training_data")
    if training_data is None:
        raise ValueError("No training data found. Please run training first.")
    
    sim_data = training_data['sim_data']
    tf_m = training_data['tf_m']
    param = training_data['param']
    
    # Run control test
    test_data = run_control_test(sim_data, tf_m, param)
    
    # Run perceptual policy analysis
    ppolicy_results = run_perceptual_policy_analysis(sim_data, tf_m, param)
    
    # Plot results
    control_fig = plot_control_test_results(test_data, param, tf_m, save_plots)
    ppolicy_figures = plot_perceptual_policy_results(ppolicy_results, save_plots)
    
    results = {
        'control_test': test_data,
        'perceptual_policy': ppolicy_results,
        'figures': {
            'control_test': control_fig,
            'perceptual_policy': ppolicy_figures
        }
    }
    
    return results

if __name__ == "__main__":
    # Parse command line arguments for different run modes
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "training":
            print("Running training only...")
            results = run_training_only()
            
        elif mode == "analysis":
            print("Running analysis only...")
            results = run_analysis_only()
            
        elif mode == "full":
            print("Running full pipeline...")
            results = run_full_pipeline()
            
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: training, analysis, full")
            sys.exit(1)
    else:
        # Default: run full pipeline
        print("Running full pipeline (default)...")
        results = run_full_pipeline()
    
    # Show plots
    plt.show()
    print("Analysis complete!")
