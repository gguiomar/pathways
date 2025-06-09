"""
Test module for pathway agents.
Handles the creation of test datasets in CONTROL conditions.
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

def setup_test_parameters(tf_m):
    """
    Set up test parameters for the pathway agents.
    
    Args:
        tf_m: Trained agent
        
    Returns:
        dict: Test parameters
    """
    tf_m.n_test_episodes = 10000
    tf_m.perturbation_type = False
    tf_m.trial_subset = [3, 5, 6, 7, 8, 9, 10, 11, 12]  # tone subset not trial subset
    
    tf_m.w_CD = 0.51
    tf_m.w_CI = 0.5
    tf_m.beta = 1.5 
    tf_m.obs_var1 = 0
    
    return tf_m

def run_control_test(sim_data, tf_m, param, force_retest=False):
    """
    Run control test experiments.
    
    Args:
        sim_data: Training simulation data
        tf_m: Trained agent
        param: Parameters
        force_retest: If True, forces retesting even if saved data exists
        
    Returns:
        dict: Test data
    """
    # Check if test data already exists
    if not force_retest and check_data_exists("control_test_data"):
        print("Loading existing control test data...")
        saved_data = load_sim_data("control_test_data")
        if saved_data is not None:
            return saved_data
    
    print("Running control test...")
    
    # Setup test parameters
    tf_m = setup_test_parameters(tf_m)
    
    # Print tone list for verification
    tone_list = [param['second_tone_list'][i] for i in tf_m.trial_subset]
    print("Tone list:", tone_list)
    
    # Run test
    print("Testing agent with tqdm progress bar...")
    test_data = tf_m.test_agent(sim_data)
    
    # Save test data
    save_sim_data(test_data, "control_test_data")
    
    return test_data

def plot_control_test_results(test_data, param, tf_m, save_plots=True):
    """
    Plot control test results.
    
    Args:
        test_data: Test simulation data
        param: Parameters
        tf_m: Trained agent
        save_plots: Whether to save plots to files
        
    Returns:
        matplotlib.figure.Figure: Test results figure
    """
    f_size = [12, 10]
    init_f = 0.7
    n_bins = 14
    s_noise = 0
    
    tf_a = pathway_analysis_v6(test_data, param)
    f_control = tf_a.grid_plot_test(f_size, '2C3A-CONTROL', test_data, param, tf_m, 
                                   init_f, n_bins, s_noise, tf_m.trial_subset)
    
    # Print performance metrics
    performance = tf_a.get_perf(test_data)
    p_break = tf_a.get_pbreak(test_data, tf_m)
    print(f'Performance: {performance:.3f} // P(Break): {p_break:.3f}')
    
    if save_plots:
        f_control.savefig('plots/control_test_results.png', bbox_inches='tight', dpi=300)
        print("Control test results saved to plots/control_test_results.png")
    
    return f_control

def get_test_metrics(test_data, tf_m):
    """
    Get performance metrics from test data.
    
    Args:
        test_data: Test simulation data
        tf_m: Trained agent
        
    Returns:
        dict: Performance metrics
    """
    tf_a = pathway_analysis_v6(test_data, {})
    
    performance = tf_a.get_perf(test_data)
    p_break = tf_a.get_pbreak(test_data, tf_m)
    
    metrics = {
        'performance': performance,
        'p_break': p_break
    }
    
    return metrics

if __name__ == "__main__":
    # This would typically be called from main.py, but can be run independently
    print("Test module - should be called from main.py with trained agent")
    print("To run independently, first train an agent using training.py")
