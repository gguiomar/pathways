"""
Training script for 250,000 episodes.
Modified version of training.py to run extended training.
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

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from matplotlib import rc

# Local imports
from agents.pathway_agents import pathway_agents_v11
from analysis.pathway_analysis import pathway_analysis_v6
from utils import save_sim_data, load_sim_data, check_data_exists

# Set matplotlib parameters
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'axes.linewidth': 2}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 14})
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 2

def setup_training_parameters_250k():
    """
    Set up training parameters for 250,000 episodes.
    
    Returns:
        dict: Training parameters
    """
    # Training episodes - MODIFIED FOR 250K
    n_eps = 250000
    n_test_eps = int(n_eps * 0.7)

    # 3rd observer variables 
    obs_var = [0, 0]  # horizontal noise for the 2 observers - not using right now
    omega_c = 1
    alpha3 = 0.01  # trying out a scaling learning rate

    # there's an extra state added here due to a bug in the psychometric
    stl = [0.1, 0.3, 0.5, 0.6000, 0.8, 1.0500, 1.2600, 1.3800, 1.6200, 1.7400, 1.9500, 2.2, 2.4000, 2.5000]

    t_dwell_var = 23 
    t_dwell_min = 65 
    t_dwell_ctr = 70 
    t_dwell_max = 100 

    t_dwell_list = np.arange(t_dwell_min, t_dwell_max, 3)
    n_trajs = t_dwell_list.shape[0]
    print('Number of trajectories', n_trajs)
    env_param = [t_dwell_ctr, t_dwell_var, t_dwell_min, t_dwell_max, n_trajs, t_dwell_list]

    # dls parameters
    dp_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]
    ip_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]

    # dms parameters
    dp_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]
    ip_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]

    train_sim_notes = "Training with 250,000 episodes - split DMS and symmetric transfer functions"

    param = {
        'n_states': 40, 
        'beta': 1.5, 
        'gamma_v': 0.98, 
        'gamma_dm': 0.98, 
        'rwd_mag': 10, 
        'pun_mag': -5, 
        'n_episodes': n_eps, 
        'n_test_eps': n_test_eps, 
        'second_tone_list': stl, 
        'alpha_i': 20, 
        'env_param': env_param, 
        'obs_var': obs_var, 
        'omega': [1, 1, 0.5, 0.5], 
        'transfer_param': [dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param],
        'alpha3': alpha3, 
        'sim_notes': train_sim_notes
    }
    
    return param

def train_agent_250k():
    """
    Train the pathway agent for 250,000 episodes.
    
    Returns:
        tuple: (sim_data, tf_m, param)
    """
    param = setup_training_parameters_250k()
    
    print(f"Starting training for {param['n_episodes']} episodes...")
    
    # Initialize agent
    tf_m = pathway_agents_v11(param)

    # Set transfer function parameters
    dp_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]
    ip_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]
    dp_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]
    ip_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]

    tf_m.set_nl_td_parameters(dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param)
    tf_m.env.jump_correction = 0
    tf_m.env.generate_environment()
    tf_m.n_blocked_states = 5
    tf_m.training_episode_bias = 0.5
    tf_m.bias_episode_ratio = 0.5

    # Plot environment trajectories
    fsize = [5, 5]
    tf_m.env.plot_env_trajectories(fsize, tf_m.env.z_vec)

    # Update transfer function parameters for training
    dp_dls_param = [11.5, 1, 2, 1, -2, 0]
    ip_dls_param = [11.5, 1, 2, 1, -2, 0]
    dp_dms_param = [11.5, 1, 2, 1, -2, 0]
    ip_dms_param = [11.5, 1, 2, 1, -2, 0]

    tf_m.set_nl_td_parameters(dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param)
    
    # Plot transfer function
    tf_fig2 = tf_m.plot_transfer_function([5, 3])

    # Train the agent with force retrain
    print("Training in progress...")
    sim_data = tf_m.train_agent()
    
    # Save training data
    training_data = {
        'sim_data': sim_data,
        'tf_m': tf_m,
        'param': param
    }
    save_sim_data(training_data, "training_data_250k")
    
    print("Training completed and data saved!")
    return sim_data, tf_m, param

def plot_training_results_250k(sim_data, param, tf_m, save_plots=True):
    """
    Plot training results for 250k training.
    
    Args:
        sim_data: Simulation data from training
        param: Training parameters
        tf_m: Trained agent
        save_plots: Whether to save plots to files
        
    Returns:
        matplotlib.figure.Figure: Training results figure
    """
    f_size = [12, 10]
    init_f = 0.7
    n_bins = 15
    
    tf_a = pathway_analysis_v6(sim_data, param)
    f1 = tf_a.grid_plot_training_2C4A(f_size, '2C3A', sim_data, param, tf_m, init_f, n_bins)
    
    if save_plots:
        f1.savefig('plots/training_results.png', bbox_inches='tight', dpi=300)
        print("Training results saved to plots/training_results.png")
    
    return f1

def plot_neural_activity_250k(sim_data, tf_m, save_plots=True):
    """
    Plot neural activity during 250k training.
    
    Args:
        sim_data: Simulation data from training
        tf_m: Trained agent
        save_plots: Whether to save plots to files
        
    Returns:
        matplotlib.figure.Figure: Neural activity figure
    """
    n_timepoints = 13
    fig_size = [8, 7]
    
    tf_a = pathway_analysis_v6(sim_data, {})  # Empty param dict for this analysis
    f2 = tf_a.grid_plot_neural_activity_DMS_DLS(fig_size, sim_data, n_timepoints, tf_m)
    
    if save_plots:
        f2.savefig('plots/neural_activity.png', bbox_inches='tight', dpi=300)
        print("Neural activity plot saved to plots/neural_activity.png")
    
    return f2

if __name__ == "__main__":
    # Run 250k training
    print("="*60)
    print("TRAINING PATHWAY AGENT FOR 250,000 EPISODES")
    print("="*60)
    
    # Train agent
    sim_data, tf_m, param = train_agent_250k()
    
    # Plot results
    print("\nGenerating plots...")
    f1 = plot_training_results_250k(sim_data, param, tf_m)
    f2 = plot_neural_activity_250k(sim_data, tf_m)
    
    plt.show()
    print("\n" + "="*60)
    print("250K TRAINING COMPLETED SUCCESSFULLY!")
    print("Plots saved to:")
    print("- plots/training_results.png")
    print("- plots/neural_activity.png")
    print("="*60)
