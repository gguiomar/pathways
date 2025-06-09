"""
Perceptual policy module for pathway agents.
Contains organized code for perceptual policy analysis and generates grid plots.
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
from ppolicy_utils import (
    test_agent_act, 
    plot_transition_matrix, 
    plot_transitions_all,
    perform_nmf_decomposition,
    plot_state_probabilities,
    create_perceptual_policy_grid_plot,
    analyze_successor_representation
)
from utils import save_sim_data, load_sim_data, check_data_exists

def setup_perceptual_policy_parameters(tf_m, param):
    """
    Set up parameters for perceptual policy analysis.
    
    Args:
        tf_m: Trained agent
        param: Parameters
        
    Returns:
        tuple: (tf_m, Act_Opt)
    """
    tf_m.n_test_episodes = 10000
    tf_m.perturbation_type = False
    tf_m.trial_subset = [3, 5, 6, 7, 8, 9, 10, 11, 12]  # tone subset not trial subset
    
    tone_list = [param['second_tone_list'][i] for i in tf_m.trial_subset]
    print("Tone list:", tone_list)
    tf_m.obs_var1 = 0
    
    return tf_m

def run_perceptual_policy_analysis(sim_data, tf_m, param, force_rerun=False):
    """
    Run perceptual policy analysis.
    
    Args:
        sim_data: Training simulation data
        tf_m: Trained agent
        param: Parameters
        force_rerun: If True, forces rerunning even if saved data exists
        
    Returns:
        dict: Perceptual policy analysis results
    """
    # Check if perceptual policy data already exists
    if not force_rerun and check_data_exists("perceptual_policy_data"):
        print("Loading existing perceptual policy data...")
        saved_data = load_sim_data("perceptual_policy_data")
        if saved_data is not None:
            return saved_data
    
    print("Running perceptual policy analysis...")
    
    # Setup parameters
    tf_m = setup_perceptual_policy_parameters(tf_m, param)
    
    # Create action preferences matrix
    A_DLS = copy.copy(sim_data['A'])
    Act_Opt = A_DLS[0, :, :] - A_DLS[1, :, :]
    
    print("Testing agent with action preferences...")
    test_data_act = test_agent_act(tf_m, Act_Opt)
    
    # Extract transition matrix
    T_test = test_data_act['T']
    M1_act = test_data_act['M1']
    
    # Store results
    results = {
        'test_data_act': test_data_act,
        'T_test': T_test,
        'M1_act': M1_act,
        'Act_Opt': Act_Opt,
        'tf_m_params': {
            'n_test_episodes': tf_m.n_test_episodes,
            'trial_subset': tf_m.trial_subset,
            'obs_var1': tf_m.obs_var1
        }
    }
    
    # Save results
    save_sim_data(results, "perceptual_policy_data")
    
    return results

def plot_perceptual_policy_results(ppolicy_results, save_plots=True):
    """
    Plot all perceptual policy analysis results.
    
    Args:
        ppolicy_results: Perceptual policy analysis results
        save_plots: Whether to save plots to files
        
    Returns:
        dict: Dictionary of all generated figures
    """
    T_test = ppolicy_results['T_test']
    M1_act = ppolicy_results['M1_act']
    
    figures = {}
    
    print("Generating perceptual policy plots...")
    
    # Individual plots
    print("1. Transition matrix...")
    figures['transition_matrix'] = plot_transition_matrix(T_test, save_plots)
    
    print("2. All transitions...")
    figures['transitions_all'], T_conv = plot_transitions_all(T_test, save_plots)
    
    print("3. NMF decomposition...")
    figures['nmf_decomposition'], W, H = perform_nmf_decomposition(T_conv, save_plots=save_plots)
    
    print("4. State probabilities...")
    figures['state_probabilities'] = plot_state_probabilities(T_test, save_plots)
    
    print("5. Successor representation...")
    figures['successor_representation'] = analyze_successor_representation(M1_act, save_plots)
    
    print("6. Comprehensive grid plot...")
    figures['grid_plot'] = create_perceptual_policy_grid_plot(T_test, save_plots)
    
    if save_plots:
        print("All perceptual policy plots saved to plots/")
    
    return figures

def analyze_transition_dynamics(ppolicy_results):
    """
    Analyze transition dynamics from perceptual policy results.
    
    Args:
        ppolicy_results: Perceptual policy analysis results
        
    Returns:
        dict: Analysis metrics
    """
    T_test = ppolicy_results['T_test']
    
    # Calculate various metrics
    pre_tone_states = T_test[0:40, :]
    post_tone_predictions = T_test[0:40, 40:80]
    
    # Transition entropy
    def calculate_entropy(prob_matrix):
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        prob_matrix = prob_matrix + eps
        prob_matrix = prob_matrix / np.sum(prob_matrix, axis=1, keepdims=True)
        entropy = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
        return entropy
    
    pre_tone_entropy = calculate_entropy(pre_tone_states)
    post_tone_entropy = calculate_entropy(post_tone_predictions)
    
    # Transition sharpness (inverse of entropy)
    pre_tone_sharpness = 1.0 / (pre_tone_entropy + 1e-10)
    post_tone_sharpness = 1.0 / (post_tone_entropy + 1e-10)
    
    metrics = {
        'pre_tone_entropy': pre_tone_entropy,
        'post_tone_entropy': post_tone_entropy,
        'pre_tone_sharpness': pre_tone_sharpness,
        'post_tone_sharpness': post_tone_sharpness,
        'mean_pre_tone_entropy': np.mean(pre_tone_entropy),
        'mean_post_tone_entropy': np.mean(post_tone_entropy),
        'mean_pre_tone_sharpness': np.mean(pre_tone_sharpness),
        'mean_post_tone_sharpness': np.mean(post_tone_sharpness)
    }
    
    return metrics

def print_analysis_summary(ppolicy_results):
    """
    Print a summary of the perceptual policy analysis.
    
    Args:
        ppolicy_results: Perceptual policy analysis results
    """
    print("\n" + "="*60)
    print("PERCEPTUAL POLICY ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic info
    test_data = ppolicy_results['test_data_act']
    print(f"Number of test episodes: {test_data['n_episodes']}")
    print(f"Transition matrix shape: {ppolicy_results['T_test'].shape}")
    print(f"Successor representation shape: {ppolicy_results['M1_act'].shape}")
    
    # Analyze transition dynamics
    metrics = analyze_transition_dynamics(ppolicy_results)
    print(f"\nTransition Dynamics:")
    print(f"  Mean pre-tone entropy: {metrics['mean_pre_tone_entropy']:.3f}")
    print(f"  Mean post-tone entropy: {metrics['mean_post_tone_entropy']:.3f}")
    print(f"  Mean pre-tone sharpness: {metrics['mean_pre_tone_sharpness']:.3f}")
    print(f"  Mean post-tone sharpness: {metrics['mean_post_tone_sharpness']:.3f}")
    
    # Transition matrix statistics
    T_test = ppolicy_results['T_test']
    print(f"\nTransition Matrix Statistics:")
    print(f"  Min value: {np.min(T_test):.3f}")
    print(f"  Max value: {np.max(T_test):.3f}")
    print(f"  Mean value: {np.mean(T_test):.3f}")
    print(f"  Sparsity: {np.sum(T_test == 0) / T_test.size:.3f}")
    
    print("="*60)

if __name__ == "__main__":
    # This would typically be called from main.py, but can be run independently
    print("Perceptual policy module - should be called from main.py with trained agent and simulation data")
    print("To run independently, first train an agent using training.py")
