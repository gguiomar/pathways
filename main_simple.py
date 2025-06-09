#%%

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# Import refactored modules

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

#%% TRAINING
print("=== TRAINING PHASE ===")

# Train agent (automatically saves/loads data)
sim_data, tf_m, param = train_agent()

# Plot training results
f1 = plot_training_results(sim_data, param, tf_m, save_plots=True)
f2 = plot_neural_activity(sim_data, tf_m, save_plots=True)

plt.show()

#%% CONTROL TESTING
print("=== CONTROL TESTING ===")

# Run control test (automatically saves/loads data)
test_data = run_control_test(sim_data, tf_m, param)

# Plot control test results and get metrics
f_control = plot_control_test_results(test_data, param, tf_m, save_plots=True)
test_metrics = get_test_metrics(test_data, tf_m)

print(f'Performance: {test_metrics["performance"]:.3f}')
print(f'P(Break): {test_metrics["p_break"]:.3f}')

plt.show()

#%% OPTOGENETICS - DLS PERTURBATIONS
print("=== DLS PERTURBATION EXPERIMENTS ===")

# Run DLS perturbation experiments (automatically saves/loads data)
dls_results = run_dls_perturbation_experiments(sim_data, tf_m, param, test_data)

# Plot DLS results
dls_figures = plot_dls_perturbation_results(dls_results, param, tf_m, save_plots=True)

print("DLS perturbation metrics:")
print("BL:", dls_results['metric_a2a_bl'])
print("CS:", dls_results['metric_a2a_cs']) 
print("CL:", dls_results['metric_a2a_cl'])

plt.show()

#%% OPTOGENETICS - DMS PERTURBATIONS
print("=== DMS PERTURBATION EXPERIMENTS ===")

# Run DMS perturbation experiments (automatically saves/loads data)
dms_results = run_dms_perturbation_experiments(sim_data, tf_m, param, test_data)

# Plot DMS results
dms_figures = plot_dms_perturbation_results(dms_results, param, tf_m, save_plots=True)

print("DMS perturbation metrics:")
print("BL:", dms_results['metric_dm_bl'])
print("CS:", dms_results['metric_dm_cs'])
print("CL:", dms_results['metric_dm_cl'])

plt.show()

#%% PERCEPTUAL POLICY ANALYSIS
print("=== PERCEPTUAL POLICY ANALYSIS ===")

# Run perceptual policy analysis (automatically saves/loads data)
ppolicy_results = run_perceptual_policy_analysis(sim_data, tf_m, param)

# Plot all perceptual policy results
ppolicy_figures = plot_perceptual_policy_results(ppolicy_results, save_plots=True)

# Print comprehensive analysis summary
print_analysis_summary(ppolicy_results)

plt.show()

#%% SUMMARY
print("=== ANALYSIS COMPLETE ===")
print(f"Training Performance: {test_metrics['performance']:.3f}")
print(f"Breaking Probability: {test_metrics['p_break']:.3f}")
print(f"All plots saved to: plots/")
print(f"All data saved to: simulation_data/")

# Display key results
print("\nKey Results Summary:")
print("-" * 40)
print(f"Training episodes: {param['n_episodes']}")
print(f"Test episodes: {test_data['n_episodes']}")
print(f"DLS perturbation levels: {dls_results['pert_list_a2a']}")
print(f"DMS perturbation levels: {dms_results['pert_list_dm']}")
print(f"Perceptual policy episodes: {ppolicy_results['tf_m_params']['n_test_episodes']}")

#%% FORCE RERUN EXAMPLES (Optional)
# Uncomment and run these cells to force recomputation

# # Force retrain agent
# sim_data, tf_m, param = train_agent(force_retrain=True)

# # Force retest
# test_data = run_control_test(sim_data, tf_m, param, force_retest=True)

# # Force rerun optogenetics
# dls_results = run_dls_perturbation_experiments(sim_data, tf_m, param, test_data, force_rerun=True)

# # Force rerun perceptual policy
# ppolicy_results = run_perceptual_policy_analysis(sim_data, tf_m, param, force_rerun=True)

#%% INDIVIDUAL PLOT ACCESS
# Access individual figures for customization

# Training plots
# f1 = training results grid plot
# f2 = neural activity plot

# Control test plot
# f_control = control test results

# DLS perturbation plots
# dls_figures = list of DLS perturbation figures

# DMS perturbation plots  
# dms_figures = list of DMS perturbation figures

# Perceptual policy plots
# ppolicy_figures['transition_matrix'] = transition matrix plot
# ppolicy_figures['transitions_all'] = all transitions plot
# ppolicy_figures['nmf_decomposition'] = NMF decomposition
# ppolicy_figures['state_probabilities'] = state probabilities
# ppolicy_figures['grid_plot'] = comprehensive grid plot

print("All figures available for further customization!")

#%%
