"""
Utility functions for pathway analysis
Contains basic helper functions for data processing and normalization
"""

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

def calc_perf(sim_data):
    """Calculates the performance of the agent on a single experiment."""
    choices = sim_data['choices']
    return sum(choices)/choices.shape[0]

def normalise_vec(vec):
    """Normalises a vector to range [0,1]"""
    return (vec-min(vec))/(max(vec)-min(vec))

def min_max_norm(vec):
    """Min-max normalization with small epsilon to avoid division by zero"""
    return (vec-np.min(vec))/(np.max(vec)-np.min(vec)+0.000001)

def psychometric_function(x, alpha, beta):
    """Standard psychometric function (sigmoid)"""
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

def fit_psychometric_curve(t_vec, psych_long):
    """Fit psychometric curve to behavioral data"""
    time_v = np.linspace(0.6, 2.4, 100)
    par0 = np.array([0.01, 1.])  # initial conditions
    
    # Convert inputs to numpy arrays
    t_vec = np.asarray(t_vec)
    psych_long = np.asarray(psych_long)
    
    # Remove NaN and inf values
    valid_mask = np.isfinite(psych_long) & np.isfinite(t_vec)
    if np.sum(valid_mask) < 2:  # Need at least 2 points to fit
        # Return default parameters if not enough valid data
        return time_v, par0, np.eye(2)
    
    t_vec_clean = t_vec[valid_mask]
    psych_long_clean = psych_long[valid_mask]
    
    try:
        par, mcov = curve_fit(psychometric_function, t_vec_clean, psych_long_clean, par0)
    except (RuntimeError, ValueError):
        # If fitting fails, return default parameters
        par = par0
        mcov = np.eye(2)
    
    return time_v, par, mcov

def conv_gauss(arr, sigma):
    """Apply Gaussian convolution to smooth data"""
    size = int(2 * np.ceil(2 * sigma) + 1)
    x = np.linspace(-size / 2, size / 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    kernel /= np.sum(kernel)
    convolved = np.convolve(arr, kernel, mode='same')
    return convolved

def get_performance_metrics(sim_data, tf_m):
    """Calculate basic performance metrics"""
    # Performance
    corr_st = np.sum((sim_data['choice_v'] == 2).astype(int), axis=0)
    corr_all = np.sum(corr_st)
    inc_st = np.sum((sim_data['choice_v'] == 3).astype(int), axis=0)
    inc_all = np.sum(inc_st)
    perf = corr_all/(corr_all+inc_all)
    
    # Breaking fixation probability
    bf_left = (sim_data['behaviour_h'][:,0:tf_m.n_states] == 0).astype(int)
    bf_right = (sim_data['behaviour_h'][:,0:tf_m.n_states] == 1).astype(int)
    total_bf_m = bf_left + bf_right
    total_bf_st = np.sum(total_bf_m, axis=0)
    p_break = np.sum(total_bf_st)/total_bf_m.shape[0]
    
    return {'performance': perf, 'p_break': p_break}
