"""
Pathway Analysis Package

This package provides comprehensive analysis tools for pathway-based reinforcement learning models.
It includes modules for behavioral analysis, neural activity analysis, plotting utilities, and helper functions.

Modules:
- utils: Basic utility functions for data processing and normalization
- behavioral_analysis: Functions for analyzing choice behavior, breaking fixation, and psychometric curves
- neural_analysis: Functions for analyzing neural activity, action preferences, and pathway contributions
- plotting: Visualization utilities and plotting functions
- main_analysis: Main analysis class that combines all functionality
"""

from .utils import (
    calc_perf, 
    normalise_vec, 
    min_max_norm, 
    psychometric_function, 
    fit_psychometric_curve,
    conv_gauss,
    get_performance_metrics
)

__all__ = [
    'calc_perf',
    'normalise_vec', 
    'min_max_norm',
    'psychometric_function',
    'fit_psychometric_curve',
    'conv_gauss',
    'get_performance_metrics',
    'BehavioralAnalyzer',
    'NeuralAnalyzer', 
    'PlottingUtils',
    'PathwayAnalysis'
]

__version__ = '1.0.0'
__author__ = 'Pathway Analysis Team'
