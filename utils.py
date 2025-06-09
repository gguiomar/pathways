"""
Common utility functions for the pathways project.
Includes pickle save/load functionality and other shared utilities.
"""

import pickle
import os
from pathlib import Path
import numpy as np

def save_sim_data(data, filename, data_dir="simulation_data"):
    """
    Save simulation data to pickle file.
    
    Args:
        data: Data to save
        filename: Name of the file (without extension)
        data_dir: Directory to save data in
    """
    Path(data_dir).mkdir(exist_ok=True)
    filepath = os.path.join(data_dir, f"{filename}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")

def load_sim_data(filename, data_dir="simulation_data"):
    """
    Load simulation data from pickle file.
    
    Args:
        filename: Name of the file (without extension)
        data_dir: Directory to load data from
        
    Returns:
        Loaded data or None if file doesn't exist
    """
    filepath = os.path.join(data_dir, f"{filename}.pkl")
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filepath}")
        return data
    else:
        print(f"File {filepath} not found")
        return None

def check_data_exists(filename, data_dir="simulation_data"):
    """
    Check if simulation data file exists.
    
    Args:
        filename: Name of the file (without extension)
        data_dir: Directory to check in
        
    Returns:
        bool: True if file exists, False otherwise
    """
    filepath = os.path.join(data_dir, f"{filename}.pkl")
    return os.path.exists(filepath)

def conv_gauss(arr, sigma):
    """
    Convolve array with Gaussian kernel.
    
    Args:
        arr: Input array
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Convolved array
    """
    size = int(2 * np.ceil(2 * sigma) + 1)
    x = np.linspace(-size / 2, size / 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    kernel /= np.sum(kernel)
    convolved = np.convolve(arr, kernel, mode='same')
    return convolved
