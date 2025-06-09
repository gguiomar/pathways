# Pathway Analysis Code - Extracted and Modularized

This repository contains the extracted and modularized code from the Jupyter notebook `advantage_updating_v49-finding_DMS_latent.ipynb`. The code has been reorganized into a clean, modular structure that is easier to read, maintain, and extend.

## Project Structure

```
pathways_extracted/
├── README.md                    # This file
├── main.py                      # Main demonstration script
├── agents/                      # Agent implementations
│   └── pathway_agents.py        # Pathway-based RL agents
├── environments/                # Environment implementations
│   └── timing_task_csc.py       # Classical conditioning timing task
├── analysis/                    # Analysis modules (modular design)
│   ├── __init__.py             # Package initialization
│   ├── main_analysis.py        # Main analysis class (unified interface)
│   ├── behavioral_analysis.py  # Behavioral analysis functions
│   ├── neural_analysis.py      # Neural activity analysis functions
│   ├── plotting.py             # Plotting utilities and visualizations
│   └── utils.py                # Utility functions and helpers
└── methods/                     # Additional methods and documentation
    ├── perceptual_policy.tex    # LaTeX documentation
    └── perceptual_policy.pdf    # Compiled PDF documentation
```

## Key Features

### Modular Design
The original monolithic analysis code has been split into focused modules:

- **`utils.py`**: Basic utility functions for data processing and normalization
- **`behavioral_analysis.py`**: Functions for analyzing choice behavior, breaking fixation, and psychometric curves
- **`neural_analysis.py`**: Functions for analyzing neural activity, action preferences, and pathway contributions
- **`plotting.py`**: Visualization utilities and plotting functions
- **`main_analysis.py`**: Main analysis class that combines all functionality

### Clean Interface
The `PathwayAnalysis` class provides a unified interface that replaces the original `pathway_analysis_v6` class:

```python
from analysis import PathwayAnalysis

# Initialize analysis
analyzer = PathwayAnalysis(sim_data, param)

# Run comprehensive analysis
results = analyzer.run_full_analysis(tf_m, save_plots=True)

# Access specialized analyzers
behavioral_data = analyzer.behavioral.get_psychometric_data(...)
neural_data = analyzer.neural.sample_action_preferences_DLS(...)
plots = analyzer.plotting.plot_psychometric_curve(...)
```

### Descriptive Names
Functions and classes now have clear, descriptive names that indicate their purpose:

- `BehavioralAnalyzer` for behavioral analysis
- `NeuralAnalyzer` for neural activity analysis
- `PlottingUtils` for visualization
- `get_psychometric_data()` instead of generic function names
- `sample_action_preferences_DLS()` for specific pathway analysis

## Usage

### Quick Start

Run the demonstration script to see the modular code in action:

```bash
python main.py
```

This will:
1. Set up simulation parameters
2. Run a pathway agent simulation
3. Analyze results using the modular tools
4. Generate comprehensive plots
5. Save results and individual visualizations

### Basic Usage

```python
import numpy as np
from agents.pathway_agents import pathway_agents_v11
from analysis import PathwayAnalysis

# Set up parameters
param = {
    'n_states': 40,
    'beta': 1.5,
    'gamma_v': 0.98,
    'gamma_dm': 0.98,
    'n_episodes': 10000,
    # ... other parameters
}

# Run simulation
agent = pathway_agents_v11(param)
sim_data = agent.train_agent()

# Analyze results
analyzer = PathwayAnalysis(sim_data, param)
results = analyzer.run_full_analysis(agent)

# Access specific analyses
performance = analyzer.get_performance_metrics(agent)
pathway_contrib = analyzer.analyze_pathway_contributions(sim_data)
```

### Individual Analysis Components

```python
# Behavioral analysis
behavioral = analyzer.behavioral
psychometric_data = behavioral.get_psychometric_data(...)
breaking_fixation_data = behavioral.get_breaking_fixation_data(...)
hazard_data = behavioral.calculate_hazard(...)

# Neural analysis
neural = analyzer.neural
dls_activity = neural.sample_action_preferences_DLS(...)
dms_activity = neural.sample_action_preferences_DMS(...)
latent_states = neural.extract_latent_states(...)

# Plotting
plotting = analyzer.plotting
fig1 = plotting.plot_psychometric_curve(...)
fig2 = plotting.plot_breaking_fixations(...)
fig3 = plotting.plot_neural_activity_dms_dls(...)
```

## Key Improvements

### 1. Modularity
- Code is split into logical, focused modules
- Each module has a single responsibility
- Easy to test and maintain individual components

### 2. Readability
- Clear, descriptive function and class names
- Comprehensive docstrings
- Consistent code style and organization

### 3. Reusability
- Functions can be used independently
- Modular design allows for easy extension
- Clean interfaces between components

### 4. Maintainability
- Separation of concerns
- Reduced code duplication
- Clear dependency structure

### 5. Documentation
- Comprehensive README
- Inline documentation
- Example usage patterns

## Analysis Capabilities

### Behavioral Analysis
- Psychometric curve fitting
- Breaking fixation analysis
- Hazard function calculation
- Choice behavior analysis
- Real-time state mapping

### Neural Activity Analysis
- Action preference sampling for DLS and DMS pathways
- Pathway contribution analysis
- Latent state extraction (SVD, NMF)
- Convergence analysis
- Neural data export to CSV

### Visualization
- Comprehensive grid plots
- Individual analysis plots
- Neural activity visualizations
- Transfer function plots
- Convergence plots

### Data Management
- Simulation data saving/loading
- Parameter management
- CSV export capabilities
- Timestamped result directories

## Dependencies

The code requires the following Python packages:
- numpy
- matplotlib
- scipy
- pandas
- scikit-learn (for dimensionality reduction)

## Original Source

This code was extracted and modularized from:
- **Original file**: `pathways/advantage_updating_v49-finding_DMS_latent.ipynb`
- **Original analysis class**: `pathway_analysis_v6`
- **Original agent class**: `pathway_agents_v11`

## Future Extensions

The modular design makes it easy to:
- Add new analysis methods
- Implement additional plotting functions
- Extend behavioral or neural analysis capabilities
- Add new agent types or environments
- Integrate with other analysis pipelines

## Contact

For questions about the code structure or usage, refer to the original research documentation or the inline code comments.
