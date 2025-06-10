# Simulation Code for the Paper: Action suppression reveals opponent parallel control via striatal circuits

## Overview

This repository contains a version of the pathways analysis codebase from the paper:

```bash
Action suppression reveals opponent parallel control via striatal circuits
BF Cruz, G Guiomar, S Soares, A Motiwala, CK Machens, JJ Paton
Nature 607 (7919), 521-526
```

## Installation

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/gguiomar/pathways/tree/main
   cd pathways
   ```

2. **Create conda environment from environment.yml**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate pathways
   ```

### Quick Start

After installation, you can run the analysis in several ways:

**Option 1: Interactive VSCode cells (Recommended)**
```bash
# Open main_simple_refactored.py in VSCode
# Run cells individually using Ctrl+Enter
```

**Option 2: Command line**
```bash
python main.py              # Full pipeline
python main.py training     # Training only
python main.py analysis     # Analysis only
```

**Option 3: Jupyter notebook**
```bash
jupyter lab
# Open and run main_simple_refactored.py as notebook
```

## Structure

```
pathways/
├── main.py                    # Main pipeline orchestrator
├── training.py               # Training functionality
├── test.py                   # Test dataset creation (CONTROL)
├── optogenetics.py          # Optogenetics experiments (DLS/DMS)
├── perceptualpolicy.py      # Perceptual policy analysis
├── ppolicy_utils.py         # Perceptual policy utility functions
├── utils.py                 # Common utilities (pickle save/load, etc.)
├── plots/                   # All generated figures
├── simulation_data/         # Saved simulation data (pickle files)
├── agents/
│   └── pathway_agents.py
├── analysis/
│   ├── __init__.py
│   ├── pathway_analysis.py  # Core analysis functions
│   └── utils.py
└── environments/
    └── timing_task_csc.py
```

## Usage

### Running the Full Pipeline

```bash
# Run complete analysis pipeline
python main.py

# Or explicitly specify full mode
python main.py full
```

### Running Individual Components

```bash
# Training only
python main.py training

# Analysis only (requires existing training data)
python main.py analysis

# Individual modules
python training.py
python test.py
python optogenetics.py
python perceptualpolicy.py
```

### Programmatic Usage

```python
from training import train_agent
from testing_control import run_control_test
from optogenetics import run_dls_perturbation_experiments
from perceptualpolicy import run_perceptual_policy_analysis

# Train agent
sim_data, tf_m, param = train_agent()

# Run control tests
test_data = run_control_test(sim_data, tf_m, param)

# Run optogenetics experiments
dls_results = run_dls_perturbation_experiments(sim_data, tf_m, param, test_data)

# Run perceptual policy analysis
ppolicy_results = run_perceptual_policy_analysis(sim_data, tf_m, param)
```

## Data Management

### Automatic Save/Load
All modules implement automatic data persistence:

```python
# Check if data exists, load if available
if not force_rerun and check_data_exists("training_data"):
    saved_data = load_sim_data("training_data")
    return saved_data

# Save results after computation
save_sim_data(results, "training_data")
```

### Data Files
- `training_data.pkl` - Training simulation results
- `control_test_data.pkl` - Control test results
- `dls_perturbation_data.pkl` - DLS perturbation results
- `dms_perturbation_data.pkl` - DMS perturbation results
- `perceptual_policy_data.pkl` - Perceptual policy analysis results
- `pipeline_summary.pkl` - Complete pipeline summary

## Configuration

### Training Parameters
Default parameters can be modified in `training.py`:

```python
def setup_training_parameters():
    n_eps = 150000  # Training episodes
    n_test_eps = int(n_eps * 0.7)  # Test episodes
    # ... other parameters
```

### Force Rerun Options
Each module supports force rerun flags:

```python
# Force retraining
train_agent(force_retrain=True)

# Force retesting
run_control_test(sim_data, tf_m, param, force_retest=True)

# Force rerun optogenetics
run_dls_perturbation_experiments(..., force_rerun=True)
```
