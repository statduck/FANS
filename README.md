# CCP Experiment Scripts

This directory contains scripts for running experiments with the DataGenerator class to analyze function and noise shifts between environments.

## Scripts

- `experiment_script.py`: The main script that runs multiple experiments and saves results
- `check_functions.py`: A helper script to check if all required functions exist
- `iscan_utils.py`: Implementation of the `f_shifted_spline` function for visualization

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- igraph (for DAG generation)
- scikit-learn (for data scaling)

## Usage

### Running the Check Functions Script

First, check if all required functions exist:

```bash
python check_functions.py
```

### Running Experiments

Run the main experiment script:

```bash
python experiment_script.py
```

This script:
1. Generates data with DataGenerator, using a different seed for each experiment 
2. Trains models on both environments using fit_to_sem
3. Compares likelihood between environments and saves node indices
4. Refits model1 to environment 2 and saves it as model1_refitted
5. Calculates JS divergences between different model fits
6. Runs ISCAN's est_node_shifts function to detect shifted nodes
7. Generates visualization of function shifts using f_shifted_spline
8. Saves all results to the experiment_results directory

### Output Files

Results are saved to the `experiment_results` directory with the following files:

- `dag_*_timestamp.png`: DAG visualization for each experiment
- `f_shifted_spline_*_timestamp.png`: Visualization of function shifts for Y node
- `f_shifted_diff_*_timestamp.png`: Visualization of differences between spline fits
- `experiment_*_timestamp.json`: Individual experiment results
- `all_results_timestamp.json`: Aggregated results from all experiments
- `summary_stats_timestamp.json`: Summary statistics of detection accuracy and JS divergences

## Customization

You can modify the following parameters in `experiment_script.py`:
- `num_experiments`: Number of experiments to run
- DataGenerator parameters (d, s0, graph_type, etc.)
- Sample size (n)
- Number of shifted nodes (base_num_shifted_nodes)

## Additional Notes

- The Y node (node with the most parents) can have function shifts or noise shifts
- Other nodes can only have noise shifts
- Detection accuracy is measured by comparing the detected shifted nodes with the true shifted nodes 