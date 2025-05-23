#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script applies the GPR partial permutation test to detect functional changes
on data from the data folder and saves results.

Usage:
    python gpr_experiment.py
"""

import os
import json
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
from gpr import partial_permutation_test  # Import our GPR implementation

# Set random seeds for reproducibility
np.random.seed(42)
cp.random.seed(42)

# Helper function to make JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)  # Convert Infinity/NaN to string
    elif isinstance(obj, (bool, np.bool_)):  # Handle boolean values explicitly
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_results_dir = f"results/gpr_{timestamp}"
if not os.path.exists(main_results_dir):
    os.makedirs(main_results_dir)

print(f"Results will be saved to {main_results_dir}/")

# Define data directory
data_dir = "data"

# Define node counts and noise types to match what's in the data folder
node_counts = [10, 20, 30, 40, 50]
noise_types = ["ER", "SF"]

# Process each node count
for node_count in node_counts:
    print(f"\n===== Processing experiments with {node_count} nodes =====")
    
    # Create results directory for this node count
    node_results_dir = f"{main_results_dir}/nodes_{node_count}"
    if not os.path.exists(node_results_dir):
        os.makedirs(node_results_dir)
    
    # Process each noise type
    for noise_type in noise_types:
        print(f"\nNoise Type: {noise_type}")
        
        # Create results directory for this noise type
        noise_results_dir = f"{node_results_dir}/{noise_type}"
        if not os.path.exists(noise_results_dir):
            os.makedirs(noise_results_dir)
        
        # Path to data for this node count and noise type
        data_path = f"{data_dir}/nodes_{node_count}/{noise_type}"
        
        # Check if this path exists
        if not os.path.exists(data_path):
            print(f"WARNING: Data path {data_path} does not exist. Skipping...")
            continue
        
        # Find all dataset indices (extract from filenames)

        print("Finding dataset indices...")
        dataset_indices = []
        for file in os.listdir(data_path):
            if file.startswith("metadata_") and file.endswith(".json"):
                idx = int(file.replace("metadata_", "").replace(".json", ""))
                dataset_indices.append(idx)
        
        dataset_indices.sort()
        
        # Process each dataset
        for dataset_idx in dataset_indices:
            print(f"\nProcessing dataset {dataset_idx}")
            
            # Load data and metadata
            data_env1 = np.load(f"{data_path}/data_env1_{dataset_idx}.npy")
            data_env2 = np.load(f"{data_path}/data_env2_{dataset_idx}.npy")
            adj = np.load(f"{data_path}/adj_{dataset_idx}.npy")
            
            with open(f"{data_path}/metadata_{dataset_idx}.json", 'r') as f:
                metadata = json.load(f)
            
            # Extract metadata
            shifted_nodes = metadata.get("shifted_nodes", [])
            shift_types = metadata.get("shift_types", {})

            # Separate function-shifted and noise-shifted nodes
            function_shifted_nodes = [node for node in shifted_nodes if shift_types.get(str(node)) == "function"]
            noise_shifted_nodes = [node for node in shifted_nodes if shift_types.get(str(node)).startswith("noise_")]

            print(f"Dataset info - Nodes: {node_count}, Graph: {noise_type}")
            print(f"Function-shifted nodes: {function_shifted_nodes}")
            print(f"Noise-shifted nodes: {noise_shifted_nodes}")
                
            # Result file name pattern
            result_filename = f"gpr_nodes{node_count}_{noise_type}_{dataset_idx}"
            
            # Initialize results dictionary
            results = {
                "dataset_info": metadata,
                "model": "gpr"
            }

            # Create pandas DataFrames from numpy arrays (needed for partial_permutation_test)
            columns = [f"X{i}" for i in range(data_env1.shape[1])]
            env1_df = pd.DataFrame(data_env1, columns=columns)
            env2_df = pd.DataFrame(data_env2, columns=columns)
            
            try:
                print("Running GPR partial permutation test for function-shifted and noise-shifted nodes...")
                
                # Combine function-shifted and noise-shifted nodes
                all_shifted_nodes = set(function_shifted_nodes + noise_shifted_nodes)
                
                # Initialize a dictionary to store estimated shift types
                estimated_shift_types = {}

                # Process each shifted node
                for node in all_shifted_nodes:
                    print(f"Testing node {node}...")
                    
                    # Get parents of the node
                    parent_indices = np.where(adj[:, node] == 1)[0]
                    
                    if len(parent_indices) == 0:
                        print(f"Node {node} has no parents. Skipping...")
                        continue
                    
                    # Define columns for the node and its parents
                    node_col = columns[node]
                    parent_cols = [columns[parent] for parent in parent_indices]
                    
                    # Subset data for the node and its parents
                    env1_subset = env1_df[[node_col] + parent_cols]
                    env2_subset = env2_df[[node_col] + parent_cols]

                    # Subset DAG for the node and its parents
                    adj_subset = adj[np.ix_([node] + list(parent_indices), [node] + list(parent_indices))]
                    
                    # Run GPR test
                    test_params = {
                        "num_permutations": 500,
                        "alpha": 0.05,
                        "gamma": 0.1,
                        "length_scale_range": (0.1, 10.0),
                        "test_statistic_type": 'pseudo',
                        "verbose": False
                    }
                    
                    gpr_results = partial_permutation_test(
                        env1_data=env1_subset,
                        env2_data=env2_subset,
                        y_node_col=node_col,
                        dag_adjacency_matrix=adj_subset,
                        **test_params
                    )
                    
                    # Store results for this node
                    if gpr_results["p_value"] < test_params["alpha"]:
                        estimated_shift_types[str(node)] = "function"
                    else:
                        estimated_shift_types[str(node)] = "noise"  # No significant shift detected

                # Add estimated shift types to results
                results["estimated_shift_types"] = estimated_shift_types

            except Exception as e:
                print(f"Error in GPR test: {str(e)}")
                results["error"] = str(e)

            # Save results for this dataset
            with open(f"{noise_results_dir}/{result_filename}.json", 'w') as f:
                serializable_results = make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {noise_results_dir}/{result_filename}.json")

print(f"\n===== All experiments completed =====")
print(f"Results saved to {main_results_dir}/") 