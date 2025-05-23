#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates synthetic data using the updated DataGenerator class and saves it to a data folder.
Uses random ER or SF graphs and applies different types of shifts to selected nodes.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datagenerator import DataGenerator, set_seed
import argparse
import random
import igraph as ig

# Constants
DATA_DIR = "data"
DATASETS_PER_CONFIG = 30  # 30 samples per configuration
NODE_COUNTS = [10, 20, 30, 40, 50]  # Node counts
GRAPH_TYPES = ["ER", "SF"]  # Graph types

# Shift cases
SHIFT_CASES = [
    DataGenerator.SHIFT_CASE1,  # function, noise_std
    DataGenerator.SHIFT_CASE2,  # function, noise_skew
    DataGenerator.SHIFT_CASE3,  # function, function
    DataGenerator.SHIFT_CASE4,  # noise_std, noise_skew
    DataGenerator.SHIFT_CASE5,  # noise_std, noise_std
    DataGenerator.SHIFT_CASE6,  # noise_skew, noise_skew
]

def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def create_directories():
    """Create the necessary directories for data storage."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def create_dataset_paths(node_count, graph_type):
    """Create directory paths for dataset storage."""
    node_data_dir = f"{DATA_DIR}/nodes_{node_count}"
    if not os.path.exists(node_data_dir):
        os.makedirs(node_data_dir)
    
    graph_data_dir = f"{node_data_dir}/{graph_type}"
    if not os.path.exists(graph_data_dir):
        os.makedirs(graph_data_dir)
    
    return graph_data_dir

def plot_dag_with_shifts(gen, filename):
    """
    Create and save a visualization of the DAG with shifted nodes highlighted.
    
    Args:
        gen: DataGenerator instance
        filename: Output path for the plot
    """
    adj = gen.adj_env1.copy()
    g = ig.Graph.Adjacency(adj.tolist(), mode="directed")
    layout = g.layout("kk")  # Kamada-Kawai layout

    # Define colors for different types of shifts
    color_no_shift = "lightblue"
    color_func_shift = "red"
    color_skew_shift = "green"
    color_var_shift = "orange"

    # Assign colors and labels to nodes
    colors = []
    labels = []
    
    for i in range(gen.d):
        if i in gen.func_shift_nodes:
            c = color_func_shift
        elif i in gen.skew_shift_nodes:
            c = color_skew_shift
        elif i in gen.variance_shift_nodes:
            c = color_var_shift
        else:
            c = color_no_shift
            
        lbl = str(i)
        colors.append(c)
        labels.append(lbl)

    g.vs["color"] = colors
    g.vs["label"] = labels
    g.vs["size"] = 30

    ig.plot(g,
           target=filename,
           layout=layout,
           bbox=(600, 400),
           margin=50,
           vertex_label_color="black")

def save_dataset_files(gen, data_env1, data_env2, config_data_dir, dataset_idx, seed):
    """Save dataset files including data, plots, and metadata."""
    # Save DAG image
    dag_filename = f"{config_data_dir}/dag_{dataset_idx}.png"
    plot_dag_with_shifts(gen, filename=dag_filename)
    
    # Save data arrays and adjacency matrix
    np.save(f"{config_data_dir}/data_env1_{dataset_idx}.npy", data_env1)
    np.save(f"{config_data_dir}/data_env2_{dataset_idx}.npy", data_env2)
    np.save(f"{config_data_dir}/adj_{dataset_idx}.npy", gen.adj_env1)
    
    # Create shift type mappings
    shift_types = {}
    for node in gen.shifted_nodes:
        if node in gen.func_shift_nodes:
            shift_types[int(node)] = "function"
        elif node in gen.skew_shift_nodes:
            shift_types[int(node)] = "noise_skew"
        elif node in gen.variance_shift_nodes:
            shift_types[int(node)] = "noise_std"
    
    # Create metadata
    metadata = {
        "seed": seed,
        "dataset_index": dataset_idx,
        "node_count": gen.d,
        "edge_count": gen.s0,
        "graph_type": gen.graph_type,
        "shifted_nodes": gen.shifted_nodes,
        "shift_types": shift_types,
        "shift_case": gen.shift_case
    }
    
    # Save metadata with custom serializer
    with open(f"{config_data_dir}/metadata_{dataset_idx}.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializer)

def print_generation_summary():
    """Print summary information about the data generation process."""
    print("\n===== Data generation completed =====")
    print(f"All datasets saved to {DATA_DIR}/")
    print(f"Generated data for node counts: {NODE_COUNTS}")
    print(f"Generated data for graph types: {GRAPH_TYPES}")
    print(f"Generated {DATASETS_PER_CONFIG} datasets for each configuration")
    print(f"Used shift cases: {SHIFT_CASES}")
    total_datasets = len(NODE_COUNTS) * len(GRAPH_TYPES) * DATASETS_PER_CONFIG
    print(f"Total datasets generated: {total_datasets}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Base seed for data generation')
    return parser.parse_args()

def main():
    """Main function to generate datasets."""
    args = parse_args()
    base_seed = args.seed
    
    # Create base directories
    create_directories()
    
    # Generate data for each node count and graph type
    total_configs = len(NODE_COUNTS) * len(GRAPH_TYPES)
    total_datasets = total_configs * DATASETS_PER_CONFIG
    dataset_counter = 0
    
    for node_count in NODE_COUNTS:
        print(f"\n===== Generating data with {node_count} nodes =====")
        
        for graph_type in GRAPH_TYPES:
            print(f"\nGraph Type: {graph_type}")
            config_dir = create_dataset_paths(node_count, graph_type)
            
            for dataset_idx in range(1, DATASETS_PER_CONFIG + 1):  # 1부터 시작
                dataset_seed = base_seed + dataset_counter
                dataset_counter += 1
                
                # Calculate which shift case to use (cyclically)
                case_idx = (dataset_idx - 1) % len(SHIFT_CASES)
                shift_case = SHIFT_CASES[case_idx]
                
                progress = (dataset_counter / total_datasets) * 100
                print(f"Dataset {dataset_idx}/{DATASETS_PER_CONFIG} for {graph_type} ({node_count} nodes) "
                      f"- Shift Case: {shift_case} - Progress: {dataset_counter}/{total_datasets} ({progress:.1f}%)")
                
                # Set seed for reproducibility
                set_seed(dataset_seed)
                
                # Define number of shifted nodes (20% of total nodes)
                num_shifted = max(2, int(node_count * 0.2))
                # Ensure even number of shifted nodes for clean splitting
                if num_shifted % 2 != 0:
                    num_shifted += 1
                
                # Create generator instance with specified shift case
                gen = DataGenerator(
                    d=node_count,
                    s0=node_count,  # Use node count for edge count
                    graph_type=graph_type,
                    noise_std=0.5,
                    shift_case=shift_case
                )
                
                # Generate data with specified number of shifted nodes
                data_env1, data_env2 = gen.sample(n=1000, num_shifted_nodes=num_shifted)
                
                # Save dataset files
                save_dataset_files(gen, data_env1, data_env2, config_dir, dataset_idx, dataset_seed)
    
    # Print summary
    print_generation_summary()

if __name__ == "__main__":
    main() 