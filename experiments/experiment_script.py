import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import argparse
from tqdm import tqdm
import torch
import scipy.stats
from fans import compare_conditional_probs, test_shifted_nodes_independence
from utils import dict_to_namespace, parse_dataset_indices, create_results_dir, make_json_serializable, setup_gpu, get_y_dependent_nodes
from model.carefl import CAREFL
import iscan
from scipy import stats
from sklearn.linear_model import LinearRegression

def run_carefl_model(data_env1, data_env2, adj, config, device, metadata):
    """Run CAREFL FANS model and return results."""
    results = {}
    try:
        print("Running CAREFL FANS model...")
        dag = adj
        
        # Create and train model for env1
        mod1 = CAREFL(config)
        mod1.device = device
        
        print("Training model1 on env1 data...")
        mod1.fit_to_sem(data_env1, dag)
        
        print("Comparing conditional probabilities between environments...")
        # Detect shifted nodes using JS threshold
        likelihood_result = compare_conditional_probs(mod1, data_env1, data_env2, dag, js_threshold=0.1)
                
        # Get detected shifted nodes
        detected_shifted_nodes = likelihood_result.get('shifted_nodes', [])
        
        # Get node-specific JS divergence values
        shifted_nodes_prob = []
        shifted_nodes_prob_log = []
        shifted_nodes_x = []
        
        # Extract node-specific results
        for key in likelihood_result.keys():
            if isinstance(key, str) and "|" in key:  # Process keys with format "X{i}|pa(X{i})"
                node_result = likelihood_result[key]
                if 'node_idx' in node_result and 'avg_js_divergence' in node_result:
                    node_idx = node_result['node_idx']
                    js_value = node_result['avg_js_divergence']
                    
                    # Store as object
                    shifted_nodes_x.append({
                        'node': node_idx,
                        'avg_js': js_value,
                        'shift_detected': node_result['shift_detected']
                    })
                    
                    # Keep original list format
                    if node_result['shift_detected']:
                        shifted_nodes_prob.append(js_value)
                        shifted_nodes_prob_log.append(node_idx)
        
        # New: Test independence between noise of shifted nodes and their parent variables
        print(f"Testing independence between noise of shifted nodes and their parent variables...")
        
        # Test independence in environment 1
        env1_independence_results = test_shifted_nodes_independence(
            model=mod1,
            data=data_env1,
            shifted_nodes=detected_shifted_nodes,
            dag=dag,
            verbose=True
        )
        
        # Test independence in environment 2
        env2_independence_results = test_shifted_nodes_independence(
            model=mod1,
            data=data_env2,
            shifted_nodes=detected_shifted_nodes,
            dag=dag,
            verbose=True
        )
        
        # Calculate dependency ratio change between environments
        node_independence_changes = {}
        for node_idx in detected_shifted_nodes:
            if node_idx in env1_independence_results['node_results'] and node_idx in env2_independence_results['node_results']:
                env1_ratio = env1_independence_results['node_results'][node_idx]['dependent_ratio']
                env2_ratio = env2_independence_results['node_results'][node_idx]['dependent_ratio']
                node_independence_changes[node_idx] = {
                    'env1_dependent_ratio': env1_ratio,
                    'env2_dependent_ratio': env2_ratio,
                    'ratio_change': env2_ratio - env1_ratio,
                    'parent_count': env1_independence_results['node_results'][node_idx]['parent_count'],
                    'dependent_parents_env1': env1_independence_results['node_results'][node_idx]['dependent_parents'],
                    'dependent_parents_env2': env2_independence_results['node_results'][node_idx]['dependent_parents']
                }
        
        true_shifted_nodes = metadata["shifted_nodes"]
        detected_set = set(detected_shifted_nodes)
        true_set = set(true_shifted_nodes)

        correctly_detected = list(true_set.intersection(detected_set))
        missed_nodes = list(true_set.difference(detected_set))

        true_detected_independence = test_shifted_nodes_independence(
            model=mod1,
            data=data_env2,
            shifted_nodes=correctly_detected,
            dag=dag,
            verbose=True
        ) if correctly_detected else {"node_results": {}, "avg_dependent_ratio": 0}

        missed_independence = test_shifted_nodes_independence(
            model=mod1,
            data=data_env2,
            shifted_nodes=missed_nodes,
            dag=dag,
            verbose=True
        ) if missed_nodes else {"node_results": {}, "avg_dependent_ratio": 0}

        # Store in results
        results = {
            # (0) True information
            "true_shifted_nodes": metadata["shifted_nodes"],
            
            # (1) Detected shifted nodes
            "detected_shifted_nodes": detected_shifted_nodes,
            "detected_shifted_nodes_prob_log": shifted_nodes_prob_log,
            "detected_shifted_nodes_prob": shifted_nodes_prob,
            "detected_shifted_nodes_x": shifted_nodes_x,
            "node_js_details": {key: likelihood_result[key] for key in likelihood_result if isinstance(key, str) and "|" in key},
            
            # (3) Independence test results for shifted nodes' noise and parent variables
            "shifted_nodes_independence": {
                "correctly_detected":{ 
                    "nodes": correctly_detected,
                    "independence_results": true_detected_independence,
                    "avg_dependent_ratio": true_detected_independence['avg_dependent_ratio']
                },
                "missed_nodes":{
                    "nodes": missed_nodes,
                    "independence_results": missed_independence,
                    "avg_dependent_ratio": missed_independence['avg_dependent_ratio']
                }
            }
        }
    except Exception as e:
        print(f"Error in CAREFL FANS model: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e)}
        
    return results

def run_iscan_model(data_env1, data_env2, adj, y_node, metadata, config_results_dir, result_filename, config_name, device):
    """Run ISCAN model and return results."""
    results = {}
    try:
        print("Running ISCAN model...")
        
        # Set iscan's device to match our current device
        iscan.set_device(device)
        
        # Create tensors but let iscan.py handle the device placement
        data_env1_tensor = torch.tensor(data_env1); data_env2_tensor = torch.tensor(data_env2)
        
        # Run ISCAN to estimate shifted nodes - iscan.py will move data to GPU
        shifted_nodes_est, order, ratio_dict = iscan.est_node_shifts(data_env1_tensor, data_env2_tensor)
        dag = np.transpose(adj)
        
        # Ensure ratio_dict values are moved to CPU if they're on GPU
        ratio_dict_cpu = {k: float(v) for k, v in ratio_dict.items()}
        
        # Store ISCAN results
        results = {
            "detected_shifted_nodes": shifted_nodes_est,
            "order": order,
            "ratio_dict": ratio_dict_cpu
        }
        
        true_shifted = set(metadata["shifted_nodes"])
        detected = set([int(n) for n in shifted_nodes_est])

        # Check if Y node detection was correct
        y_detected = y_node in [int(n) for n in shifted_nodes_est]
        results["y_node_detection"] = {
            "detected": y_detected,
            "expected": True  # Y is always shifted in new setup
        }
        
        # Test functional shifts for ALL true shifted nodes regardless of detection status
        print(f"Testing functional shifts for ALL true shifted nodes, regardless of detection status")
        
        # Variables to store results
        functional_shift_results = {}
        true_node_results = {}
        
        # Record results for each true shifted node
        for true_node in true_shifted:
            is_detected = true_node in detected
            has_functional_shift = False

            # Find all parents of the shifted node from DAG
            parents = np.where(adj[:, true_node] == 1)[0].tolist()
                
            if parents:  # Only test if node has parents
                print(f"Testing functional shifts for true shifted node {true_node} with parents: {parents}")
                    
                # Find functionally shifted edges focused on this shifted node
                shift_detected, f_shifted_edges, test_details = iscan.find_functionally_shifted_edges_with_dag(
                    y_node=true_node,
                    parents_y=parents,
                    data_env1=data_env1_tensor,
                    data_env2=data_env2_tensor
                )

                has_functional_shift = shift_detected
                
                # Store functional shift results
                functional_shift_results[str(true_node)] = {
                    "shifted_edges": f_shifted_edges,
                    "test_details": test_details,
                    "has_functional_shift": has_functional_shift
                }
            
            # Store results for true shifted node
            true_node_results[true_node] = {
                "is_detected_as_shifted": is_detected,
                "has_functional_shift_detected": has_functional_shift if parents else "No parents to test"
            }
        
        # Add results to output
        results["true_shifted_node_summary"] = true_node_results
        results["functional_shift_results"] = functional_shift_results
        
        # Aggregate results
        all_shifted_edges = []
        for node_results in functional_shift_results.values():
            if node_results.get("shifted_edges"):
                all_shifted_edges.extend(node_results["shifted_edges"])
        
        results["all_shifted_edges"] = all_shifted_edges
        
    except Exception as e:
        print(f"Error in ISCAN model: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e)}
    
    return results

def run_linearccp_model(data_env1, data_env2, adj, metadata):
    """Run LinearCCP method to detect shifted nodes based on linear regression coefficients and residuals."""
    results = {}
    try:
        print("Running LinearCCP model...")
        
        # Select nodes with parents for analysis
        nodes_with_parents = []
        for node in range(adj.shape[1]):
            parents = np.where(adj[:, node] == 1)[0]
            if len(parents) > 0:
                nodes_with_parents.append(node)
                
        # Variables to store results
        shifted_nodes = []
        functional_change_nodes = []
        noise_change_nodes = []
        node_results = {}
        
        # Analyze each node
        for node in nodes_with_parents:
            # Find parent nodes
            parents = np.where(adj[:, node] == 1)[0]
            if len(parents) == 0:
                continue
            
            # Regression analysis in environment 1
            X_env1 = data_env1[:, parents]
            y_env1 = data_env1[:, node]
            
            model_env1 = LinearRegression()
            model_env1.fit(X_env1, y_env1)
            beta_env1 = model_env1.coef_
            residuals_env1 = y_env1 - model_env1.predict(X_env1)
            
            # Regression analysis in environment 2
            X_env2 = data_env2[:, parents]
            y_env2 = data_env2[:, node]
            
            model_env2 = LinearRegression()
            model_env2.fit(X_env2, y_env2)
            beta_env2 = model_env2.coef_
            residuals_env2 = y_env2 - model_env2.predict(X_env2)
            
            # (1) Test for coefficient differences (Chow test)
            # Combine data from both environments
            X_combined = np.vstack((X_env1, X_env2))
            y_combined = np.concatenate((y_env1, y_env2))
            
            # Create dummy variable for environment (0: env1, 1: env2)
            env_dummy = np.concatenate((np.zeros(len(y_env1)), np.ones(len(y_env2))))
            
            # Create interaction terms (environment dummy × each parent variable)
            interaction_terms = np.zeros((len(env_dummy), len(parents)))
            for i, parent_idx in enumerate(parents):
                interaction_terms[:, i] = env_dummy * X_combined[:, i]
            
            # Create extended design matrix
            X_extended = np.hstack((X_combined, env_dummy.reshape(-1, 1), interaction_terms))
            
            # Fit extended model
            extended_model = LinearRegression()
            extended_model.fit(X_extended, y_combined)
            
            # Extract interaction coefficients (last len(parents) coefficients)
            interaction_coefs = extended_model.coef_[-len(parents):]
            
            # Calculate SSR for F-test
            # Restricted model: model without interaction terms (only environment dummy)
            X_restricted = np.hstack((X_combined, env_dummy.reshape(-1, 1)))
            restricted_model = LinearRegression()
            restricted_model.fit(X_restricted, y_combined)
            
            # Calculate SSR for restricted and extended models
            y_pred_restricted = restricted_model.predict(X_restricted)
            y_pred_extended = extended_model.predict(X_extended)
            
            ssr_restricted = np.sum((y_combined - y_pred_restricted) ** 2)
            ssr_extended = np.sum((y_combined - y_pred_extended) ** 2)
            
            # Calculate degrees of freedom
            df1 = len(parents)  # Number of constraints (interaction terms)
            df2 = len(y_combined) - len(extended_model.coef_) - 1  # Residual degrees of freedom for extended model
            
            # Calculate F-statistic and p-value
            f_stat = ((ssr_restricted - ssr_extended) / df1) / (ssr_extended / df2)
            p_value_chow = 1 - stats.f.cdf(f_stat, df1, df2)
            
            function_changed = p_value_chow < 0.05
            
            # (2) Test for residual distribution differences (Kolmogorov-Smirnov test)
            ks_stat, p_value_ks = stats.ks_2samp(residuals_env1, residuals_env2)
            noise_changed = p_value_ks < 0.05
            
            # Store results
            is_shifted = function_changed or noise_changed
            
            if is_shifted:
                shifted_nodes.append(node)
                if function_changed:
                    functional_change_nodes.append(node)
                if noise_changed:
                    noise_change_nodes.append(node)
            
            node_results[node] = {
                "is_shifted": is_shifted,
                "parents": parents.tolist(),
                "function_test": {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value_chow),
                    "is_function_changed": function_changed,
                    "beta_env1": beta_env1.tolist(),
                    "beta_env2": beta_env2.tolist()
                },
                "noise_test": {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value_ks),
                    "is_noise_changed": noise_changed
                }
            }
            
            print(f"Node {node}: Shifted={is_shifted}, Function Changed={function_changed}, Noise Changed={noise_changed}")
        
        # Calculate accuracy metrics
        true_shifted_nodes = metadata.get("shifted_nodes", [])
        
        true_positives = [node for node in shifted_nodes if node in true_shifted_nodes]
        false_positives = [node for node in shifted_nodes if node not in true_shifted_nodes]
        false_negatives = [node for node in true_shifted_nodes if node not in shifted_nodes]
        
        precision = len(true_positives) / max(len(shifted_nodes), 1)
        recall = len(true_positives) / max(len(true_shifted_nodes), 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
        
        # Final results
        results = {
            "shifted_nodes": shifted_nodes,
            "functional_change_nodes": functional_change_nodes,
            "noise_change_nodes": noise_change_nodes,
            "node_results": node_results,
            "metrics": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        }
    except Exception as e:
        print(f"Error in LinearCCP model: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e)}
    
    return results

def process_dataset(dataset_idx, data_path, config_results_dir, args, device, config, node_count, config_name):
    """Process a single dataset and return results."""
    print(f"Processing dataset {dataset_idx} in {config_name}")
    
    # Load data and metadata
    data_env1 = np.load(f"{data_path}/data_env1_{dataset_idx}.npy")
    data_env2 = np.load(f"{data_path}/data_env2_{dataset_idx}.npy")
    adj = np.load(f"{data_path}/adj_{dataset_idx}.npy")
    
    with open(f"{data_path}/metadata_{dataset_idx}.json", 'r') as f:
        metadata = json.load(f)
    
    # Extract metadata
    shifted_nodes = metadata["shifted_nodes"]
    dag_structure = metadata.get("dag_structure", "standard")
    y_node = metadata.get("y_node", 0)  # Default to 0 if not specified
    
    print(f"Dataset info - Nodes: {node_count},  Config: {config_name}")
    print(f"Shifted nodes: {shifted_nodes}, Y node: {y_node}")
    
    # Get the noise type from metadata if available, otherwise use a default
    noise_type = metadata.get("noise_type", "Unknown")
    
    result_filename = f"{args.model}_nodes{node_count}_{config_name}_{dataset_idx}"
    
    # Initialize results dictionary
    results = {
        "dataset_info": metadata,
        "model": args.model,
        "config_name": config_name,
        "dag_structure": dag_structure,
        "device": str(device)
    }
    
    # Run the appropriate model
    if args.model == 'recovery':
        results["recovery"] = run_carefl_model(data_env1, data_env2, adj, config, device, metadata)
    elif args.model == 'iscan':
        results["iscan"] = run_iscan_model(
            data_env1=data_env1, 
            data_env2=data_env2, 
            adj=adj, 
            y_node=y_node,
            metadata=metadata, 
            config_results_dir=config_results_dir, 
            result_filename=result_filename,
            config_name=config_name,
            device=device
        )

    elif args.model == 'linearccp':
        results["linearccp"] = run_linearccp_model(data_env1, data_env2, adj, metadata)
    
    # Save results for this dataset
    with open(f"{config_results_dir}/{result_filename}.json", 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    print(f"Results saved to {config_results_dir}/{result_filename}.json")
    
    return results

# ----- Main Function -----

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run experiments with selected model')
    parser.add_argument('--model', type=str, choices=['recovery', 'iscan', 'linearccp'], required=True,
                      help='Model to run: recovery (CAREFL), iscan, or linearccp')
    parser.add_argument('--nodes', type=int, nargs='+', default=[5, 10, 15], 
                      help='Node counts to process (default: 5 10 15)')
    parser.add_argument('--noise_types', type=str, nargs='+', default=["Gaussian", "Gumbel", "Laplace"],
                      help='Noise types to process (default: Gaussian Gumbel Laplace)')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU, default: -1)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for results (default: auto-generated with timestamp)')
    parser.add_argument('--config_type', type=str, choices=['ER','SF'], default='all',
                      help='Filter experiments by configuration type (default: all)')
    parser.add_argument('--dataset_indices', type=str, default=None,
                      help='Indices of datasets to process (e.g., "0-5" for datasets 0,1,2,3,4,5)')
    args = parser.parse_args()

    # Set up device (GPU/CPU)
    device = setup_gpu(args.gpu)

    # Print selected model and parameters
    print(f"Running experiments with model: {args.model}")
    print(f"Processing node counts: {args.nodes}")
    print(f"Processing noise types: {args.noise_types}")
    print(f"Configuration type filter: {args.config_type}")
    if args.dataset_indices:
        print(f"Dataset indices: {args.dataset_indices}")
    print(f"Using device: {device}")

    # Create results directory
    main_results_dir = create_results_dir(args.output_dir, args.model)

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config = dict_to_namespace(config)

    # Add device configuration to config
    config.device = device

    # Define data directory
    data_dir = "data"

    # Use the node counts and noise types specified in command line arguments
    node_counts = args.nodes

    # Filter configurations based on config_type argument
    if args.config_type == 'all':
        configurations = ["ER", "SF"]
    else:
        configurations = [args.config_type]

    print(f"Processing configurations: {configurations}")

    # Process each node count
    for node_count in node_counts:
        print(f"\n===== Processing experiments with {node_count} nodes =====")
        
        # Create results directory for this node count
        node_results_dir = f"{main_results_dir}/nodes_{node_count}"
        if not os.path.exists(node_results_dir):
            os.makedirs(node_results_dir)
    
        # Process each configuration
        for config_name in configurations:
            print(f"\nConfiguration: {config_name}")
            
            # Create results directory for this configuration
            config_results_dir = f"{node_results_dir}/{config_name}"
            if not os.path.exists(config_results_dir):
                os.makedirs(config_results_dir)

            # Path to data for this node count, noise type, and configuration
            data_path = f"{data_dir}/nodes_{node_count}/{config_name}"
            
            # Skip if the data directory doesn't exist
            if not os.path.exists(data_path):
                print(f"Data directory {data_path} not found, skipping...")
                continue
            
            # Find all dataset indices (extract from filenames)
            dataset_indices = []
            for file in os.listdir(data_path):
                if file.startswith("metadata_") and file.endswith(".json"):
                    idx = int(file.replace("metadata_", "").replace(".json", ""))
                    dataset_indices.append(idx)
            
            dataset_indices.sort()
            
            # Filter dataset indices if specified
            selected_indices = parse_dataset_indices(args.dataset_indices, dataset_indices)
            if args.dataset_indices:
                print(f"Processing datasets with indices: {selected_indices}")
            
            # Process each dataset
            for dataset_idx in selected_indices:
                process_dataset(
                    dataset_idx, 
                    data_path, 
                    config_results_dir, 
                    args, 
                    device, 
                    config, 
                    node_count, 
                    config_name
                )

    print(f"\n===== All experiments completed =====")
    print(f"Results saved to {main_results_dir}/")
    print(f"GPU usage: {device}")
    print(f"DAG structure: Y node selected first with all other nodes as ancestors")
    print(f"Generated data for {len(configurations)} configurations: {', '.join(configurations)}")

if __name__ == "__main__":
    main()

