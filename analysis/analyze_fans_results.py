#!/usr/bin/env python

import os
import json
import numpy as np
import glob
import pandas as pd
import re # For parsing dataset index

# Define the path to the directory containing the JSON files and data
json_dir = "results/fans/"
data_root_dir = "data/"

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def get_dataset_idx_from_filename(filename):
    """
    Extracts dataset index from filenames like 'model_nodesN_config_IDX.json'
    """
    # Adjusted regex to be more general for model names like 'fans'
    match = re.search(r'_(\d+)\.json$', filename) 
    if match:
        return int(match.group(1))
    return None

def analyze_results(results_base_dir, data_root_dir_arg):
    """
    Analyze fans result directories by graph type and node count.
    Results are filtered by 'nodes_with_parents'. If not in JSON's dataset_info,
    it's dynamically loaded from the corresponding adj file in data_root_dir_arg.
    
    Args:
        results_base_dir: Base path for results directory
        data_root_dir_arg: Root directory for original dataset adj files
    """
    columns = ["NodeCount", "GraphType", "DatasetCount", 
               "F1_Shifted_Nodes", 
               "F1_Function_Shift_Combined", "F1_Noise_Shift_Combined", "Macro_F1_Combined",
               "F1_Function_Shift_True_Pop", "F1_Noise_Shift_True_Pop", "Macro_F1_True_Pop"]
    results_df = pd.DataFrame(columns=columns)
    
    node_dirs = sorted([d for d in os.listdir(results_base_dir) if d.startswith("nodes_")])
    
    for node_dir in node_dirs:
        node_count = int(node_dir.replace("nodes_", ""))
        node_path = os.path.join(results_base_dir, node_dir)
        
        for graph_type in ["ER", "SF"]:
            graph_type_path = os.path.join(node_path, graph_type)
            
            if not os.path.exists(graph_type_path):
                print(f"Warning: Directory {graph_type_path} does not exist.")
                continue
            
            result_files = glob.glob(os.path.join(graph_type_path, "fans_*.json"))
            
            if not result_files:
                print(f"Warning: No result files in {graph_type_path}.")
                continue
            
            print(f"\n===== Analyzing: {node_count} nodes, {graph_type} graph type ({len(result_files)} files) =====")
            
            metrics = {
                'f1_shifted': [], 'f1_function_combined': [], 'f1_noise_combined': [],
                'macro_f1_combined': [], 'f1_function_true_pop': [], 
                'f1_noise_true_pop': [], 'macro_f1_true_pop': []
            }
            
            valid_datasets_count = 0
            for file_path in result_files:
                file_name = os.path.basename(file_path)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    dataset_info = data.get("dataset_info", {})
                    nodes_with_parents = set(dataset_info.get("nodes_with_parents", []))

                    if not nodes_with_parents:
                        print(f"Info: 'nodes_with_parents' not found in dataset_info for '{file_name}'. Attempting dynamic load from adj file...")
                        dataset_idx = get_dataset_idx_from_filename(file_name)
                        if dataset_idx is None:
                            print(f"Warning: Could not parse dataset index from '{file_name}'. Skipping this file.")
                            continue
                        
                        adj_file_path = os.path.join(data_root_dir_arg, f"nodes_{node_count}", graph_type, f"adj_{dataset_idx}.npy")
                        
                        if not os.path.exists(adj_file_path):
                            print(f"Warning: Corresponding adj file '{adj_file_path}' not found. Skipping '{file_name}'.")
                            continue
                        
                        try:
                            adj = np.load(adj_file_path)
                            temp_nodes_with_parents_list = []
                            if adj is not None and adj.ndim == 2 and adj.shape[0] > 0 and adj.shape[0] == adj.shape[1]:
                                for node_idx_loop in range(adj.shape[1]): # adj.shape[1] is num_nodes
                                    if np.sum(adj[:, node_idx_loop]) > 0: # In-degree > 0 (assuming adj[parent, child])
                                        temp_nodes_with_parents_list.append(int(node_idx_loop))
                                nodes_with_parents = set(temp_nodes_with_parents_list)
                                print(f"Info: Dynamically loaded 'nodes_with_parents' for '{file_name}' from '{adj_file_path}': {len(nodes_with_parents)} nodes.")
                                if not nodes_with_parents:
                                     print(f"Info: Dynamically loaded 'nodes_with_parents' is empty for '{file_name}' (e.g., graph with no edges).")
                            else:
                                print(f"Warning: Loaded adj file '{adj_file_path}' has incorrect format or is empty. Skipping '{file_name}'.")
                                continue
                        except Exception as e_load_adj:
                            print(f"Warning: Error loading adj file '{adj_file_path}': {e_load_adj}. Skipping '{file_name}'.")
                            continue
                    
                    if "fans" not in data:
                        print(f"Warning: No fans results in '{file_name}'. Skipping.")
                        continue
                    
                    fans_data = data["fans"]
                    
                    if "error" in fans_data:
                        print(f"Warning: Error in '{file_name}': {fans_data['error']}. Skipping.")
                        continue
                    
                    # Use true_shifted_nodes from dataset_info as the primary ground truth
                    true_shifted_original = set(dataset_info.get("shifted_nodes", [])) 
                    if not true_shifted_original:
                         print(f"Warning: 'shifted_nodes' not found in dataset_info for '{file_name}'. Skipping.")
                         continue

                    detected_shifted_original = set(fans_data.get("detected_shifted_nodes", []))
                    
                    true_shifted = true_shifted_original.intersection(nodes_with_parents)
                    detected_shifted = detected_shifted_original.intersection(nodes_with_parents)
                    
                    if not true_shifted and nodes_with_parents:
                        print(f"Info: No true_shifted_nodes remain in the 'nodes_with_parents' population for '{file_name}' after filtering. Metrics might be 0 or 1.")
                    elif not true_shifted and not nodes_with_parents:
                         print(f"Info: Evaluation population ('nodes_with_parents') is empty and no true shifted nodes for '{file_name}'.")
                                   
                    shift_types = dataset_info.get("shift_types", {})
                    
                    true_function_shifts = {node for node in true_shifted 
                                          if shift_types.get(str(node), "") == "function"}
                    true_noise_shifts = {node for node in true_shifted 
                                       if shift_types.get(str(node), "").startswith("noise") or 
                                          shift_types.get(str(node), "") == "noise"}
                    
                    fans_function_shifts = set()
                    fans_noise_shifts = set()

                    classifiable_nodes_details = {}
                    if "shifted_nodes_independence" in fans_data and \
                       "correctly_detected" in fans_data["shifted_nodes_independence"] and \
                       "independence_results" in fans_data["shifted_nodes_independence"]["correctly_detected"] and \
                       "node_results" in fans_data["shifted_nodes_independence"]["correctly_detected"]["independence_results"]:
                        classifiable_nodes_details = fans_data["shifted_nodes_independence"]["correctly_detected"]["independence_results"]["node_results"]

                    for node_id_str, node_info in classifiable_nodes_details.items():
                        node_int = int(node_id_str)
                        if node_int in detected_shifted: # Ensure classified node is in our evaluation population
                            dependent_ratio = node_info.get("dependent_ratio", 0)
                            if dependent_ratio >= 0.66:
                                fans_function_shifts.add(node_int)
                            else:
                                fans_noise_shifts.add(node_int)
                    
                    shifted_tp = len(true_shifted.intersection(detected_shifted))
                    shifted_fp = len(detected_shifted - true_shifted)
                    shifted_fn = len(true_shifted - detected_shifted)
                    
                    shifted_precision = shifted_tp / max(shifted_tp + shifted_fp, 1)
                    shifted_recall = shifted_tp / max(shifted_tp + shifted_fn, 1)
                    f1_shifted = calculate_f1_score(shifted_precision, shifted_recall)
                    metrics['f1_shifted'].append(f1_shifted)
                    
                    combined_set = true_shifted.union(detected_shifted)
                    
                    true_func_in_combined = true_function_shifts.intersection(combined_set)
                    est_func_in_combined = fans_function_shifts.intersection(combined_set)
                    
                    if len(true_func_in_combined) == 0 and len(est_func_in_combined) == 0:
                        f1_function_combined = 1.0
                    else:
                        func_tp = len(true_func_in_combined.intersection(est_func_in_combined))
                        func_fp = len(est_func_in_combined - true_func_in_combined)
                        func_fn = len(true_func_in_combined - est_func_in_combined)
                        func_precision = func_tp / max(func_tp + func_fp, 1)
                        func_recall = func_tp / max(func_tp + func_fn, 1)
                        f1_function_combined = calculate_f1_score(func_precision, func_recall)
                    metrics['f1_function_combined'].append(f1_function_combined)
                    
                    true_noise_in_combined = true_noise_shifts.intersection(combined_set)
                    est_noise_in_combined = fans_noise_shifts.intersection(combined_set)
                    
                    if len(true_noise_in_combined) == 0 and len(est_noise_in_combined) == 0:
                        f1_noise_combined = 1.0
                    else:
                        noise_tp = len(true_noise_in_combined.intersection(est_noise_in_combined))
                        noise_fp = len(est_noise_in_combined - true_noise_in_combined)
                        noise_fn = len(true_noise_in_combined - est_noise_in_combined)
                        noise_precision = noise_tp / max(noise_tp + noise_fp, 1)
                        noise_recall = noise_tp / max(noise_tp + noise_fn, 1)
                        f1_noise_combined = calculate_f1_score(noise_precision, noise_recall)
                    metrics['f1_noise_combined'].append(f1_noise_combined)
                    
                    metrics['macro_f1_combined'].append((f1_function_combined + f1_noise_combined) / 2)

                    population_true_shifted = true_shifted

                    true_func_in_tspop = true_function_shifts 
                    est_func_in_tspop = fans_function_shifts.intersection(population_true_shifted)
                    if not population_true_shifted: f1_function_true_pop = 0.0 
                    elif len(true_func_in_tspop) == 0 and len(est_func_in_tspop) == 0: f1_function_true_pop = 1.0
                    else:
                        func_tp_tspop = len(true_func_in_tspop.intersection(est_func_in_tspop))
                        func_fp_tspop = len(est_func_in_tspop - true_func_in_tspop)
                        func_fn_tspop = len(true_func_in_tspop - est_func_in_tspop)
                        func_precision_tspop = func_tp_tspop / max(func_tp_tspop + func_fp_tspop, 1)
                        func_recall_tspop = func_tp_tspop / max(func_tp_tspop + func_fn_tspop, 1)
                        f1_function_true_pop = calculate_f1_score(func_precision_tspop, func_recall_tspop)
                    metrics['f1_function_true_pop'].append(f1_function_true_pop)

                    true_noise_in_tspop = true_noise_shifts 
                    est_noise_in_tspop = fans_noise_shifts.intersection(population_true_shifted)
                    if not population_true_shifted: f1_noise_true_pop = 0.0
                    elif len(true_noise_in_tspop) == 0 and len(est_noise_in_tspop) == 0: f1_noise_true_pop = 1.0
                    else:
                        noise_tp_tspop = len(true_noise_in_tspop.intersection(est_noise_in_tspop))
                        noise_fp_tspop = len(est_noise_in_tspop - true_noise_in_tspop)
                        noise_fn_tspop = len(true_noise_in_tspop - est_noise_in_tspop)
                        noise_precision_tspop = noise_tp_tspop / max(noise_tp_tspop + noise_fp_tspop, 1)
                        noise_recall_tspop = noise_tp_tspop / max(noise_tp_tspop + noise_fn_tspop, 1)
                        f1_noise_true_pop = calculate_f1_score(noise_precision_tspop, noise_recall_tspop)
                    metrics['f1_noise_true_pop'].append(f1_noise_true_pop)
                    
                    metrics['macro_f1_true_pop'].append((f1_function_true_pop + f1_noise_true_pop) / 2)
                    
                    valid_datasets_count +=1
                    
                    # print(f"\n--- {file_name} results (Eval pop: {len(nodes_with_parents)} nodes with parents) ---")
                    # print(f"Original true: {len(true_shifted_original)}, Original detected: {len(detected_shifted_original)}")
                    # print(f"Filtered true: {len(true_shifted)}, Filtered detected: {len(detected_shifted)}")
                    # print(f"Shift detection F1: {f1_shifted:.3f}")

                except Exception as e:
                    print(f"Error processing file '{file_name}': {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics.items()}
            
            results_df = pd.concat([results_df, pd.DataFrame([{
                "NodeCount": node_count, "GraphType": graph_type, "DatasetCount": valid_datasets_count,
                "F1_Shifted_Nodes": avg_metrics['f1_shifted'],
                "F1_Function_Shift_Combined": avg_metrics['f1_function_combined'],
                "F1_Noise_Shift_Combined": avg_metrics['f1_noise_combined'],
                "Macro_F1_Combined": avg_metrics['macro_f1_combined'],
                "F1_Function_Shift_True_Pop": avg_metrics['f1_function_true_pop'],
                "F1_Noise_Shift_True_Pop": avg_metrics['f1_noise_true_pop'],
                "Macro_F1_True_Pop": avg_metrics['macro_f1_true_pop']
            }])], ignore_index=True)
            
            print(f"\n===== Summary for FANS: {node_count} nodes, {graph_type} (eval on {valid_datasets_count} datasets with 'nodes_with_parents') =====")
            print(f"Avg F1 (Shifted Nodes): {avg_metrics['f1_shifted']:.3f}")
            print(f"Avg F1 (Function Shift in Combined): {avg_metrics['f1_function_combined']:.3f}")
            print(f"Avg F1 (Noise Shift in Combined): {avg_metrics['f1_noise_combined']:.3f}")
            print(f"Avg F1 (Function Shift in True Shifted Pop): {avg_metrics['f1_function_true_pop']:.3f}")
            print(f"Avg F1 (Noise Shift in True Shifted Pop): {avg_metrics['f1_noise_true_pop']:.3f}")
            print(f"Avg Macro F1 (Combined): {avg_metrics['macro_f1_combined']:.3f}")
            print(f"Avg Macro F1 (True Pop): {avg_metrics['macro_f1_true_pop']:.3f}")
    
    print("\n========== Overall FANS Results Summary (Filtered by nodes_with_parents) ==========")
    sorted_results_df = results_df.sort_values(by=["GraphType", "NodeCount"])
    print(sorted_results_df.to_string(index=False, float_format="{:.3f}".format))
    
    results_df.to_csv("fans_analysis_results_filtered.csv", index=False, float_format='%.3f')
    print("\nFiltered FANS results saved to fans_analysis_results_filtered.csv")

if __name__ == "__main__":
    analyze_results(json_dir, data_root_dir) 