#!/usr/bin/env python

import os
import json
import numpy as np
import glob
from collections import defaultdict
import pandas as pd

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def analyze_results(results_base_dir):
    """
    Analyze linearccp result directories by graph type and node count
    
    Args:
        results_base_dir: Base path for results directory
    """
    # Prepare dataframe for results
    columns = ["NodeCount", "GraphType", "DatasetCount", 
               "F1_Shifted_Nodes",
               "F1_Function_Shift_Combined", 
               "F1_Noise_Shift_Combined", 
               "F1_Function_Shift_True_Pop", "F1_Noise_Shift_True_Pop",
               "Macro_F1_Combined", "Macro_F1_True_Pop"]
    results_df = pd.DataFrame(columns=columns)
    
    # Process each node count directory
    node_dirs = sorted([d for d in os.listdir(results_base_dir) if d.startswith("nodes_")])
    
    for node_dir in node_dirs:
        node_count = int(node_dir.replace("nodes_", ""))
        node_path = os.path.join(results_base_dir, node_dir)
        
        # Process each graph type
        for graph_type in ["ER", "SF"]:
            graph_type_path = os.path.join(node_path, graph_type)
            
            if not os.path.exists(graph_type_path):
                print(f"Warning: Directory {graph_type_path} does not exist.")
                continue
            
            # Find result files
            result_files = glob.glob(os.path.join(graph_type_path, "linearccp_*.json"))
            
            if not result_files:
                print(f"Warning: No result files in {graph_type_path}.")
                continue
            
            print(f"\n===== Analyzing: {node_count} nodes, {graph_type} graph type ({len(result_files)} files) =====")
            
            # Variables to store results
            metrics = {
                'f1_shifted': [],
                'f1_function_combined': [],
                'f1_noise_combined': [],
                'f1_function_true_pop': [],
                'f1_noise_true_pop': [],
                'macro_f1_combined': [],
                'macro_f1_true_pop': []
            }
            
            # Analyze each file
            for file_path in result_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if linearccp results exist
                    if "linearccp" not in data:
                        continue
                    
                    linearccp_data = data["linearccp"]
                    
                    # Check for errors
                    if "error" in linearccp_data:
                        continue
                    
                    # Get dataset info
                    dataset_info = data.get("dataset_info", {})
                    
                    # Find nodes with parents from node_results
                    nodes_with_parents = set()
                    
                    if "node_results" in linearccp_data:
                        for node_id, node_info in linearccp_data["node_results"].items():
                            if "parents" in node_info and node_info["parents"]:
                                nodes_with_parents.add(int(node_id))
                    
                    # Skip if no nodes with parents
                    if not nodes_with_parents:
                        print(f"Warning: No nodes with parents in {file_path}.")
                        continue
                    
                    # Get true and detected shifted nodes
                    true_shifted = set(dataset_info.get("shifted_nodes", []))
                    detected_shifted = set(linearccp_data.get("shifted_nodes", []))
                    
                    if not true_shifted:
                        print(f"Warning: No true shifted nodes in {file_path}.")
                        continue
                    
                    # Filter to nodes with parents
                    true_shifted_with_parents = true_shifted.intersection(nodes_with_parents)
                    detected_shifted_with_parents = detected_shifted.intersection(nodes_with_parents)
                    
                    # 1. Calculate Shifted Nodes F1 (true shifted nodes vs detected shifted nodes)
                    shifted_tp = len(true_shifted_with_parents.intersection(detected_shifted_with_parents))
                    shifted_fp = len(detected_shifted_with_parents - true_shifted_with_parents)
                    shifted_fn = len(true_shifted_with_parents - detected_shifted_with_parents)
                    shifted_tn = len(nodes_with_parents - true_shifted_with_parents - detected_shifted_with_parents)
                    
                    shifted_precision = shifted_tp / max(shifted_tp + shifted_fp, 1)
                    shifted_recall = shifted_tp / max(shifted_tp + shifted_fn, 1)
                    
                    f1_shifted = calculate_f1_score(shifted_precision, shifted_recall)
                    
                    metrics['f1_shifted'].append(f1_shifted)
                    
                    # Get shift types
                    shift_types = dataset_info.get("shift_types", {})
                    
                    # Get linearccp's function/noise shift predictions
                    predicted_function_shifts = set(linearccp_data.get("functional_change_nodes", []))
                    predicted_noise_shifts = set(linearccp_data.get("noise_change_nodes", []))
                    
                    # Filter to nodes with parents
                    predicted_function_shifts = predicted_function_shifts.intersection(nodes_with_parents)
                    predicted_noise_shifts = predicted_noise_shifts.intersection(nodes_with_parents)
                    
                    # Handle nodes predicted as both function and noise shifts
                    both_predicted = predicted_function_shifts.intersection(predicted_noise_shifts)
                    only_function_predicted = predicted_function_shifts - both_predicted
                    only_noise_predicted = predicted_noise_shifts - both_predicted
                    
                    # Categorize true shifted nodes
                    true_function_shifts = {node for node in true_shifted_with_parents 
                                         if shift_types.get(str(node), "") == "function"}
                    
                    true_noise_shifts = {node for node in true_shifted_with_parents 
                                      if shift_types.get(str(node), "").startswith("noise") or 
                                         shift_types.get(str(node), "") == "noise"}
                    
                    # Create combined set (union of true and detected shifted nodes)
                    combined_set = true_shifted_with_parents.union(detected_shifted_with_parents)
                    
                    # 2. Calculate Function Shift F1 in combined set
                    true_func_in_combined = true_function_shifts.intersection(combined_set)
                    est_func_in_combined = only_function_predicted.intersection(combined_set)
                    
                    # Special case: if no function shifts in either true or estimated, F1=1
                    if len(true_func_in_combined) == 0 and len(est_func_in_combined) == 0:
                        f1_function_combined = 1.0
                    else:
                        func_tp = len(true_func_in_combined.intersection(est_func_in_combined))
                        func_fp = len(est_func_in_combined - true_func_in_combined)
                        func_fn = len(true_func_in_combined - est_func_in_combined)
                        
                        func_precision = func_tp / max(func_tp + func_fp, 1)
                        func_recall = func_tp / max(func_tp + func_fn, 1)
                        f1_function_combined = calculate_f1_score(func_precision, func_recall)
                    
                    # Calculate traditional accuracy for function shifts (over all nodes with parents)
                    
                    metrics['f1_function_combined'].append(f1_function_combined)
                    
                    # 3. Calculate Noise Shift F1 in combined set
                    true_noise_in_combined = true_noise_shifts.intersection(combined_set)
                    est_noise_in_combined = only_noise_predicted.intersection(combined_set)
                    
                    # Special case: if no noise shifts in either true or estimated, F1=1
                    if len(true_noise_in_combined) == 0 and len(est_noise_in_combined) == 0:
                        f1_noise_combined = 1.0
                    else:
                        noise_tp = len(true_noise_in_combined.intersection(est_noise_in_combined))
                        noise_fp = len(est_noise_in_combined - true_noise_in_combined)
                        noise_fn = len(true_noise_in_combined - est_noise_in_combined)
                        
                        noise_precision = noise_tp / max(noise_tp + noise_fp, 1)
                        noise_recall = noise_tp / max(noise_tp + noise_fn, 1)
                        f1_noise_combined = calculate_f1_score(noise_precision, noise_recall)
                    
                    # Calculate traditional accuracy for noise shifts (over all nodes with parents)
                    
                    metrics['f1_noise_combined'].append(f1_noise_combined)
                    
                    # --- New F1 scores with true_shifted_with_parents as population ---
                    population_true_shifted = true_shifted_with_parents

                    # Function Shift F1 (Population: true_shifted_with_parents)
                    true_func_in_tspop = true_function_shifts
                    est_func_in_tspop = only_function_predicted.intersection(population_true_shifted)

                    if not population_true_shifted:
                        f1_function_true_pop = 0.0
                    elif len(true_func_in_tspop) == 0 and len(est_func_in_tspop) == 0:
                        f1_function_true_pop = 1.0
                    else:
                        func_tp_tspop = len(true_func_in_tspop.intersection(est_func_in_tspop))
                        func_fp_tspop = len(est_func_in_tspop - true_func_in_tspop)
                        func_fn_tspop = len(true_func_in_tspop - est_func_in_tspop)
                        
                        func_precision_tspop = func_tp_tspop / max(func_tp_tspop + func_fp_tspop, 1)
                        func_recall_tspop = func_tp_tspop / max(func_tp_tspop + func_fn_tspop, 1)
                        f1_function_true_pop = calculate_f1_score(func_precision_tspop, func_recall_tspop)
                    metrics['f1_function_true_pop'].append(f1_function_true_pop)

                    # Noise Shift F1 (Population: true_shifted_with_parents)
                    true_noise_in_tspop = true_noise_shifts
                    est_noise_in_tspop = only_noise_predicted.intersection(population_true_shifted)

                    if not population_true_shifted:
                        f1_noise_true_pop = 0.0
                    elif len(true_noise_in_tspop) == 0 and len(est_noise_in_tspop) == 0:
                        f1_noise_true_pop = 1.0
                    else:
                        noise_tp_tspop = len(true_noise_in_tspop.intersection(est_noise_in_tspop))
                        noise_fp_tspop = len(est_noise_in_tspop - true_noise_in_tspop)
                        noise_fn_tspop = len(true_noise_in_tspop - est_noise_in_tspop)
                        
                        noise_precision_tspop = noise_tp_tspop / max(noise_tp_tspop + noise_fp_tspop, 1)
                        noise_recall_tspop = noise_tp_tspop / max(noise_tp_tspop + noise_fn_tspop, 1)
                        f1_noise_true_pop = calculate_f1_score(noise_precision_tspop, noise_recall_tspop)
                    metrics['f1_noise_true_pop'].append(f1_noise_true_pop)
                    
                    # 4. Calculate Macro F1 and Accuracy (based on combined F1s)
                    macro_f1_combined = (f1_function_combined + f1_noise_combined) / 2
                    metrics['macro_f1_combined'].append(macro_f1_combined)

                    # Calculate Macro F1 for True Pop
                    macro_f1_true_pop = (f1_function_true_pop + f1_noise_true_pop) / 2
                    metrics['macro_f1_true_pop'].append(macro_f1_true_pop)
                    
                    # Output per-file results
                    file_name = os.path.basename(file_path)
                    print(f"\n--- {file_name} results ---")
                    print(f"Nodes with parents: {len(nodes_with_parents)}")
                    print(f"Combined set size: {len(combined_set)}")
                    print(f"True shifted nodes with parents: {len(true_shifted_with_parents)}")
                    print(f"Detected shifted nodes with parents: {len(detected_shifted_with_parents)}")
                    print(f"True function shifts: {len(true_function_shifts)}, True noise shifts: {len(true_noise_shifts)}")
                    print(f"Function shifts in combined: {len(true_func_in_combined)}, Estimated: {len(est_func_in_combined)}")
                    print(f"Noise shifts in combined: {len(true_noise_in_combined)}, Estimated: {len(est_noise_in_combined)}")
                    print(f"Shift detection: F1={f1_shifted:.3f}")
                    print(f"Function shift (Combined Pop): F1={f1_function_combined:.3f}")
                    print(f"Noise shift (Combined Pop): F1={f1_noise_combined:.3f}")
                    print(f"Function shift (True Shifted Pop): F1={f1_function_true_pop:.3f}")
                    print(f"Noise shift (True Shifted Pop): F1={f1_noise_true_pop:.3f}")
                    print(f"Macro F1 (Combined): {macro_f1_combined:.3f}")
                    print(f"Macro F1 (True Pop): {macro_f1_true_pop:.3f}")
                    
                except Exception as e:
                    print(f"Error analyzing file {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate average metrics
            avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics.items()}
            
            # Add to results dataframe
            results_df = pd.concat([results_df, pd.DataFrame([{
                "NodeCount": node_count,
                "GraphType": graph_type,
                "DatasetCount": len(metrics['f1_shifted']),
                "F1_Shifted_Nodes": avg_metrics['f1_shifted'],
                "F1_Function_Shift_Combined": avg_metrics['f1_function_combined'],
                "F1_Noise_Shift_Combined": avg_metrics['f1_noise_combined'],
                "F1_Function_Shift_True_Pop": avg_metrics['f1_function_true_pop'],
                "F1_Noise_Shift_True_Pop": avg_metrics['f1_noise_true_pop'],
                "Macro_F1_Combined": avg_metrics['macro_f1_combined'],
                "Macro_F1_True_Pop": avg_metrics['macro_f1_true_pop']
            }])], ignore_index=True)
            
            # Output summary for current graph type
            print(f"\n===== Summary: {node_count} nodes, {graph_type} graph type, {len(metrics['f1_shifted'])} datasets =====")
            print(f"F1 (Shifted Nodes): {avg_metrics['f1_shifted']:.3f}")
            print(f"F1 (Function Shift in Combined): {avg_metrics['f1_function_combined']:.3f}")
            print(f"F1 (Noise Shift in Combined): {avg_metrics['f1_noise_combined']:.3f}")
            print(f"F1 (Function Shift in True Shifted Pop): {avg_metrics['f1_function_true_pop']:.3f}")
            print(f"F1 (Noise Shift in True Shifted Pop): {avg_metrics['f1_noise_true_pop']:.3f}")
            print(f"Macro F1 (Combined): {avg_metrics['macro_f1_combined']:.3f}")
            print(f"Macro F1 (True Pop): {avg_metrics['macro_f1_true_pop']:.3f}")
    
    # Output overall summary
    print("\n========== Results Summary ==========")
    # Sort by GraphType before printing
    sorted_results_df = results_df.sort_values(by=["GraphType", "NodeCount"])
    print(sorted_results_df.to_string(index=False, float_format="{:.3f}".format))
    
    # Save to file
    results_df.to_csv("linearccp_analysis_results.csv", index=False, float_format='%.3f')
    print("\nResults saved to linearccp_analysis_results.csv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LinearCCP result files")
    parser.add_argument("--dir", type=str, default="results/linearccp",
                      help="Results directory path")
    
    args = parser.parse_args()
    analyze_results(args.dir)