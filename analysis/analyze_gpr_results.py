import json
import os
from sklearn.metrics import f1_score

# Define the path to the directory containing the JSON files
json_dir = "results/gpr/"
node_ls = [10, 20, 30, 40, 50]
graph_ls = ["ER", "SF"]

for node in node_ls:
    for graph in graph_ls:
        # Initialize lists for all three tasks
        # Task 1: Function shift detection
        true_labels_func = []
        predicted_labels_func = []
        
        # Task 2: Noise shift detection
        true_labels_noise = []
        predicted_labels_noise = []
        
        # Task 3: Multi-class classification
        true_labels_multi = []
        predicted_labels_multi = []

        # Iterate through all JSON files
        for i in range(1, 31):
            # Construct the file path
            file_path = os.path.join(json_dir, f"nodes_{node}", graph, f"gpr_nodes{node}_{graph}_{i}.json")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract common data
            dataset_info = data["dataset_info"]
            shifted_nodes = {int(node) for node in dataset_info["shifted_nodes"]}
            shift_types = dataset_info["shift_types"]
            estimated_shift_types = data["estimated_shift_types"]

            # Identify true shifts
            true_function_shifts = {int(node) for node, shift_type in shift_types.items() if shift_type == "function"}
            true_noise_shifts = {int(node) for node, shift_type in shift_types.items() if shift_type.startswith("noise_")}

            # Identify predicted shifts
            predicted_function_shifts = {int(node) for node, shift_type in estimated_shift_types.items() if shift_type == "function"}
            predicted_noise_shifts = {int(node) for node, shift_type in estimated_shift_types.items() if shift_type == "noise"}

            # Use all shifted nodes as the evaluation set
            relevant_nodes = shifted_nodes

            # Task 1: Function shift detection (binary)
            true_label_func = [1 if node in true_function_shifts else 0 for node in relevant_nodes]
            predicted_label_func = [1 if node in predicted_function_shifts else 0 for node in relevant_nodes]
            true_labels_func.extend(true_label_func)
            predicted_labels_func.extend(predicted_label_func)
            
            # Task 2: Noise shift detection (binary)
            true_label_noise = [1 if node in true_noise_shifts else 0 for node in relevant_nodes]
            predicted_label_noise = [1 if node in predicted_noise_shifts else 0 for node in relevant_nodes]
            true_labels_noise.extend(true_label_noise)
            predicted_labels_noise.extend(predicted_label_noise)
            
            # Task 3: Multi-class classification (0: No shift, 1: Function shift, 2: Noise shift)
            true_label_multi = [
                1 if shift_types[str(node)] == "function" else 2 if shift_types[str(node)].startswith("noise_") else 0
                for node in relevant_nodes
            ]
            predicted_label_multi = [
                1 if estimated_shift_types.get(str(node)) == "function" else 2 if estimated_shift_types.get(str(node), "") == "noise" else 0
                for node in relevant_nodes
            ]
            true_labels_multi.extend(true_label_multi)
            predicted_labels_multi.extend(predicted_label_multi)

        # Calculate and print F1-scores for all three tasks
        print(f"\n=== Results for {graph} with {node} nodes ===")
        
        # Task 1: Function shift detection F1-score
        if true_labels_func and predicted_labels_func:
            f1_func = f1_score(true_labels_func, predicted_labels_func)
            print(f"F1-score for function shift detection: {f1_func:.4f}")
        else:
            print(f"No valid data for function shift detection")
            
        # Task 2: Noise shift detection F1-score
        if true_labels_noise and predicted_labels_noise:
            f1_noise = f1_score(true_labels_noise, predicted_labels_noise)
            print(f"F1-score for noise shift detection: {f1_noise:.4f}")
        else:
            print(f"No valid data for noise shift detection")
            
        # Task 3: Multi-class classification F1-score
        if true_labels_multi and predicted_labels_multi:
            f1_multi = f1_score(true_labels_multi, predicted_labels_multi, average="macro")
            print(f"F1-score for multi-class classification: {f1_multi:.4f}")
        else:
            print(f"No valid data for multi-class classification")