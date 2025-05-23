import json
import os
from sklearn.metrics import f1_score

# Define the path to the directory containing the JSON files
json_dir = "results/gpr/"
node_ls = [10, 20, 30, 40, 50]
graph_ls = ["ER", "SF"]

for node in node_ls:
    for graph in graph_ls:
        # Initialize lists to store true labels and predictions
        true_labels = []
        predicted_labels = []

        # Iterate through all JSON files (assuming they are named gpr_nodes{node}_{graph}_{i}.json)
        for i in range(1, 31):
            # Construct the file path
            file_path = os.path.join(json_dir, f"nodes_{node}", graph, f"gpr_nodes{node}_{graph}_{i}.json")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract true shift nodes and predicted function shift nodes
            dataset_info = data["dataset_info"]
            shifted_nodes = {int(node) for node in dataset_info["shifted_nodes"]}
            shift_types = dataset_info["shift_types"]
            estimated_shift_types = data["estimated_shift_types"]

            # Identify nodes with true function shifts
            true_function_shifts = {int(node) for node, shift_type in shift_types.items() if shift_type == "function"}

            # Identify nodes with predicted function shifts
            predicted_function_shifts = {int(node) for node, shift_type in estimated_shift_types.items() if shift_type == "function"}

            # Use all shifted nodes (noise or function) as the evaluation set
            relevant_nodes = shifted_nodes

            # Create binary labels for the relevant nodes
            true_label = [1 if node in true_function_shifts else 0 for node in relevant_nodes]
            predicted_label = [1 if node in predicted_function_shifts else 0 for node in relevant_nodes]
            
            # Append to the lists
            true_labels.extend(true_label)
            predicted_labels.extend(predicted_label)

        # Calculate F1-score
        if true_labels and predicted_labels:  # Ensure there is data to calculate F1-score
            f1 = f1_score(true_labels, predicted_labels)
            print(f"F1-score for function shift detection in {graph} for {node} nodes: {f1:.4f}")
        else:
            print(f"No valid data for {graph} with {node} nodes.")

import json
import os
from sklearn.metrics import f1_score

for node in node_ls:
    for graph in graph_ls:
        # Initialize lists to store true labels and predictions
        true_labels = []
        predicted_labels = []

        # Iterate through all JSON files (assuming they are named gpr_nodes{node}_{graph}_{i}.json)
        for i in range(1, 31):
            # Construct the file path
            file_path = os.path.join(json_dir, f"nodes_{node}", graph, f"gpr_nodes{node}_{graph}_{i}.json")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract true shift nodes and predicted noise shift nodes
            dataset_info = data["dataset_info"]
            shifted_nodes = {int(node) for node in dataset_info["shifted_nodes"]}
            shift_types = dataset_info["shift_types"]
            estimated_shift_types = data["estimated_shift_types"]

            # Identify nodes with true noise shifts
            true_noise_shifts = {int(node) for node, shift_type in shift_types.items() if shift_type.startswith("noise_")}

            # Identify nodes with predicted noise shifts
            predicted_noise_shifts = {int(node) for node, shift_type in estimated_shift_types.items() if shift_type == ("noise")}

            # Use all shifted nodes (noise or function) as the evaluation set
            relevant_nodes = shifted_nodes

            # Create binary labels for the relevant nodes
            true_label = [1 if node in true_noise_shifts else 0 for node in relevant_nodes]
            predicted_label = [1 if node in predicted_noise_shifts else 0 for node in relevant_nodes]
            
            # Append to the lists
            true_labels.extend(true_label)
            predicted_labels.extend(predicted_label)

        # Calculate F1-score
        if true_labels and predicted_labels:  # Ensure there is data to calculate F1-score
            f1 = f1_score(true_labels, predicted_labels)
            print(f"F1-score for noise shift detection in {graph} for {node} nodes: {f1:.4f}")
        else:
            print(f"No valid data for {graph} with {node} nodes.")

import json
import os
from sklearn.metrics import f1_score

# Define the path to the directory containing the JSON files
json_dir = "results/gpr/"
node_ls = [10, 20, 30, 40, 50]
graph_ls = ["ER", "SF"]

for node in node_ls:
    for graph in graph_ls:
        # Initialize lists to store true labels and predictions
        true_labels = []
        predicted_labels = []

        # Iterate through all JSON files (assuming they are named gpr_nodes{node}_{graph}_{i}.json)
        for i in range(1, 31):
            # Construct the file path
            file_path = os.path.join(json_dir, f"nodes_{node}", graph, f"gpr_nodes{node}_{graph}_{i}.json")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract true shift nodes and predicted shift nodes
            dataset_info = data["dataset_info"]
            shifted_nodes = {int(node) for node in dataset_info["shifted_nodes"]}
            shift_types = dataset_info["shift_types"]
            estimated_shift_types = data["estimated_shift_types"]

            # Use all shifted nodes (noise or function) as the evaluation set
            relevant_nodes = shifted_nodes

            # Create multi-class labels for the relevant nodes
            # 0: No shift, 1: Function shift, 2: Noise shift
            true_label = [
                1 if shift_types[str(node)] == "function" else 2 if shift_types[str(node)].startswith("noise_") else 0
                for node in relevant_nodes
            ]
            predicted_label = [
                1 if estimated_shift_types.get(str(node)) == "function" else 2 if estimated_shift_types.get(str(node), "") == "noise" else 0
                for node in relevant_nodes
            ]
            
            # Append to the lists
            true_labels.extend(true_label)
            predicted_labels.extend(predicted_label)

        # Calculate F1-score
        if true_labels and predicted_labels:  # Ensure there is data to calculate F1-score
            f1 = f1_score(true_labels, predicted_labels, average="macro")  # Use macro-average for multi-class F1
            print(f"F1-score for classification task in {graph} for {node} nodes: {f1:.4f}")
        else:
            print(f"No valid data for {graph} with {node} nodes.")