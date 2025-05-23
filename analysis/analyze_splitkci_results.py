import json
import os
import numpy as np
from sklearn.metrics import f1_score

# Define the path to the directory containing the JSON and adjacency matrix files
json_dir = "results/splitkci/"
adjacency_dir = "data/"
node_ls = [10, 20, 30, 40, 50]
graph_ls = ["ER", "SF"]

for node in node_ls:
    for graph in graph_ls:
        # Initialize lists to store true labels and predictions
        true_labels = []
        predicted_labels = []

        # Iterate through all JSON files (assuming they are named results_1.json to results_30.json)
        for i in range(1, 31):

            # Load the adjacency matrix for the current graph
            adjacency_file = os.path.join(adjacency_dir, f"nodes_{node}", graph, f"adj_{i}.npy")
            if not os.path.exists(adjacency_file):
                print(f"Adjacency matrix file not found: {adjacency_file}")
                continue

            adjacency_matrix = np.load(adjacency_file)

            # Identify nodes that have at least one parent (non-zero entries in their column)
            nodes_with_parents = set(np.where(adjacency_matrix.sum(axis=0) > 0)[0])

            # Construct the file path
            file_path = os.path.join(json_dir, f"nodes_{node}", graph, f"results_{i}.json")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract shifted_nodes and detected_shifts
            shifted_nodes = data["shifted_nodes"]
            detected_shifts = data["detected_shifts"]
            
            # Filter nodes to include only those with parents
            relevant_nodes = [node for node in range(node) if node in nodes_with_parents]

            # Create binary labels for the relevant nodes
            true_label = [1 if node in shifted_nodes else 0 for node in relevant_nodes]
            predicted_label = [1 if node in detected_shifts else 0 for node in relevant_nodes]
            
            # Append to the lists
            true_labels.extend(true_label)
            predicted_labels.extend(predicted_label)

        # Calculate F1-score
        if true_labels and predicted_labels:  # Ensure there is data to calculate F1-score
            f1 = f1_score(true_labels, predicted_labels)
            print(f"F1-score for shift detection in {graph} for {node} nodes: {f1:.4f}")
        else:
            print(f"No valid data for {graph} with {node} nodes.")