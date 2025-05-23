import os
import json
import numpy as np
import torch
from datetime import datetime
from splitkci.dependence_measures import KCIMeasure

# GPU settings and seed initialization
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.manual_seed(42)
np.random.seed(42)

# Load data
def load_data(adj_path, env1_path, env2_path, metadata_path):
    adj_matrix = np.load(adj_path)
    data_env1 = np.load(env1_path)
    data_env2 = np.load(env2_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    shifted_nodes = metadata.get("shifted_nodes", [])
    return adj_matrix, data_env1, data_env2, shifted_nodes, metadata

# Identify parent-child relationships
def get_parent_child_relationships(adj_matrix):
    parent_child_pairs = []
    for child in range(adj_matrix.shape[0]):
        parents = np.where(adj_matrix[:, child] == 1)[0]
        if len(parents) > 0:
            parent_child_pairs.append((child, parents))
    return parent_child_pairs

# Detect shifts
def detect_shifts(parent_child_pairs, data_env1, data_env2):
    detected_shifts = []

    for child, parents in parent_child_pairs:
        # Extract parent and child node data
        parent_data_env1 = data_env1[:, parents]
        parent_data_env2 = data_env2[:, parents]
        child_data_env1 = data_env1[:, child]
        child_data_env2 = data_env2[:, child]

        # Combine data from both environments
        parent_data = np.vstack([parent_data_env1, parent_data_env2])
        child_data = np.hstack([child_data_env1, child_data_env2])
        env_labels = np.array([0] * len(child_data_env1) + [1] * len(child_data_env2))

        # Adjust dimensions if necessary
        if child_data.ndim == 1:
            child_data = child_data[:, None]

        # Convert data to PyTorch tensors
        parent_data = torch.tensor(parent_data, dtype=torch.float32)
        child_data = torch.tensor(child_data, dtype=torch.float32)
        env_labels = torch.tensor(env_labels, dtype=torch.float32)

        # Dynamically set sigma2 based on data variance
        kernel_a_args = {"sigma2": parent_data.var().item()}
        kernel_b_args = {"sigma2": child_data.var().item()}
        kernel_ca_args = {"sigma2": parent_data.var().item()}
        kernel_cb_args = {"sigma2": child_data.var().item()}

        # Create KCIMeasure object
        kci = KCIMeasure(
            kernel_a="gaussian",
            kernel_ca="gaussian",
            kernel_b="gaussian",
            kernel_cb="gaussian",
            kernel_ca_args=kernel_ca_args,
            kernel_a_args=kernel_a_args,
            kernel_cb_args=kernel_cb_args,
            kernel_b_args=kernel_b_args,
            biased=True
        )

        # Compute p-value using KCIMeasure
        p_value = kci.test(parent_data, child_data, env_labels, train_test_split=0.001) # 0.001 is the threshold for train-test split to efficiently compute p-value

        # Determine if a shift occurred (p-value < 0.05)
        if p_value < 0.05:
            detected_shifts.append(child)

    return detected_shifts

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Iterate over all cases
data_dir = "data"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_results_dir = f"results/SplitKCI_{timestamp}"
os.makedirs(main_results_dir, exist_ok=True)

# Define node counts and noise types
node_counts = [10, 20, 30, 40, 50]
noise_types = ["ER", "SF"]

# Process each node count
for node_count in node_counts:
    print(f"\n===== Processing experiments with {node_count} nodes =====")
    
    # Create results directory for this node count
    node_results_dir = f"{main_results_dir}/nodes_{node_count}"
    os.makedirs(node_results_dir, exist_ok=True)
    
    # Process each noise type
    for noise_type in noise_types:
        print(f"\nNoise Type: {noise_type}")
        
        # Create results directory for this noise type
        noise_results_dir = f"{node_results_dir}/{noise_type}"
        os.makedirs(noise_results_dir, exist_ok=True)
        
        # Path to data for this node count and noise type
        data_path = f"{data_dir}/nodes_{node_count}/{noise_type}"
        
        # Check if this path exists
        if not os.path.exists(data_path):
            print(f"WARNING: Data path {data_path} does not exist. Skipping...")
            continue
        
        # Find all dataset indices (extract from filenames)
        dataset_indices = [
            int(file.replace("metadata_", "").replace(".json", ""))
            for file in os.listdir(data_path) if file.startswith("metadata_") and file.endswith(".json")
        ]
        dataset_indices.sort()
        
        # Process each dataset
        for dataset_idx in dataset_indices:
            print(f"\nProcessing dataset {dataset_idx}")
            
            # Load data and metadata
            adj_path = f"{data_path}/adj_{dataset_idx}.npy"
            env1_path = f"{data_path}/data_env1_{dataset_idx}.npy"
            env2_path = f"{data_path}/data_env2_{dataset_idx}.npy"
            metadata_path = f"{data_path}/metadata_{dataset_idx}.json"
            
            adj_matrix, data_env1, data_env2, shifted_nodes, metadata = load_data(adj_path, env1_path, env2_path, metadata_path)
            parent_child_pairs = get_parent_child_relationships(adj_matrix)

            # Perform shift detection
            detected_shifts = detect_shifts(parent_child_pairs, data_env1, data_env2)

            # Save results
            result = {
                "dataset_info": metadata,
                "detected_shifts": detected_shifts,
                "shifted_nodes": shifted_nodes
            }

            # Save results for this dataset
            result_filename = f"results_{dataset_idx}.json"
            result_filepath = os.path.join(noise_results_dir, result_filename)
            with open(result_filepath, "w") as f:
                json.dump(make_json_serializable(result), f, indent=2)
            
            print(f"Results saved to {result_filepath}")

print(f"\n===== All experiments completed =====")
print(f"Results saved to {main_results_dir}/")