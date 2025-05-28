import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from types import SimpleNamespace
import torch
import scipy.stats
from torch.utils.data import Dataset
import yaml
from types import SimpleNamespace

# ----- Utility Functions -----

def make_json_serializable(obj):
    """Convert objects to JSON serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else str(obj)
    elif isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def is_nan_or_inf(value):
    """Check if value is NaN or infinity."""
    if isinstance(value, str):
        return True  # Serialized NaN or inf values
    return np.isnan(value) if isinstance(value, (int, float)) else True

def calc_mean_std(values):
    """Calculate mean and standard deviation."""
    return {
        "mean": np.mean(values),
        "std": np.std(values)
    }

def get_y_dependent_nodes(model, data, y_node, dag):
    """
    Extract nodes that are not independent of Y based on correlation tests.
    
    Args:
        model: CAREFL model
        data: Dataset to analyze
        y_node: Index of the Y variable
        dag: DAG structure
        
    Returns:
        Dictionary containing dependent node indices and their correlation values
    """
    # Extract noise values using the model
    with torch.no_grad():
        z_values = model.forward_structural_only(data)
    
    # Extract Y node's noise
    y_noise = z_values[:, y_node]
    
    # Dictionary to store dependent nodes and their correlation values
    results = {
        "node_indices": [],
        "correlation_values": {}
    }
    
    # Check correlation with each variable
    n_vars = data.shape[1]
    
    for i in range(n_vars):
        if i == y_node:
            continue  # Skip self-comparison
        
        x_values = data[:, i]
        
        # Use the centralized correlation testing function
        corr_results = calculate_correlation_tests(y_noise, x_values)
        
        # Add to dependent nodes if correlation is significant
        if corr_results['any_significant']:
            results["node_indices"].append(i)
            results["correlation_values"][i] = {
                "spearman": {
                    "correlation": float(corr_results['spearman']['correlation']),
                    "p_value": float(corr_results['spearman']['p_value']),
                    "significant": bool(corr_results['spearman']['significant'])
                }
            }
            
    return results

def setup_gpu(gpu_id):
    """Set up GPU device based on provided ID."""
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # Use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    return device

def create_results_dir(output_dir, model_name):
    """Create and return results directory path."""
    if output_dir:
        main_results_dir = output_dir
        print(f"Using provided output directory: {main_results_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_results_dir = f"results/{model_name}_{timestamp}"
        print(f"Using auto-generated output directory: {main_results_dir}")

    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)
    
    return main_results_dir

def parse_dataset_indices(indices_arg, available_indices):
    """Parse dataset indices from command line argument."""
    if not indices_arg:
        return available_indices
        
    if '-' in indices_arg:
        # Range format: "0-5"
        start, end = map(int, indices_arg.split('-'))
        selected_indices = list(range(start, end + 1))
    elif ',' in indices_arg:
        # Comma-separated format: "0,3,5"
        selected_indices = [int(idx) for idx in indices_arg.split(',')]
    else:
        # Single number format: "5"
        selected_indices = [int(indices_arg)]
    
    # Filter to only include available indices
    return [idx for idx in selected_indices if idx in available_indices]

def dict_to_namespace(d):
    """
    Convert a dictionary to a namespace recursively
    
    Parameters:
    -----------
    d : dict
        Dictionary to convert
    
    Returns:
    --------
    SimpleNamespace
        Namespace created from the dictionary
    """
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
    
    Returns:
    --------
    SimpleNamespace
        Configuration as a namespace
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return dict_to_namespace(config)

def default_config():
    """
    Create a default configuration for CAREFL model
    
    Returns:
    --------
    SimpleNamespace
        Default configuration
    """
    config = {
        'flow': {
            'nl': 3,               # number of layers
            'nh': 16,              # hidden units
            'scale': True,         # use scaling in affine coupling layers
            'scale_base': True,    # learn base scale parameter
            'shift_base': True     # learn base shift parameter
        },
        'training': {
            'batch_size': 128,     # batch size
            'epochs': 200,         # training epochs
            'seed': 42,            # random seed
            'split': 0.8,          # train/test split
            'verbose': True        # verbose output
        },
        'optim': {
            'lr': 1e-2,            # learning rate
            'weight_decay': 1e-5,  # weight decay
            'beta1': 0.9,          # beta1 for Adam
            'amsgrad': True,       # use amsgrad variant
            'scheduler': True      # use LR scheduler
        }
    }
    return dict_to_namespace(config) 

def dict_to_namespace(d):
    """Convert dictionary to namespace recursively"""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def calculate_correlation_tests(x, y, alpha=0.05, kernel_width=None):
    """
    Calculate correlation between x and y using Spearman rank correlation and run independence test
    
    Args:
        x: First variable (numpy array or torch tensor)
        y: Second variable (numpy array or torch tensor)
        alpha: Significance level for hypothesis testing
        kernel_width: Width parameter for RBF kernel in HSIC (None for median heuristic)
        
    Returns:
        dict: Dictionary containing correlation results
    """
    # Convert torch tensors to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # Ensure 1D arrays
    x = x.flatten()
    y = y.flatten()
    
    # Spearman correlation (monotonic dependency)
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
        
    # Determine if dependency is significant
    return {
        'spearman': {
            'correlation': spearman_corr,
            'p_value': spearman_p,
            'significant': spearman_p < alpha
        },
        'any_significant': spearman_p < alpha
    }

def spearman_corr_sq(x, y, eps=1e-8):
    """
    Calculate squared Spearman rank correlation (for torch tensors)
    
    Args:
        x: First variable (torch tensor)
        y: Second variable (torch tensor)
        eps: Small value for numerical stability
        
    Returns:
        torch.Tensor: Squared Spearman correlation
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
        
    # Get the ranks
    x_rank = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    y_rank = torch.argsort(torch.argsort(y, dim=0), dim=0).float()
    
    # Normalize to [0,1]
    x_rank = x_rank / (x_rank.max() + eps)
    y_rank = y_rank / (y_rank.max() + eps)
    
    # Calculate correlation directly using the ranks
    mean_x = torch.mean(x_rank)
    mean_y = torch.mean(y_rank)
    xm = x_rank - mean_x
    ym = y_rank - mean_y
    cov = torch.mean(xm * ym)
    std_x = torch.sqrt(torch.mean(xm**2) + eps)
    std_y = torch.sqrt(torch.mean(ym**2) + eps)
    corr = cov / (std_x * std_y)
    # Clamp correlation to avoid potential numerical issues
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr**2

def visualize_data_distributions(data1, data2, gen):
    """
    Visualize the distributions of data from two environments and their statistical comparisons.
    
    Parameters:
    -----------
    data1 : np.ndarray
        Data from environment 1
    data2 : np.ndarray
        Data from environment 2
    gen : DataGenerator
        Generator object containing information about shifted nodes and Y-node
    
    Returns:
    --------
    tuple
        Shifted nodes, node order by importance, and node importance dictionary
    """
    # Visualize distribution of each node
    plt.figure(figsize=(15, 10))

    # Compare data distributions between environment 1 and environment 2
    for i in range(data1.shape[1]):
        plt.subplot(3, 4, i+1)
        sns.kdeplot(data=data1[:, i], label=f'Env1 X{i}', color='blue')
        sns.kdeplot(data=data2[:, i], label=f'Env2 X{i}', color='red')
        plt.title(f'Node X{i} Distribution')
        
        # Mark if this node is a shifted node
        if i in gen.shifted_nodes:
            plt.axvline(x=0, color='green', linestyle='--', alpha=0.7)
            if i == gen.y_node:
                plt.text(0.05, 0.95, f'Y-node\nShift: {gen.shift_type[i]}', 
                        transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            else:
                plt.text(0.05, 0.95, f'Shifted\nType: {gen.shift_type[i]}', 
                        transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Statistical comparison between environment 1 and environment 2 - separate graph
    plt.figure(figsize=(15, 6))
    means_env1 = np.mean(data1, axis=0)
    means_env2 = np.mean(data2, axis=0)
    vars_env1 = np.var(data1, axis=0)
    vars_env2 = np.var(data2, axis=0)

    x = np.arange(data1.shape[1])
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()

    # Mean bar chart
    bars1 = ax1.bar(x - width/2, means_env1, width, label='Env1 Mean', color='blue', alpha=0.6)
    bars2 = ax1.bar(x + width/2, means_env2, width, label='Env2 Mean', color='red', alpha=0.6)
    ax1.set_ylabel('Mean')
    ax1.set_title('Statistical Comparison Between Environments')

    # Variance line chart
    line1 = ax2.plot(x - width/2, vars_env1, 'o-', label='Env1 Variance', color='darkblue')
    line2 = ax2.plot(x + width/2, vars_env2, 'o-', label='Env2 Variance', color='darkred')
    ax2.set_ylabel('Variance')

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'X{i}' for i in range(data1.shape[1])])

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Mark shifted nodes
    for i in gen.shifted_nodes:
        ax1.axvline(x=i, color='green', linestyle='--', alpha=0.3)
        ax1.text(i, ax1.get_ylim()[1]*0.9, f'Shifted', rotation=90, color='green')

    plt.tight_layout()
    plt.show()

    # Print statistical table
    print("\nStatistical Comparison:")
    print("Node | Mean(Env1) | Mean(Env2) | Mean Diff | Var(Env1) | Var(Env2) | Var Ratio")
    print("-" * 75)
    for i in range(data1.shape[1]):
        mean_diff = abs(means_env2[i] - means_env1[i])
        var_ratio = vars_env2[i] / vars_env1[i] if vars_env1[i] > 0 else float('inf')
        node_status = "SHIFTED" if i in gen.shifted_nodes else ""
        print(f"X{i:<3} | {means_env1[i]:9.3f} | {means_env2[i]:9.3f} | {mean_diff:9.3f} | {vars_env1[i]:8.3f} | {vars_env2[i]:8.3f} | {var_ratio:8.3f} {node_status}")
    
    # Return the same values as the original code would have
    return gen.shifted_nodes, list(range(data1.shape[1])), {i: np.array(mean_diff) for i, mean_diff in enumerate(abs(means_env2 - means_env1))}

class CustomSyntheticDatasetDensity(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }

