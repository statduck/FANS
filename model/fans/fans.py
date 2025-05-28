import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from utils import calculate_correlation_tests

def decompose_probability(model, data, dag, reference_quantiles=None, quantiles=None):
    """
    Calculate probabilities using the updated approach with z-space manipulations
    
    Args:
        model: Trained CAREFL model
        data: Data for probability calculation
        dag: Adjacency matrix representing causal relationships
        reference_quantiles: Reference quantile values to use (if None, calculate from data)
        quantiles: List of quantile values (uses default values if None)
    
    Returns:
        dict: Sampled target node values for each node with parent nodes
    """
    # Convert data to tensor if needed
    if not isinstance(data, torch.Tensor):
        tensor_data = torch.tensor(data.astype(np.float32)).to(model.device)
    else:
        tensor_data = data.to(model.device)
    
    # Get latent representations of original data
    with torch.no_grad():
        z_orig = torch.tensor(model._forward_flow(data.astype(np.float32) if isinstance(data, np.ndarray) else data.cpu().numpy())).to(model.device)
    
    n_vars = data.shape[1]
    sampled_from_cond = {}
    
    # Use provided reference quantiles or calculate from data
    if reference_quantiles is not None:
        quantiles_values = reference_quantiles
    else:
        if quantiles is None:
            # Default quantile values changed to 0.25, 0.5, 0.75
            quantiles = [0.25, 0.5, 0.75]
        quantiles_values = np.quantile(data, quantiles, axis=0)
    
    for i in range(n_vars):
        parents = np.where(dag[:, i] == 1)[0]
        
        if len(parents) > 0:  # Only process nodes with parents
            # Dictionary to store results for each quantile
            quantile_results = {}
            
            # Calculate for each quantile
            for q_idx, q_value in enumerate(quantiles):
                q_values = quantiles_values[q_idx]
                fixed_data = data.copy()
                
                # (1) First generate z from original data (already done with z_orig)
                orig_z = z_orig.clone()
                
                # Set all parent nodes to current quantile values
                for p in parents:
                    fixed_data[:, p] = q_values[p]
                
                # (2) Generate z with fixed parent nodes
                with torch.no_grad():
                    fixed_z = torch.tensor(model._forward_flow(fixed_data)).to(model.device)
                
                # (3) Overwrite all values in original z except target node with fixed_z
                combined_z = orig_z.clone()
                for j in range(n_vars):
                    if j != i:  # Only overwrite non-target nodes
                        combined_z[:, j] = fixed_z[:, j]
                
                # (4) Generate x from combined_z and extract target node
                with torch.no_grad():
                    sampled_x = torch.tensor(model._backward_flow(combined_z.cpu().numpy())).to(model.device)
                    target_values = sampled_x[:, i].cpu().numpy()
                
                # (5) Store results for current quantile
                quantile_results[f'q{q_idx}'] = {
                    'sampled_values': target_values,
                    'quantile_value': q_value
                }
            
            # Store all quantile results
            sampled_from_cond[f'X{i}|pa(X{i})'] = {
                'original_values': data[:, i].copy(),
                'quantile_results': quantile_results,
                'node_idx': i,
                'quantiles': quantiles
            }
    
    return sampled_from_cond

def compare_conditional_probs(model, data1, data2, dag, env1_name="Env1", env2_name="Env2", js_threshold=0.1, visualize=True):
    """
    Compare sampled distributions between two environments for each quantile
    
    Args:
        model: Trained CAREFL model
        data1: Environment 1 data (reference distribution)
        data2: Environment 2 data (target distribution)
        dag: Causal graph structure
        env1_name: Environment 1 name
        env2_name: Environment 2 name
        js_threshold: Shift detection threshold
        visualize: Whether to visualize the results
        
    Returns:
        dict: Comparison results and list of detected changed nodes
    """
    
    # Set model to evaluation mode
    if hasattr(model, 'flow') and model.flow is not None:
        model.flow.eval()

    # Change quantiles to 0.25, 0.5, 0.75
    quantiles = [0.25, 0.5, 0.75]
    env1_quantile_values = np.quantile(data1, quantiles, axis=0)
    
    # Generate samples for each environment using env1's quantiles for both
    env1_samples = decompose_probability(model, data1, dag, reference_quantiles=env1_quantile_values, quantiles=quantiles)
    env2_samples = decompose_probability(model, data2, dag, reference_quantiles=env1_quantile_values, quantiles=quantiles)

    # Verify sample lengths and contents
    for key in env1_samples.keys():
        print(f"Verification for {key}:")
        for q_idx, q in enumerate(quantiles):
            q_key = f'q{q_idx}'
            if q_key in env1_samples[key]['quantile_results']:
                env1_q_samples = env1_samples[key]['quantile_results'][q_key]['sampled_values']
                env2_q_samples = env2_samples[key]['quantile_results'][q_key]['sampled_values']
                
                # Also print quantile value
                q_val = env1_samples[key]['quantile_results'][q_key]['quantile_value']
                print(f"  Quantile {q_idx} (p={q_val:.2f}) - Env1 samples: {len(env1_q_samples)}, "
                      f"range: [{env1_q_samples.min():.4f}, {env1_q_samples.max():.4f}]")
                print(f"  Quantile {q_idx} (p={q_val:.2f}) - Env2 samples: {len(env2_q_samples)}, "
                      f"range: [{env2_q_samples.min():.4f}, {env2_q_samples.max():.4f}]")
        print()

    results = {}
    
    if visualize:
        # Create subplots for each node by quantile
        for key in env1_samples.keys():
            node_idx = env1_samples[key]['node_idx']
            
            # Calculate subplot layout
            n_quantiles = len(quantiles)
            n_cols = min(2, n_quantiles)
            n_rows = math.ceil(n_quantiles / n_cols)
            
            plt.figure(figsize=(n_cols*6, n_rows*4))
            plt.suptitle(f'Quantile-based Distribution Comparison for {key}: {env1_name} vs {env2_name}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Create separate subplot for each quantile
            quantile_js_values = []
            
            for q_idx in range(n_quantiles):
                q_key = f'q{q_idx}'
                
                # Extract samples for each quantile
                samples1 = env1_samples[key]['quantile_results'][q_key]['sampled_values']
                samples2 = env2_samples[key]['quantile_results'][q_key]['sampled_values']
                
                # Estimate distribution using KDE
                kde_samples1 = gaussian_kde(samples1, bw_method='scott')
                kde_samples2 = gaussian_kde(samples2, bw_method='scott')
                
                # Set evaluation range
                min_val = min(np.min(samples1), np.min(samples2))
                max_val = max(np.max(samples1), np.max(samples2))
                eval_range = np.linspace(min_val, max_val, 1000)
                
                # Calculate PDF using KDE
                kde_pdf_samples1 = kde_samples1(eval_range)
                kde_pdf_samples2 = kde_samples2(eval_range)
                
                # Normalize PDFs
                kde_pdf_norm_samples1 = kde_pdf_samples1 / np.sum(kde_pdf_samples1)
                kde_pdf_norm_samples2 = kde_pdf_samples2 / np.sum(kde_pdf_samples2)
                
                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(kde_pdf_norm_samples1, kde_pdf_norm_samples2)
                quantile_js_values.append(js_div)
                
                # Visualize KDE for each quantile
                plt.subplot(n_rows, n_cols, q_idx + 1)
                plt.plot(eval_range, kde_pdf_norm_samples1, 'b-', label=f'{env1_name}')
                plt.plot(eval_range, kde_pdf_norm_samples2, 'r-', label=f'{env2_name}')
                
                plt.title(f'Quantile {q_idx} (p={quantiles[q_idx]:.2f}): JS={js_div:.4f}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                # Highlight background if threshold exceeded
                if js_div > js_threshold:
                    plt.axvspan(min_val, max_val, alpha=0.1, color='red')
            
            # Calculate average JS divergence
            avg_js = np.mean(quantile_js_values)
            
            # Store results
            results[key] = {
                'quantile_js_values': quantile_js_values,
                'avg_js_divergence': avg_js,
                'shift_detected': avg_js > js_threshold,
                'node_idx': node_idx
            }
            
            # Add text with average JS info
            plt.figtext(0.5, 0.01, f'Average JS Divergence: {avg_js:.4f} (Threshold: {js_threshold})', 
                       ha='center', fontsize=12, 
                       bbox={'facecolor': 'orange' if avg_js > js_threshold else 'lightgreen', 
                             'alpha': 0.2, 'pad': 5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.show()
    
    # Identify nodes with detected shifts
    shifted_nodes = [results[key]['node_idx'] for key in results.keys() 
                    if 'shift_detected' in results[key] and results[key]['shift_detected']]    

    results['shifted_nodes'] = shifted_nodes

    # Create summary table
    if len(results) > 0:
        summary_data = []
        for key in [k for k in results.keys() if isinstance(k, str) and '|' in k]:
            # Create string with JS values for each quantile
            quantile_js_str = ", ".join([f"Q{i}={js:.4f}" for i, js in 
                                        enumerate(results[key]['quantile_js_values'])])
            
            summary_data.append([
                key, 
                f"{results[key]['avg_js_divergence']:.4f}",
                quantile_js_str,
                results[key]['shift_detected']
            ])
        
        if summary_data:
            print("\nQuantile-based Distribution Shift Summary:")
            print("Variable | Avg JS Divergence | Quantile JS Values | Shift Detected")
            print("-" * 70)
            for row in summary_data:
                print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    
    return results

def test_node_noise_independence(model, data, node_idx, dag):
    """
    Test independence between a node's noise and its parent variables
    
    Args:
        model: CAREFL model
        data: Data
        node_idx: Node index to test
        dag: Graph structure
    
    Returns:
        dict: Independence test results (including ratio of dependent parent nodes)
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Find node's parents
    parents = np.where(dag[:, node_idx] == 1)[0]
    
    # No test needed for exogenous variables (nodes without parents)
    if len(parents) == 0:
        return {
            'node_idx': node_idx,
            'parent_count': 0,
            'dependent_parents': [],
            'independent_parents': [],
            'dependent_ratio': 0.0,
            'parent_independence_results': []
        }
    
    # Calculate noise (exogenous variable) for the specific node
    with torch.no_grad():
        z_values = model._forward_flow(data)
    
    node_noise = z_values[:, node_idx]
    independence_results = []
    dependent_parents = []
    independent_parents = []
    
    # Test independence with each parent variable
    for parent_idx in parents:
        parent_values = data[:, parent_idx]
        
        # Run independence tests
        corr_results = calculate_correlation_tests(node_noise, parent_values)
        
        # Classify based on independence
        if corr_results['any_significant']:
            dependent_parents.append(parent_idx)
        else:
            independent_parents.append(parent_idx)
        
        # Store results
        independence_results.append({
            'parent_idx': parent_idx,
            'spearman': corr_results['spearman'],
            'is_dependent': corr_results['any_significant']
        })
    
    # Calculate dependency ratio
    dependent_ratio = len(dependent_parents) / len(parents) if parents.size > 0 else 0.0
    
    return {
        'node_idx': node_idx,
        'parent_count': len(parents),
        'dependent_parents': dependent_parents,
        'independent_parents': independent_parents,
        'dependent_ratio': dependent_ratio,
        'parent_independence_results': independence_results
    }

def test_shifted_nodes_independence(model, data, shifted_nodes, dag, verbose=False):
    """
    Perform independence tests between noise and parent variables for all shifted nodes
    
    Args:
        model: CAREFL model
        data: Data
        shifted_nodes: List of nodes with detected changes
        dag: Graph structure
        verbose: Whether to print detailed output
    
    Returns:
        dict: Independence test results for each shifted node
    """
    results = {}
    
    for node_idx in shifted_nodes:
        # Test independence for each node
        node_result = test_node_noise_independence(model, data, node_idx, dag)
        results[node_idx] = node_result
        
        if verbose:
            print(f"\n=== Node {node_idx} Independence Test Results ===")
            if node_result['parent_count'] == 0:
                print(f"Node {node_idx} is an exogenous variable with no parents.")
            else:
                print(f"Number of parent variables: {node_result['parent_count']}")
                print(f"Parent variables with dependency: {node_result['dependent_parents']}")
                print(f"Parent variables without dependency: {node_result['independent_parents']}")
                print(f"Dependency ratio: {node_result['dependent_ratio']:.2f}")
    
    # Calculate average dependency ratio for all shifted nodes
    avg_dependent_ratio = sum(results[node_idx]['dependent_ratio'] for node_idx in shifted_nodes) / len(shifted_nodes) if shifted_nodes else 0.0
    
    summary = {
        'node_results': results,
        'avg_dependent_ratio': avg_dependent_ratio,
        'node_count': len(shifted_nodes)
    }
    
    if verbose:
        print(f"\n=== Overall Summary ===")
        print(f"Number of shifted nodes tested: {len(shifted_nodes)}")
        print(f"Average dependency ratio: {avg_dependent_ratio:.2f}")
    
    return summary

def calc_log_likelihood(model, data):
    """
    Calculate log likelihood of data given the model
    
    Args:
        model: CAREFL model
        data: Tensor data
        
    Returns:
        float: Average log likelihood
    """
    device = model.device
    
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.tensor(data.astype(np.float32)).to(device)
    else:
        data = data.to(device)
    
    with torch.no_grad():
        model.flow.eval()  # Ensure model is in evaluation mode
        _, prior_logprob, log_det = model.flow(data)
        log_likelihood = prior_logprob + log_det
        return torch.mean(log_likelihood).item()
