import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def run_linearccp_analysis(data_env1, data_env2, adj, metadata):
    """
    Run LinearCCP analysis to detect shifted nodes using linear regression.
    
    Args:
        data_env1: Data from environment 1
        data_env2: Data from environment 2
        adj: Adjacency matrix representing the DAG
        metadata: Metadata containing true shifted nodes information
    
    Returns:
        dict: Results containing detected shifted nodes and analysis details
    """
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
        X_env1, y_env1 = data_env1[:, parents], data_env1[:, node]
        model_env1 = LinearRegression()
        model_env1.fit(X_env1, y_env1)
        beta_env1 = model_env1.coef_
        residuals_env1 = y_env1 - model_env1.predict(X_env1)
        
        # Regression analysis in environment 2
        X_env2, y_env2 = data_env2[:, parents], data_env2[:, node]
        model_env2 = LinearRegression()
        model_env2.fit(X_env2, y_env2)
        beta_env2 = model_env2.coef_
        residuals_env2 = y_env2 - model_env2.predict(X_env2)
        
        # Test for coefficient differences (Chow test)
        X_combined, y_combined = np.vstack((X_env1, X_env2)), np.concatenate((y_env1, y_env2))
        env_dummy = np.concatenate((np.zeros(len(y_env1)), np.ones(len(y_env2))))
        interaction_terms = np.zeros((len(env_dummy), len(parents)))
        for i, parent_idx in enumerate(parents):
            interaction_terms[:, i] = env_dummy * X_combined[:, i]        
        X_extended = np.hstack((X_combined, env_dummy.reshape(-1, 1), interaction_terms))
        extended_model = LinearRegression()
        extended_model.fit(X_extended, y_combined)
    
        X_restricted = np.hstack((X_combined, env_dummy.reshape(-1, 1)))
        restricted_model = LinearRegression()
        restricted_model.fit(X_restricted, y_combined)
        
        y_pred_restricted = restricted_model.predict(X_restricted)
        y_pred_extended = extended_model.predict(X_extended)
        
        ssr_restricted = np.sum((y_combined - y_pred_restricted) ** 2)
        ssr_extended = np.sum((y_combined - y_pred_extended) ** 2)
        
        df1, df2 = len(parents), len(y_combined) - len(extended_model.coef_) - 1
        f_stat = ((ssr_restricted - ssr_extended) / df1) / (ssr_extended / df2)
        p_value_chow = 1 - stats.f.cdf(f_stat, df1, df2)
        
        function_changed = p_value_chow < 0.05
        
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
    return results