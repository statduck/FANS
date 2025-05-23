import pandas as pd
from kneed import KneeLocator
import numpy as np
import torch
from typing import Union, Tuple
import torch
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy import stats
from typing import List, Set, Dict
from typing import Optional
from scipy import stats
import warnings
import patsy

# Setup GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_device(new_device):
    """Set the global device for iscan module."""
    global device
    device = new_device
    print(f"ISCAN using device: {device}")

def stein_hess(X: torch.Tensor, eta_G: float, eta_H: float, s: Optional[float] = None) -> torch.Tensor:
    X = X.to(device)
    n, d = X.shape
    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s ** 2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s ** 2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n, device=device)), nablaK)

    nabla2K = torch.einsum('kij,ik->kj', -1 / s ** 2 + X_diff ** 2 / s ** 4, K)
    return -G ** 2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n, device=device)), nabla2K)

def _find_elbow(diff_dict: dict, hard_thres: float = 30, online: bool = True) -> np.ndarray:
    r"""
    Return selected shifted nodes by finding elbow point on sorted variance

    Parameters
    ----------
    diff_dict : dict
        A dictionary where ``key`` is the index of variables/nodes, and ``value`` is its variance ratio.
    hard_thres : float, optional
        | Variance ratios larger than hard_thres will be directly regarded as shifted. 
        | Selected nodes in this step will not participate in elbow method, by default 30.
    online : bool, optional
        If ``True``, the heuristic will find a more aggressive elbow point, by default ``True``.

    Returns
    -------
    np.ndarray
        A dict with selected nodes and corresponding variance
    """
    diff = pd.DataFrame()
    diff.index = diff_dict.keys()
    diff["ratio"] = [x for x in diff_dict.values()]
    shift_node_part1 = diff[diff["ratio"] >= hard_thres].index
    undecide_diff = diff[diff["ratio"] < hard_thres]
    kn = KneeLocator(range(undecide_diff.shape[0]), undecide_diff["ratio"].values,
                     curve='convex', direction='decreasing',online=online,interp_method="interp1d")
    shift_node_part2 = undecide_diff.index[:kn.knee]
    shift_node = np.concatenate((shift_node_part1,shift_node_part2))
    return shift_node

def _get_min_rank_sum(HX: torch.Tensor, HY: torch.Tensor) -> int:
    order_X = torch.argsort(HX.var(axis=0))
    rank_X = torch.argsort(order_X)

    order_Y = torch.argsort(HY.var(axis=0))
    rank_Y = torch.argsort(order_Y)
    l = int((rank_X + rank_Y).argmin().cpu().item())
    return l

def est_node_shifts(
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor], 
        eta_G: float = 0.001, 
        eta_H: float = 0.001,
        normalize_var: bool = False, 
        shifted_node_thres: float = 2.,
        elbow: bool = False,
        elbow_thres: float = 30.,
        elbow_online: bool = True,
        use_both_rank: bool = True,
        verbose: bool = False,
    ) -> Tuple[list, list, dict]:
    r"""
    | Implementation of the iSCAN method of Chen et al. (2023).
    | Returns an estimated topological ordering, and estimated shifted nodes
    
    References
    ----------
    - T. Chen, K. Bello, B. Aragam, P. Ravikumar. (2023). `iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models. <https://arxiv.org/abs/2306.17361>`_
    
    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
        Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples, 
        and :math:`d` is the number of variables/nodes.
    Y : Union[np.ndarray, torch.Tensor]
        Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples, 
        and :math:`d` is the number of variables/nodes.
    eta_G : float, optional
        hyperparameter for the score's Jacobian estimation, by default 0.001.
    eta_H : float, optional
        hyperparameter for the score's Jacobian estimation, by default 0.001.
    normalize_var : bool, optional
        If ``True``, the Hessian's diagonal is normalized by the expected value, by default ``False``.
    shifted_node_thres : float, optional
        Threshold to decide whether or not a variable has a distribution shift, by default 2.
    elbow : bool, optional
        If ``True``, iscan uses the elbow heuristic to determine shifted nodes. By default ``True``.
    elbow_thres : float, optional
        If using the elbow method, ``elbow_thres`` is the ``hard_thres`` in 
        :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default 30.
    elbow_online : bool, optional
        If using the elbow method, ``elbow_online`` is ``online`` in 
        :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default ``True``.
    use_both_rank : bool, optional
        estimate topo order by X's and Y's rank sum. If False, only use X for topo order, by default ``False``.
    verbose : bool, optional
        If ``True``, prints to stdout the variances of the Hessian entries for the running leafs.
        By default ``False``.
    Returns
    -------
    Tuple[list, list, dict]
        estimated shifted nodes, topological order, and dict of variance ratios for all nodes.
    """
    vprint = print if verbose else lambda *a, **k: None

    # Convert to tensor and move to GPU
    if type(X) is np.ndarray:
        X = torch.from_numpy(X).to(device)
    else:
        X = X.to(device)
        
    if type(Y) is np.ndarray:
        Y = torch.from_numpy(Y).to(device)
    else:
        Y = Y.to(device)
    
    n, d = X.shape
    order = [] # estimates a valid topological sort
    active_nodes = list(range(d))
    shifted_nodes = [] # list of estimated shifted nodes
    dict_stats = dict() # dictionary of variance ratios for all nodes

    for _ in range(d - 1):
        A = torch.concat((X, Y))
        HX = stein_hess(X, eta_G, eta_H)
        HY = stein_hess(Y, eta_G, eta_H)
        if normalize_var:
            HX = HX / HX.mean(axis=0)
            HY = HY / HY.mean(axis=0)
        
        # estimate common leaf
        l = _get_min_rank_sum(HX,HY) if use_both_rank else int(HX.var(axis=0).argmin())

        # compute sample variances for the leaf node
        HX = HX.var(axis=0)[l]
        HY = HY.var(axis=0)[l]

        HA = stein_hess(A, eta_G, eta_H).var(axis=0)[l] # TODO: compute this more efficiently

        vprint(f"l: {active_nodes[l]} Var_X = {HX} Var_Y = {HY} Var_Pool = {HA}")
        # store shifted nodes based on threshold and variance ratio
        if torch.min(HX, HY) * shifted_node_thres < HA:
            shifted_nodes.append(active_nodes[l])

        dict_stats[active_nodes[l]] = (HA / torch.min(HX, HY)).cpu().numpy()
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:, 0:l], X[:, l + 1:]])
        Y = torch.hstack([Y[:, 0:l], Y[:, l + 1:]])
    order.append(active_nodes[0])
    order.reverse()
    dict_stats = dict(sorted(dict_stats.items(), key=lambda item: -item[1]))
    
    if elbow:
        # if using the elbow heuristic, replace the list of shifted nodes
        shifted_nodes = _find_elbow(dict_stats, elbow_thres, elbow_online)
        
    return shifted_nodes, order, dict_stats

def _fit_ols_regression(
    target_data: np.ndarray,
    feature_data: np.ndarray,
    feature_names: List[str]
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fits an OLS regression model using statsmodels.

    Args:
        target_data: Target variable data (1D array).
        feature_data: Feature matrix including polynomial features and a constant.
        feature_names: Names corresponding to columns in feature_data.

    Returns:
        Fitted OLS model results object, or None if fitting fails.
    """
    if feature_data.shape[0] <= feature_data.shape[1]:
        warnings.warn(f"Not enough samples ({feature_data.shape[0]}) relative to features ({feature_data.shape[1]}) for stable regression. Skipping.", UserWarning)
        return None
    if np.linalg.matrix_rank(feature_data) < feature_data.shape[1]:
        warnings.warn(f"Collinearity detected (rank < num_features). Regression might be unstable. Skipping.", UserWarning)
        # Note: statsmodels might handle perfect collinearity by dropping terms,
        # but explicitly checking rank is safer.
        return None

    try:
        # Ensure feature_data is float type for OLS
        feature_data = feature_data.astype(float)
        ols_model = sm.OLS(target_data.astype(float), feature_data, hasconst=True) # Assuming constant is included
        results = ols_model.fit()

        # Check for issues in fitting results if needed (e.g., condition number)
        # if results.condition_number > 1000: # Example threshold
        #     warnings.warn(f"High condition number ({results.condition_number:.2f}) detected. Regression results might be sensitive.", UserWarning)

        return results
    except Exception as e:
        warnings.warn(f"Error during OLS fitting: {e}", UserWarning)
        return None

def _test_vector_coefficient_equality(
    coef1_vector: np.ndarray, 
    cov1_matrix: np.ndarray,
    coef2_vector: np.ndarray, 
    cov2_matrix: np.ndarray,
    n1: int,
    n2: int
) -> float:
    """
    Performs a multivariate test for the equality of two coefficient vectors.
    Uses Hotelling's T² test for the null hypothesis that two coefficient vectors are equal.

    Args:
        coef1_vector: Vector of coefficient estimates from environment 1.
        cov1_matrix: Covariance matrix of the estimates from environment 1.
        coef2_vector: Vector of coefficient estimates from environment 2.
        cov2_matrix: Covariance matrix of the estimates from environment 2.
        n1: Sample size from environment 1.
        n2: Sample size from environment 2.

    Returns:
        The p-value of the test. Returns 1.0 if the test cannot be performed.
    """
    # Input validation
    if (np.any(np.isnan(coef1_vector)) or np.any(np.isnan(coef2_vector)) or
        np.any(np.isnan(cov1_matrix)) or np.any(np.isnan(cov2_matrix))):
        warnings.warn("NaN values detected in inputs to vector coefficient test. Returning p=1.0.", UserWarning)
        return 1.0
        
    # Length of coefficient vectors
    p = len(coef1_vector)
    
    # Check for valid dimensions
    if cov1_matrix.shape != (p, p) or cov2_matrix.shape != (p, p):
        warnings.warn(f"Covariance matrix dimensions don't match coefficient vector length. Returning p=1.0.", UserWarning)
        return 1.0

    try:
        # Difference in coefficient vectors
        delta = coef1_vector - coef2_vector
        
        # Pooled covariance matrix (weighted by sample sizes)
        # In regression context, we can use the covariance matrices directly
        pooled_cov = cov1_matrix + cov2_matrix
        
        # Try to compute inverse - might fail if near-singular
        try:
            inv_pooled_cov = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            # If regular inverse fails, try pseudoinverse
            inv_pooled_cov = np.linalg.pinv(pooled_cov)
            warnings.warn("Using pseudoinverse for covariance matrix due to singularity issues.", UserWarning)
        
        # Compute Hotelling's T²
        t_squared = np.dot(np.dot(delta.T, inv_pooled_cov), delta)
        
        # Calculate degrees of freedom
        df1 = p  # Number of variables
        df2 = n1 + n2 - p - 1  # Degrees of freedom
        
        if df2 <= 0:
            warnings.warn(f"Insufficient degrees of freedom (df2={df2}). Returning p=1.0.", UserWarning)
            return 1.0
        
        # Convert to F-statistic
        f_stat = t_squared * df2 / (df1 * (n1 + n2 - 2))
        
        # Calculate p-value from F-distribution
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
        
        return p_value
    
    except Exception as e:
        warnings.warn(f"Error in multivariate coefficient test: {e}. Returning p=1.0.", UserWarning)
        return 1.0

# --- Main Function ---

def _get_basis_features(
    data: np.ndarray,
    parent_indices: List[int],
    basis_type: str = 'spline',
    poly_degree: int = 2,
    spline_df: Optional[int] = None
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Generates basis features (spline) for given parent nodes.

    Args:
        data: Full data matrix (samples x nodes).
        parent_indices: Indices of the parent nodes.
        basis_type: Type of basis (only 'spline' is supported now).
        poly_degree: Degree parameter (used to calculate default spline_df if not provided).
        spline_df: Degrees of freedom for each spline basis.
                   If None, a default (e.g., 4) will be used.

    Returns:
        Tuple containing (basis feature matrix, feature names), or None if issues arise.
    """
    if not parent_indices:
        warnings.warn("No parent indices provided, cannot generate basis features.", UserWarning)
        return None

    X_parents = data[:, parent_indices]
    if X_parents.ndim == 1:
         X_parents = X_parents[:, np.newaxis]

    parent_names = [f'x{i}' for i in parent_indices]
    data_dict = {name: X_parents[:, i] for i, name in enumerate(parent_names)}

    feature_matrix = None
    feature_names = []

                
    if spline_df is None:
        spline_df = max(4, poly_degree + 1)  # Default df if not specified
        warnings.warn(f"spline_df not specified, using default df={spline_df}", UserWarning)

    # Create patsy formula string for B-splines (bs) for each parent
    # Example: "bs(x0, df=4) + bs(x1, df=4)"
    terms = [f"bs({name}, df={spline_df}, degree=3)" for name in parent_names]  # Using cubic B-splines (degree=3)
    formula = " + ".join(terms) # intercept explicitly if needed (patsy adds one by default if formula doesn't remove it)

    # dmatrix creates the design matrix directly
    feature_matrix = patsy.dmatrix(formula, data_dict, return_type='dataframe')
    feature_names = feature_matrix.design_info.column_names
    feature_matrix = feature_matrix.values  # Convert to numpy array

    # Ensure the matrix is returned as numpy array
    if not isinstance(feature_matrix, np.ndarray):
            feature_matrix = np.asarray(feature_matrix)

    return feature_matrix, list(feature_names)

def find_functionally_shifted_edges_with_dag(
    y_node: int,
    parents_y: List[int],
    data_env1: Union[np.ndarray, torch.Tensor],
    data_env2: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.05,
    spline_df: Optional[int] = None
) -> Tuple[bool, Set[Tuple[int, int]], Dict]:
    """
    Tests whether the functional relationship between a specific Y node and its parents has shifted between environments.
    Uses multivariate hypothesis testing to compare coefficient vectors between environments.

    Args:
        y_node: Index of the target node (Y) whose functional relationship we want to test.
        shifted_parents: List of parent node indices for the Y node.
        data_env1: Data matrix (samples x nodes) for environment 1.
        data_env2: Data matrix (samples x nodes) for environment 2.
        alpha: Significance level for hypothesis testing coefficient equality.
        spline_df: Degrees of freedom for spline basis.

    Returns:
        A tuple containing:
        - bool: Whether a functional shift was detected (True if detected)
        - set of tuples: The functionally shifted edges (k, y_node) representing parent k -> y_node
        - dict: Additional test results including p-values and coefficients
    """
    # --- Input Validation and Data Conversion ---
    if type(data_env1) is torch.Tensor:
        data_env1 = data_env1.cpu().numpy()
    if type(data_env2) is torch.Tensor:
        data_env2 = data_env2.cpu().numpy()
    
    num_nodes = data_env1.shape[1]
    if y_node < 0 or y_node >= num_nodes:
        raise ValueError(f"y_node ({y_node}) must be within valid range [0, {num_nodes-1}]")
    
    # Use the provided parent nodes directly
    
    # Test result info
    functionally_shifted_edges = set()
    test_results = {
        "y_node": y_node,
        "parents": parents_y,
        "functional_shift_detected": False,
        "p_value_overall": None,
        "parent_specific_p_values": {},
        "coefficients_env1": {},
        "coefficients_env2": {}
    }
    
    # If Y has no parents, there's no functional relationship to test
    if not parents_y:
        warnings.warn(f"Node {y_node} has no parents. No functional relationship to test.")
        return False, functionally_shifted_edges, test_results
    
    # --- Generate basis features for Y's parents ---
    basis_info1 = _get_basis_features(
        data_env1, parents_y, basis_type='spline', spline_df=spline_df
    )
    basis_info2 = _get_basis_features(
        data_env2, parents_y, basis_type='spline', spline_df=spline_df
    )
    
    if basis_info1 is None or basis_info2 is None:
        warnings.warn(f"Could not generate basis features for node {y_node}. Skipping.")
        return False, functionally_shifted_edges, test_results
    
    X_basis1, feature_names1 = basis_info1
    X_basis2, feature_names2 = basis_info2
    
    if feature_names1 != feature_names2:
        warnings.warn(f"Internal error: Feature name mismatch between environments. "
                    f"Env1 features: {feature_names1}, Env2 features: {feature_names2}.")
        return False, functionally_shifted_edges, test_results
        
    if X_basis1.shape[1] == 0 or X_basis2.shape[1] == 0:
        warnings.warn(f"Basis feature matrix has zero columns for node {y_node}. Skipping.")
        return False, functionally_shifted_edges, test_results
    
    # --- Fit regression models ---
    target_y_env1 = data_env1[:, y_node]
    target_y_env2 = data_env2[:, y_node]
    
    # Pass feature names for potential use in OLS results indexing
    results1 = _fit_ols_regression(target_y_env1, X_basis1, feature_names1)
    results2 = _fit_ols_regression(target_y_env2, X_basis2, feature_names2)
    
    if results1 is None or results2 is None:
        return False, functionally_shifted_edges, test_results  # Warnings already issued
    
    # Get samples sizes from each environment
    n1 = X_basis1.shape[0]
    n2 = X_basis2.shape[0]
    
    # --- Step 1: Test for overall functional shift first ---
    # Extract coefficient vectors (excluding intercept for non-constant features)
    try:
        # Get all non-intercept coefficients (i.e., exclude the first coefficient if it's an intercept)
        if "1" in feature_names1 or "Intercept" in feature_names1:
            # Find the intercept index
            intercept_idx = feature_names1.index("1") if "1" in feature_names1 else feature_names1.index("Intercept")
            non_intercept_indices = [i for i in range(len(feature_names1)) if i != intercept_idx]
        else:
            # All features are non-intercept features
            non_intercept_indices = list(range(len(feature_names1)))
        
        coef1_vector = np.array([results1.params[i] for i in non_intercept_indices])
        coef2_vector = np.array([results2.params[i] for i in non_intercept_indices])
        
        # Get full covariance matrices from regression results
        cov1_full = results1.cov_params()
        cov2_full = results2.cov_params()
        
        # Select relevant submatrices
        if hasattr(cov1_full, 'iloc'):  # pandas DataFrame
            cov1_matrix = cov1_full.iloc[non_intercept_indices, non_intercept_indices].values
            cov2_matrix = cov2_full.iloc[non_intercept_indices, non_intercept_indices].values
        else:  # numpy array
            cov1_matrix = cov1_full[np.ix_(non_intercept_indices, non_intercept_indices)]
            cov2_matrix = cov2_full[np.ix_(non_intercept_indices, non_intercept_indices)]
        
        # Perform multivariate test on full coefficient vectors (overall test)
        overall_p_value = _test_vector_coefficient_equality(
            coef1_vector, cov1_matrix, coef2_vector, cov2_matrix, n1, n2
        )
        
        test_results["p_value_overall"] = overall_p_value
        test_results["functional_shift_detected"] = overall_p_value < alpha
        
        # Store model coefficients for both environments
        test_results["coefficients_env1"] = {name: float(val) for name, val in zip(feature_names1, results1.params)}
        test_results["coefficients_env2"] = {name: float(val) for name, val in zip(feature_names2, results2.params)}
        
        # --- Step 2: If overall shift detected, identify which specific parents contribute to the shift ---
        if overall_p_value < alpha:
            # Check each parent for contribution to the functional shift
            for k in parents_y:
                # Identify features related to parent k
                k_var_name = f'x{k}'  # Base variable name for parent k
                
                # Find terms generated by patsy for xk, e.g., 'bs(xk, df=4)[0]', 'bs(xk, df=4)[1]', ...
                k_related_features = [
                    feat for feat in feature_names1
                    if feat.startswith(f"bs({k_var_name},") or feat.startswith(f"ns({k_var_name},") or feat.startswith(f"cr({k_var_name},")
                ]
                
                if not k_related_features:
                    continue
                
                try:
                    # Extract coefficient vectors for parent k's features
                    coef1_indices = [feature_names1.index(feat) for feat in k_related_features]
                    coef2_indices = [feature_names2.index(feat) for feat in k_related_features]
                    
                    # Extract coefficient vectors
                    coef1_k_vector = np.array([results1.params[i] for i in coef1_indices])
                    coef2_k_vector = np.array([results2.params[i] for i in coef2_indices])
                    
                    # Extract covariance submatrices
                    if hasattr(cov1_full, 'iloc'):  # pandas DataFrame
                        cov1_k_matrix = cov1_full.iloc[coef1_indices, coef1_indices].values
                        cov2_k_matrix = cov2_full.iloc[coef2_indices, coef2_indices].values
                    else:  # numpy array
                        cov1_k_matrix = cov1_full[np.ix_(coef1_indices, coef1_indices)]
                        cov2_k_matrix = cov2_full[np.ix_(coef2_indices, coef2_indices)]
                    
                    # Test if parent k's specific contribution has shifted
                    k_p_value = _test_vector_coefficient_equality(
                        coef1_k_vector, cov1_k_matrix, coef2_k_vector, cov2_k_matrix, n1, n2
                    )
                    
                    # Record parent-specific p-value
                    test_results["parent_specific_p_values"][k] = k_p_value
                    
                    # If parent k has a significant shift, add to the set of shifted edges
                    if k_p_value < alpha:
                        functionally_shifted_edges.add((k, y_node))
                        
                except Exception as e:
                    warnings.warn(f"Error testing coefficients for edge {k}->{y_node}: {e}")
                    continue
        
    except Exception as e:
        warnings.warn(f"Error in overall functional shift test: {e}")
        return False, functionally_shifted_edges, test_results
    
    # Return whether a shift was detected, which specific edges shifted, and test details
    return test_results["functional_shift_detected"], functionally_shifted_edges, test_results

