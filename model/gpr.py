import numpy as np
import pandas as pd
import cupy as cp
from scipy.linalg import eigh  # More stable for symmetric matrices
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, norm
from scipy.optimize import minimize_scalar # For kernel parameter optimization
from math import lgamma # Log gamma function for multivariate gamma function needed for log likelihood
from sklearn.preprocessing import StandardScaler  # For standardizing X and Y data
from joblib import Parallel, delayed # For parallel processing

# Set random seeds for reproducibility
np.random.seed(42)
cp.random.seed(42)


# Helper function to calculate Gaussian Kernel matrix
def gaussian_kernel(X, length_scale=1.0):
    """
    Computes the Gaussian (RBF) kernel matrix.
    K(x, x') = exp(- ||x - x'||^2 / (2 * length_scale^2))
    Note: The paper uses K(x,x') = exp{-sum omega_k (x_k - x_k')^2}.
          This is equivalent to the RBF kernel with appropriate scaling.
          Here, for simplicity, we use a single length_scale (isotropic).
          1 / (2 * length_scale^2) corresponds to omega.
    """
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1)
    K = np.exp(-sq_dists / (2 * length_scale**2))
    return K


# Function to run EM algorithm for VCM I
def run_em_vcm1(Y, G1, max_iter=200, tol=1e-5, verbose=False):
    """
    Runs the EM algorithm to find MLEs for tau1_sq and tau2_sq in VCM I on GPU.
    Returns estimated tau1_sq, tau2_sq, and the maximized log-likelihood.
    """
    n = len(Y)
    Y_gpu = cp.asarray(Y.reshape(-1, 1))  # Ensure Y is column vector and move to GPU
    G1_gpu = cp.asarray(G1)  # Move G1 to GPU

    tau2_sq_t = cp.var(Y_gpu)
    if tau2_sq_t < 1e-6:  # Handle zero variance case
        tau2_sq_t = 1e-6

    tau1_sq_t = tau2_sq_t * 0.1  # Default initial guess for signal variance

    # Precompute eigen-decomposition
    try:
        # Add small jitter for stability if G1 is singular/near-singular
        jitter = 1e-6 * cp.mean(cp.diag(G1_gpu))
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(G1_gpu + cp.identity(n) * jitter)
        # Sort eigenvalues/vectors descending
        idx = cp.argsort(eigenvalues_gpu)[::-1]
        eigenvalues_gpu = eigenvalues_gpu[idx]
        eigenvectors_gpu = eigenvectors_gpu[:, idx]
        # Ensure eigenvalues are non-negative
        eigenvalues_gpu = cp.maximum(eigenvalues_gpu, 0)
        D_gpu = eigenvalues_gpu
        V_gpu = eigenvectors_gpu
        U_gpu = V_gpu.T @ Y_gpu  # This should result in a n x 1 vector
    except cp.linalg.LinAlgError:
        print("Eigen-decomposition failed.")
        return None, None, -cp.inf

    if verbose:
        print(f"Initial: tau1^2={tau1_sq_t:.4f}, tau2^2={tau2_sq_t:.4f}")

    for i in range(max_iter):
        # EM step
        tau1_sq_t = max(tau1_sq_t, 1e-8)
        tau2_sq_t = max(tau2_sq_t, 1e-8)

        D_diag_gpu = tau1_sq_t * D_gpu + tau2_sq_t
        if cp.any(D_diag_gpu <= 0):  # Check for invalid eigenvalues
            return None, None, -cp.inf

        # Update tau1_sq and tau2_sq
        tau1_sq_new = (1 / n) * cp.sum((D_gpu / D_diag_gpu**2) * (U_gpu.flatten()**2))
        tau2_sq_new = (1 / n) * cp.sum((1 / D_diag_gpu**2) * (U_gpu.flatten()**2))

        # Check for convergence
        delta = max(abs(tau1_sq_new - tau1_sq_t), abs(tau2_sq_new - tau2_sq_t))
        if verbose and (i % 20 == 0):
            print(f"Iter {i}: tau1^2={tau1_sq_new:.4f}, tau2^2={tau2_sq_new:.4f}, delta={delta:.6f}")

        if delta < tol:
            if verbose:
                print(f"EM converged after {i+1} iterations.")
            break

        tau1_sq_t, tau2_sq_t = tau1_sq_new, tau2_sq_new
    else:  # No break
        if verbose:
            print(f"EM did not converge after {max_iter} iterations.")

    # Calculate final log-likelihood
    D_diag_gpu = tau1_sq_t * D_gpu + tau2_sq_t
    log_det_Sigma = cp.sum(cp.log(D_diag_gpu))
    Y_SigmaInv_Y = cp.sum((U_gpu.flatten()**2) / D_diag_gpu)
    neg_ll = 0.5 * (log_det_Sigma + Y_SigmaInv_Y + n * cp.log(2 * cp.pi))

    return float(tau1_sq_t), float(tau2_sq_t), -float(neg_ll)  # Return positive log-likelihood

# Function to find the best kernel parameter (length_scale) using MLE under H0
def find_best_kernel_param(X, Y, initial_ls_range=(0.1, 10.0), verbose=False):
    """Finds the best Gaussian kernel length_scale by maximizing marginal likelihood under H0."""
    
    prev_tau1_sq = None
    prev_tau2_sq = None

    def objective(ls):
        nonlocal prev_tau1_sq, prev_tau2_sq
        if isinstance(ls, np.ndarray):
            ls = ls.item() if ls.size == 1 else float(ls)
            
        if ls <= 1e-4:  # Avoid too small length scales
            return np.inf
        if verbose:
            print(f"  Testing length_scale = {ls:.4f}")
        
        K_n = gaussian_kernel(X, length_scale=ls)
        n = K_n.shape[0]
        jitter = 1e-6 * np.mean(np.diag(K_n))
        K_n_stable = K_n + np.identity(n) * jitter
        
        # Run EM algorithm with previous initial values
        _, _, log_likelihood = run_em_vcm1(Y, K_n_stable, verbose=False)
        
        # Update initial values for the next iteration
        prev_tau1_sq, prev_tau2_sq = _, _
        
        if verbose:
            print(f"    -> Neg LogLikelihood = {-log_likelihood:.4f}")
        
        return -log_likelihood if log_likelihood > -np.inf else np.inf

    result = minimize_scalar(objective, bounds=initial_ls_range, method='bounded')

    if result.success:
        best_ls = result.x
        max_log_likelihood = -result.fun
        if verbose:
            print(f"Best length_scale found: {best_ls:.4f}, Max LogLik: {max_log_likelihood:.4f}")
        return best_ls, max_log_likelihood
    else:
        print("Kernel parameter optimization failed.")
        fallback_ls = np.mean(initial_ls_range)
        _, _, max_log_likelihood = run_em_vcm1(Y, gaussian_kernel(X, fallback_ls), verbose=False)
        return fallback_ls, max_log_likelihood


# Function to calculate permutation size b_n
def calculate_bn(n, eigenvalues, xi_n_hat, alpha=0.05, alpha0_factor=1e-4, threshold_factor=1e-3):
    """Calculates the permutation size b_n based on estimated parameters."""
    alpha0 = alpha0_factor * alpha
    threshold = threshold_factor * alpha

    best_bn = 0
    for bn_candidate in range(1, n): # Check bn from 1 to n-1
        idx_cn = n - bn_candidate # Index for (n-bn+1)th largest eigenvalue (0-based)
        if idx_cn < 0: continue

        cn_plus_1 = eigenvalues[idx_cn] # (n-bn+1)th largest eigenvalue

        # Calculate omega_tilde_hat
        omega_tilde_hat = xi_n_hat * cn_plus_1

        # Calculate chi2 quantile Q_{b_n}(1-alpha0)
        # Need try-except for potential issues with chi2.ppf (e.g., df=0 or invalid alpha0)
        try:
            # Ensure bn_candidate > 0 for chi2
            if bn_candidate <= 0: continue
            q_bn = chi2.ppf(1 - alpha0, df=bn_candidate)
            if not np.isfinite(q_bn) or q_bn < 0: continue # Check for valid quantile
        except ValueError:
            continue # Skip if quantile calculation fails

        # Calculate v_tilde_hat
        term_inside_exp = 0.5 * omega_tilde_hat * q_bn
        # Avoid potential overflow in exp
        if term_inside_exp > 50: # Heuristic limit
            v_tilde_hat = np.inf
        else:
             try:
                 v_tilde_hat = 0.5 * np.exp(term_inside_exp) - 0.5
             except OverflowError:
                 v_tilde_hat = np.inf

        # Check condition
        if v_tilde_hat + alpha0 <= threshold:
            best_bn = bn_candidate
        else:
            # Since v_tilde increases with bn (usually), we can stop early
            break

    # Ensure bn is at least 1 if possible, otherwise 0.
    if best_bn == 0 and n > 1:
       # Check if bn=1 satisfies condition, maybe threshold was too strict?
       # Recalculate for bn=1 explicitly
        idx_cn1 = n - 1
        cn1 = eigenvalues[idx_cn1]
        omega_tilde_hat1 = xi_n_hat * cn1
        try:
             q_bn1 = chi2.ppf(1-alpha0, df=1)
             if np.isfinite(q_bn1) and q_bn1 >= 0:
                 term_inside_exp1 = 0.5 * omega_tilde_hat1 * q_bn1
                 if term_inside_exp1 <= 50:
                     v_tilde_hat1 = 0.5 * np.exp(term_inside_exp1) - 0.5
                     if v_tilde_hat1 + alpha0 <= threshold:
                          best_bn = 1
                 else: # Exp overflow case for bn=1
                      pass
             else: # Invalid quantile case for bn=1
                  pass
        except ValueError:
             pass # Quantile calc fails for bn=1

    return best_bn


def partial_permutation_test(env1_data, env2_data, y_node_col, dag_adjacency_matrix,
                             num_permutations=1000, alpha=0.05, gamma=0.1,
                             length_scale_range=(0.1, 10.0),
                             test_statistic_type='pseudo', # 'pseudo', 'H1', 'H1_prime'
                             verbose=True):
    """
    Performs the Kernel-Based Partial Permutation Test.

    Args:
        env1_data (pd.DataFrame): Data from environment/group 1.
        env2_data (pd.DataFrame): Data from environment/group 2.
        y_node_col (str or int): Column name or index for the response variable Y.
        dag_adjacency_matrix (np.ndarray): Adjacency matrix M (M[j,i]=1 -> j is parent of i).
        num_permutations (int): Number of permutations (B) for p-value estimation.
        alpha (float): Significance level for choosing bn.
        gamma (float): Parameter for GPR variance scaling (e.g., 0 < gamma < 1/4).
        length_scale_range (tuple): Range for optimizing Gaussian kernel length scale.
        test_statistic_type (str): Which likelihood ratio to use ('pseudo', 'H1', 'H1_prime').
                                    Currently only 'pseudo' is fully supported by EM.
        verbose (bool): Whether to print progress information.

    Returns:
        dict: Dictionary containing p-value, observed statistic, bn, kernel parameters, etc.
    """
    
    # Fix random seed
    cp.random.seed(42)
    
    # --- 1. Data Preparation ---
    if verbose: print("1. Preparing Data...")
    y_node_idx = env1_data.columns.get_loc(y_node_col) if isinstance(y_node_col, str) else y_node_col

    # Find parent nodes (columns) for Y
    parent_indices = np.where(dag_adjacency_matrix[:, y_node_idx] == 1)[0]

    if len(parent_indices) == 0:
        print(f"Warning: Y node {y_node_col} has no parents in the DAG. Using intercept only.")
        # Create dummy X for intercept modeling if needed, or handle appropriately
        # For now, we'll proceed assuming at least one parent or modify kernel/model.
        # Let's create a constant feature for X if no parents.
        X1 = np.ones((len(env1_data), 1))
        X2 = np.ones((len(env2_data), 1))
    else:
        X1 = env1_data.iloc[:, parent_indices].values
        X2 = env2_data.iloc[:, parent_indices].values
        
    Y1 = env1_data.iloc[:, y_node_idx].values
    Y2 = env2_data.iloc[:, y_node_idx].values

    # Combine data
    X = np.vstack((X1, X2))
    Y = np.concatenate((Y1, Y2)).reshape(-1, 1) # Ensure Y is column vector
    Z = np.concatenate((np.ones(len(Y1)), 2 * np.ones(len(Y2)))).astype(int) # Group indicators
    n = len(Y)
    n1 = len(Y1)
    n2 = len(Y2)
    d = X.shape[1]

    # Standardize X and Y (important for kernel methods and stability)
    x_scaler = StandardScaler().fit(X)
    X_scaled = x_scaler.transform(X)
    y_scaler = StandardScaler().fit(Y)
    Y_scaled = y_scaler.transform(Y)
    
    if verbose: print(f"   n={n} (n1={n1}, n2={n2}), d={d}")

    # --- 2. Choose Kernel Matrix (Gaussian Kernel Parameter Selection) ---
    if verbose: print("\n2. Selecting Kernel Parameter (Gaussian)...")
    # Find best length scale using MLE under H0 (pooled data)
    # NOTE: This implements the basic MLE approach, not the refined one comparing
    #       pooled vs group estimates mentioned in Sec 5.1 for simplicity.
    best_ls, h0_max_log_likelihood = find_best_kernel_param(X_scaled, Y_scaled, length_scale_range, verbose=verbose)
    
    # Ensure scalar values for reporting
    best_ls_scalar = float(best_ls) if isinstance(best_ls, np.ndarray) else best_ls
    h0_max_log_likelihood_scalar = float(h0_max_log_likelihood) if isinstance(h0_max_log_likelihood, np.ndarray) else h0_max_log_likelihood
    
    if verbose: print(f"   Chosen length_scale = {best_ls_scalar:.4f}")
    
    # Calculate final Kernel matrix with chosen length scale
    K_n = gaussian_kernel(X_scaled, length_scale=best_ls)
    
    # Add jitter for numerical stability in eigen-decomposition
    jitter_k = 1e-7 * np.mean(np.diag(K_n)) 
    K_n_stable = K_n + np.identity(n) * jitter_k


    # --- 3. Eigen-decomposition ---
    if verbose: print("\n3. Performing Eigen-decomposition...")
    try:
        # Move data to GPU
        K_n_stable_gpu = cp.asarray(K_n_stable)

        # Perform eigen-decomposition on GPU using CuPy
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(K_n_stable_gpu)

        # Sort eigenvalues and eigenvectors in descending order
        idx = cp.argsort(eigenvalues_gpu)[::-1]
        eigenvalues_gpu = eigenvalues_gpu[idx]
        eigenvectors_gpu = eigenvectors_gpu[:, idx]

        # Copy eigenvalues to CPU
        eigenvalues = cp.asnumpy(eigenvalues_gpu)
        eigenvectors = cp.asnumpy(eigenvectors_gpu)

        # Ensure eigenvalues are non-negative by setting a small floor value
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        Gamma = eigenvectors
        C = eigenvalues
    except cp.linalg.LinAlgError:
        print("ERROR: Eigen-decomposition failed even with jitter.")
        return {"error": "Eigen-decomposition failed"}

    if verbose: print(f"   Eigenvalues range: {C.min():.4e} to {C.max():.4e}")

    # --- 4. Estimate Parameters under H0 and Calculate xi_n_hat ---
    if verbose: print("\n4. Estimating Parameters under H0 for bn calculation...")
    # We already have the MLE log-likelihood for H0 from kernel optimization
    # Now run EM one more time with the final K_n to get the variance estimates
    delta0_sq_scaled_hat, sigma0_sq_hat, h0_max_ll_check = run_em_vcm1(Y_scaled, K_n_stable, verbose=False)
    
    # Ensure scalar values
    delta0_sq_scaled_hat_scalar = float(delta0_sq_scaled_hat) if isinstance(delta0_sq_scaled_hat, np.ndarray) else delta0_sq_scaled_hat
    sigma0_sq_hat_scalar = float(sigma0_sq_hat) if isinstance(sigma0_sq_hat, np.ndarray) else sigma0_sq_hat
    h0_max_ll_check_scalar = float(h0_max_ll_check) if isinstance(h0_max_ll_check, np.ndarray) else h0_max_ll_check
    
    if h0_max_ll_check_scalar == -np.inf:
         print("ERROR: MLE estimation under H0 failed.")
         return {"error": "H0 MLE failed"}
         
    # Calculate xi_n_hat = (delta0_sq / n^(1-gamma)) / sigma0_sq
    # Note: delta0_sq_scaled_hat corresponds to delta0_sq / n^(1-gamma)
    if sigma0_sq_hat_scalar < 1e-8: # Avoid division by zero
        print("Warning: Estimated sigma0_sq is near zero. Setting xi_n_hat to large value.")
        xi_n_hat = 1e10 
    else:
        xi_n_hat = delta0_sq_scaled_hat_scalar / sigma0_sq_hat_scalar
        
    if verbose:
        print(f"   Estimated delta0_sq_scaled = {delta0_sq_scaled_hat_scalar:.4f}")
        print(f"   Estimated sigma0_sq = {sigma0_sq_hat_scalar:.4f}")
        print(f"   Estimated xi_n_hat = {xi_n_hat:.4f}")
        print(f"   Max Log Likelihood (H0) = {h0_max_log_likelihood_scalar:.4f}")


    # --- 5. Calculate Permutation Size bn ---
    if verbose: print("\n5. Calculating Permutation Size bn...")
    bn = calculate_bn(n, C, xi_n_hat, alpha=alpha)
    if verbose: print(f"   Chosen bn = {bn}")
    if bn == 0:
        print("Warning: Could not find suitable bn > 0. Permutation test cannot proceed.")
        # Return results indicating failure or choose a minimal bn=1?
        # For now, return error/indication.
        return {"p_value": 1.0, "observed_statistic": np.nan, "bn": 0, "message": "bn=0"}


    # --- 6. Calculate Observed Test Statistic ---
    if verbose: print("\n6. Calculating Observed Test Statistic...")
    # Using Pseudo Likelihood Ratio: T = L(H_pseudo) / L(H0) or logT = logL(H_pseudo) - logL(H0)
    
    # Log Likelihood under H0 is already calculated: h0_max_log_likelihood
    
    # Calculate Log Likelihood under H_pseudo
    # Requires running EM VCM1 for each group separately
    h_pseudo_log_likelihood = 0
    group_params = {}
    
    for h in np.unique(Z):
        if verbose: print(f"   Calculating MLE for Group {h} (H_pseudo)...")
        group_indices = np.where(Z == h)[0]
        Y_h = Y_scaled[group_indices]
        K_n_h = K_n[np.ix_(group_indices, group_indices)] # Group specific kernel matrix
        n_h = len(Y_h)
        if n_h <= 1: # Cannot run EM on single data point
            print(f"Warning: Group {h} has <= 1 sample. Skipping its contribution to H_pseudo likelihood.")
            ll_h = -np.inf # Or handle differently? Assign 0 log-likelihood?
            delta_h_sq = np.nan
            sigma_h_sq = np.nan
        else:
            jitter_h = 1e-7 * np.mean(np.diag(K_n_h)) # Add jitter
            K_n_h_stable = K_n_h + np.identity(n_h) * jitter_h
            delta_h_sq, sigma_h_sq, ll_h = run_em_vcm1(Y_h, K_n_h_stable, verbose=False)
            # Ensure scalar values
            ll_h = float(ll_h) if isinstance(ll_h, np.ndarray) else ll_h
            delta_h_sq = float(delta_h_sq) if isinstance(delta_h_sq, np.ndarray) else delta_h_sq
            sigma_h_sq = float(sigma_h_sq) if isinstance(sigma_h_sq, np.ndarray) else sigma_h_sq
            
            if ll_h == -np.inf:
                 print(f"Warning: MLE estimation failed for Group {h} under H_pseudo.")
                 # Decide how to handle failure: maybe exclude from test? For now, make overall stat invalid.
                 h_pseudo_log_likelihood = -np.inf 
                 break # Stop calculating pseudo likelihood if one group fails

        h_pseudo_log_likelihood += ll_h
        group_params[h] = {'delta_sq_scaled': delta_h_sq, 'sigma_sq': sigma_h_sq, 'log_likelihood': ll_h}

    if h_pseudo_log_likelihood == -np.inf:
        print("ERROR: Failed to calculate H_pseudo likelihood.")
        return {"error": "H_pseudo MLE failed"}
        
    # Observed statistic (using log likelihood difference for numerical stability)
    observed_log_statistic = h_pseudo_log_likelihood - h0_max_log_likelihood_scalar
    observed_log_statistic_scalar = float(observed_log_statistic) if isinstance(observed_log_statistic, np.ndarray) else observed_log_statistic
    
    if verbose: 
        print(f"   Max Log Likelihood (H_pseudo) = {h_pseudo_log_likelihood:.4f}")
        print(f"   Observed Log Test Statistic (logL(H_pseudo) - logL(H0)) = {observed_log_statistic_scalar:.4f}")

    # --- 7. Permutation Loop ---
    if verbose: print(f"\n7. Running {num_permutations} Permutations (bn={bn})...")
    # Move data to GPU
    W_gpu = cp.asarray(Gamma.T @ Y_scaled)  # Project original Y_scaled
    perm_indices_gpu = cp.arange(n - bn, n)  # Indices to permute

    # Variable to store initial values
    prev_tau1_sq = None
    prev_tau2_sq = None

    # Define single permutation test function on GPU
    def single_permutation_test_gpu(W_gpu, perm_indices_gpu, Gamma_gpu, K_n_stable_gpu, Z_gpu, prev_tau1_sq, prev_tau2_sq):
        """Performs a single permutation test on GPU."""
        # Create permuted W (Wp) using discrete permutation
        permuted_subset_gpu = cp.random.permutation(W_gpu[perm_indices_gpu])
        W_gpu[perm_indices_gpu] = permuted_subset_gpu  # Modify directly

        # Transform back to Yp
        Yp_gpu = Gamma_gpu @ W_gpu

        # Calculate test statistic for Yp
        # H0 likelihood for Yp
        tau1_sq, tau2_sq, h0_ll_p = run_em_vcm1(
            cp.asnumpy(Yp_gpu),
            cp.asnumpy(K_n_stable_gpu),
            verbose=False
        )
        h0_ll_p = float(h0_ll_p) if isinstance(h0_ll_p, np.ndarray) else h0_ll_p

        if h0_ll_p == -np.inf:
            W_gpu[perm_indices_gpu] = cp.sort(W_gpu[perm_indices_gpu])  # Restore original state
            return -np.inf, prev_tau1_sq, prev_tau2_sq  # Mark as invalid and return previous values

        # Update initial values
        prev_tau1_sq = tau1_sq
        prev_tau2_sq = tau2_sq

        # H_pseudo likelihood for Yp
        h_pseudo_ll_p = 0
        for h in cp.unique(Z_gpu):
            group_indices_gpu = cp.where(Z_gpu == h)[0]
            Yp_h_gpu = Yp_gpu[group_indices_gpu]
            K_n_h_gpu = K_n_stable_gpu[cp.ix_(group_indices_gpu, group_indices_gpu)]
            n_h = len(Yp_h_gpu)
            if n_h <= 1:
                W_gpu[perm_indices_gpu] = cp.sort(W_gpu[perm_indices_gpu])  # Restore original state
                return -np.inf, prev_tau1_sq, prev_tau2_sq  # Invalid group
            else:
                jitter_h = 1e-7 * cp.mean(cp.diag(K_n_h_gpu))
                K_n_h_stable_gpu = K_n_h_gpu + cp.identity(n_h) * jitter_h
                tau1_sq, tau2_sq, ll_hp = run_em_vcm1(
                    cp.asnumpy(Yp_h_gpu),
                    cp.asnumpy(K_n_h_stable_gpu),
                    verbose=False
                )
                ll_hp = float(ll_hp) if isinstance(ll_hp, np.ndarray) else ll_hp

                if ll_hp == -np.inf:
                    W_gpu[perm_indices_gpu] = cp.sort(W_gpu[perm_indices_gpu])  # Restore original state
                    return -np.inf, prev_tau1_sq, prev_tau2_sq  # Mark as invalid

                h_pseudo_ll_p += ll_hp

                # Update initial values
                prev_tau1_sq = tau1_sq
                prev_tau2_sq = tau2_sq

        if h_pseudo_ll_p == -np.inf:
            W_gpu[perm_indices_gpu] = cp.sort(W_gpu[perm_indices_gpu])  # Restore original state
            return -np.inf, prev_tau1_sq, prev_tau2_sq  # Mark as invalid

        # Restore original state
        W_gpu[perm_indices_gpu] = cp.sort(W_gpu[perm_indices_gpu])

        return h_pseudo_ll_p - h0_ll_p, prev_tau1_sq, prev_tau2_sq

    # Execute in parallel on GPU
    Gamma_gpu = cp.asarray(Gamma)
    K_n_stable_gpu = cp.asarray(K_n_stable)
    Z_gpu = cp.asarray(Z)

    # Process all permutations on GPU
    permuted_log_statistics_gpu = cp.zeros(num_permutations, dtype=cp.float32)
    for i in range(num_permutations):
        if verbose and (i % 10 == 0):
            print(f"   Permutation {i+1}/{num_permutations}...")
        permuted_log_statistics_gpu[i], prev_tau1_sq, prev_tau2_sq = single_permutation_test_gpu(
            W_gpu, perm_indices_gpu, Gamma_gpu, K_n_stable_gpu, Z_gpu, prev_tau1_sq, prev_tau2_sq
        )

    # Copy results to CPU
    permuted_log_statistics = cp.asnumpy(permuted_log_statistics_gpu)

    # --- 8. Calculate p-value ---
    if verbose: print("\n8. Calculating p-value...")
    
    # Count valid statistics (where calculation didn't fail)
    permuted_log_statistics = np.array(permuted_log_statistics)
    valid_perms = np.isfinite(permuted_log_statistics)
    num_valid_perms = np.sum(valid_perms)
    
    if num_valid_perms == 0:
        print("Warning: All permutations resulted in invalid statistics.")
        p_value = 1.0 # Or np.nan?
    else:
        # Compare observed statistic only to valid permuted statistics
        sorted_stats = np.sort(permuted_log_statistics[valid_perms])
        count_ge = len(sorted_stats) - np.searchsorted(sorted_stats, observed_log_statistic_scalar, side='left')
        p_value = (count_ge + 1) / (num_valid_perms + 1)

    if verbose: print(f"   p-value = {p_value:.4f}")

    results = {
        "p_value": p_value,
        "observed_log_statistic": observed_log_statistic_scalar,
        "bn": bn,
        "kernel_length_scale": best_ls_scalar,
        "h0_mle_delta0_sq_scaled": delta0_sq_scaled_hat_scalar,
        "h0_mle_sigma0_sq": sigma0_sq_hat_scalar,
        "h0_max_log_likelihood": h0_max_log_likelihood_scalar,
        "hpseudo_max_log_likelihood": h_pseudo_log_likelihood,
        "group_mle_params": group_params,
        "num_permutations_run": num_permutations,
        "num_valid_permutations": num_valid_perms
    }

    return results


# --- Example Usage (Simulation Study in Original Paper of GPR(Li et al., 2021))---
if __name__ == '__main__':

    # Import necessary libraries only if running as main script
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook

    np.random.seed(42)

    import matplotlib.pyplot as plt
    from itertools import product
    import matplotlib
    from matplotlib.gridspec import GridSpec
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    # Define all f0 functions

    def f0_case4(x1, x2):
        return 3 / (1 + np.power(x1, 2) + np.power(x2, 2)) - 2

    f0_functions = {
        "f4: 3 / (1 + x1^2 + x2^2) - 2": f0_case4
    }

    # Case configurations as per Table 1 (a)-(d)
    case_configs = {
        "(a)": {"p1": 0.5, "a1": 0.5, "a2": 0.5},
        "(b)": {"p1": 0.2, "a1": 0.5, "a2": 0.5},
        "(c)": {"p1": 0.5, "a1": 0.8, "a2": 0.2},
        "(d)": {"p1": 0.2, "a1": 0.8, "a2": 0.2},
        "(e)": {"p1": 0.5, "a1": 1.0, "a2": 0.0}
    }

    # Simulation setup
    def generate_scenario2_data(n, p1, a1, a2, f0, sigma0=0.1):
        Z = np.random.choice([1, 2], size=n, p=[p1, 1 - p1])
        X = np.zeros((n, 2))
        for i in range(n):
            a_z = a1 if Z[i] == 1 else a2
            for j in range(2):
                if np.random.rand() < a_z:
                    X[i, j] = np.random.uniform(-1, 0)
                else:
                    X[i, j] = np.random.uniform(0, 1)
        Y = np.array([f0(x[0], x[1]) for x in X]) + np.random.normal(0, 0.1, size=n)
        return X, Y, Z

    # Simulate and collect data
    n = 200  # Sample size n = 200 as per Figure 2
    num_repetitions = 30   # Number of repetitions per combination (Figure 2 requires 1000+ for p-value)
                           # Start with a smaller value (e.g., 30) for testing

    # Dictionary to store results: keys are (case_name, f0_name), values are lists of p-values
    simulation_results = {}

    # Use tqdm to display overall progress
    total_iterations = len(f0_functions) * len(case_configs) * num_repetitions
    pbar_total = tqdm(total=total_iterations, desc="Overall Simulation Progress")

    for rep_idx in range(num_repetitions):
        # Use a different seed for each repetition (optional, can start with a fixed seed for reproducibility)
        # np.random.seed(42 + rep_idx)  # Uncomment if needed

        for (f0_name, f0_func) in f0_functions.items():
            for (case_name, case_params) in case_configs.items():
                # Initialize result storage (on the first repetition)
                if (case_name, f0_name) not in simulation_results:
                    simulation_results[(case_name, f0_name)] = []

                # Generate data
                X, Y, Z = generate_scenario2_data(n=n, f0=f0_func, **case_params)
                
                env1_idx = (Z == 1)
                env2_idx = (Z == 2)

                # Check if there is sufficient data
                if np.sum(env1_idx) < 2 or np.sum(env2_idx) < 2:
                    # print(f"Warning: Insufficient data for rep {rep_idx+1}, case {case_name}, f0 {f0_name}. Skipping.")
                    pbar_total.update(1)
                    continue

                env1_data = pd.DataFrame(X[env1_idx], columns=["X1", "X2"])
                env1_data["Y"] = Y[env1_idx]

                env2_data = pd.DataFrame(X[env2_idx], columns=["X1", "X2"])
                env2_data["Y"] = Y[env2_idx]

                # Dummy DAG (Y depends on X1 and X2)
                # The partial_permutation_test function assumes X1, X2 are parents of Y
                # dag_adjacency_matrix: M[j,i]=1 -> j is parent of i
                # Assume Y is the 3rd column (index 2), X1 (index 0), X2 (index 1)
                dag_sim = np.zeros((3, 3))  # X1, X2, Y
                dag_sim[0, 2] = 1  # X1 → Y
                dag_sim[1, 2] = 1  # X2 → Y

                # Run the test (set verbose=False to minimize output)
                gpr_res = partial_permutation_test(env1_data, env2_data, "Y", dag_sim,
                                                   num_permutations=500, verbose=True)  # Paper uses B=500
                
                simulation_results[(case_name, f0_name)].append(gpr_res['p_value'])
                pbar_total.update(1)
        
        print(f"Completed repetition {rep_idx + 1}/{num_repetitions}")
    
    pbar_total.close()
    print("Simulations complete. Plotting results...")

    # --- Plot Results ---
    fig = plt.figure(figsize=(12, 8))  # Adjust figure size
    gs = GridSpec(2, 3, figure=fig)

    # Function expressions (mathematical representation)
    function_expressions = {
        "f4: 3 / (1 + x1^2 + x2^2) - 2": r'$(1 + x_1^2 + x_2^2)^{-1}$'
    }

    # Colors and markers for each case (refer to Figure 2 in the paper)
    case_styles = {
        "(a)": {"color": "black", "linestyle": "-", "marker": "o", "markersize": 3},
        "(b)": {"color": "red", "linestyle": "--", "marker": "s", "markersize": 3},
        "(c)": {"color": "green", "linestyle": "-.", "marker": "^", "markersize": 3},
        "(d)": {"color": "blue", "linestyle": ":", "marker": "D", "markersize": 3},
        "(e)": {"color": "cyan", "linestyle": "-", "marker": "p", "markersize": 3}
    }

    for idx, (f0_key, f0_func) in enumerate(f0_functions.items()):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        for case_name, case_params in case_configs.items():
            p_values = np.array(simulation_results.get((case_name, f0_key), [np.nan]))  # Handle missing results with NaN
            
            if np.all(np.isnan(p_values)) or len(p_values) == 0:  # Skip if no valid p-values
                print(f"No valid p-values for {f0_key}, {case_name}")
                continue

            sorted_p = np.sort(p_values)
            empirical_cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
            
            style = case_styles[case_name]
            ax.step(sorted_p, empirical_cdf, 
                    label=f"Case {case_name}", 
                    color=style["color"], 
                    linestyle=style["linestyle"],
                    linewidth=1.2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Uniform (Ref.)')  # Diagonal line
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel('p-value')
        ax.set_ylabel('empirical CDF')
        ax.set_title(function_expressions[f0_key], fontsize=10, y=-0.3)  # Move title down

    # Common legend (Figure 2 style)
    handles, labels = [], []
    for case_name in case_configs.keys():
        style = case_styles[case_name]
        # Create dummy lines for legend
        handles.append(plt.Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], lw=1.5))
        labels.append(f"Case {case_name}")
    
    # Add reference line to legend
    handles.append(plt.Line2D([0], [0], color='k', linestyle='--', lw=0.8))
    labels.append('Uniform (Ref.)')

    fig.legend(handles, labels, loc='upper center', ncol=len(case_configs)+1, bbox_to_anchor=(0.5, 0.98), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust space for legend
    plt.savefig('fig2_simulation_gpr_py_style_simple.png', dpi=300)
    print("Figure saved as fig2_simulation_gpr_py_style.png")