import random, os
import numpy as np
import igraph as ig
from typing import Union, List, Tuple, Dict, Optional
from scipy import stats

# The code below is from Chen, Tianyu, et al. "iSCAN: identifying causal mechanism shifts among nonlinear additive noise models." Advances in Neural Information Processing Systems 36 (2023): 44671-44706.

def set_seed(seed: int = 42) -> None:
    """
    Sets random seed of ``random`` and ``numpy`` to specified value
    for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Value for RNG, by default 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class DataGenerator:
    """
    Generate synthetic data to evaluate shifted nodes and shifted edges
    """
    SHIFT_CASE1 = "function_noise_std"
    SHIFT_CASE2 = "function_noise_skew"
    SHIFT_CASE3 = "function_function"
    SHIFT_CASE4 = "noise_std_noise_skew"
    SHIFT_CASE5 = "noise_std_noise_std"
    SHIFT_CASE6 = "noise_skew_noise_skew"
    
    def __init__(self, d: int, s0: int, graph_type: str, 
                 noise_std: Union[float, np.ndarray, List[float]] = .5,
                 shift_case: str = SHIFT_CASE1
                ):
        """
        Defines the class of graphs and data to be generated.

        Parameters
        ----------
        d : int
            Number of variables
        s0 : int
            Expected number of edges in the random graphs
        graph_type : str
            One of ``["ER", "SF"]``. ``ER`` and ``SF`` refer to Erdos-Renyi and Scale-free graphs, respectively.
        noise_std : Union[float, np.ndarray, List[float]], optional
            Sets the scale (variance) of the noise distributions. Should be either a scalar, or a list (array) of dim d. By default 0.5.
        shift_case : str, optional
            Type of shift to apply to nodes. One of the six predefined cases.
        """
        self.d, self.s0, self.graph_type = d, s0, graph_type
        self.noise_std = noise_std
        self.shift_case = shift_case
        
        if np.isscalar(noise_std):
            self.noise_std = np.ones(d) * noise_std
        else:
            if len(noise_std) != d:
                raise ValueError('noise_std must be a scalar or have length d')
        
        # Initialize node shift lists
        self.func_shift_nodes = []
        self.skew_shift_nodes = []
        self.variance_shift_nodes = []
    
    def _normal_noise(self, scale: float, size: int) -> np.ndarray:
        """
        Generate samples from a normal distribution with mean 0.
        
        Args:
            scale: Scale parameter (standard deviation)
            size: Number of samples to generate
            
        Returns:
            np.ndarray: Samples from normal distribution
        """
        return np.random.normal(loc=0, scale=scale, size=size)
    
    def _skewed_normal_noise(self, scale: float, size: int, alpha: float = 3.0) -> np.ndarray:
        """
        Generate samples from a skewed normal distribution with mean 0.
        
        Args:
            scale: Scale parameter (standard deviation)
            size: Number of samples to generate
            alpha: Skewness parameter
            
        Returns:
            np.ndarray: Samples from skewed normal distribution
        """
        return stats.skewnorm.rvs(a=alpha, loc=0, scale=scale, size=size)
    
    def _simulate_dag(self, d: int, s0: int, graph_type: str) -> np.ndarray:
        """
        Simulate random DAG with some expected number of edges.

        Parameters
        ----------
        d : int
            num of nodes
        s0 : int
            expected num of edges
        graph_type : str
            ER, SF

        Returns
        -------
        np.ndarray
            :math:`(d, d)` binary adj matrix of a sampled DAG
        """
        def _random_acyclic_orientation(B_und):
            return np.triu(B_und, k=1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER': # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF': # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
            B_und = _graph_to_adjmat(G)
            B = _random_acyclic_orientation(B_und)
        else:
            raise ValueError('unknown graph type')
        
        assert ig.Graph.Adjacency(B.tolist()).is_dag()
        
        return B

    def _choose_shifted_nodes(self, adj: np.ndarray, num_shifted_nodes: int) -> np.ndarray:
        """
        Randomly choose shifted nodes from non-root nodes

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        num_shifted_nodes : int
            Number of desired shifted nodes

        Returns
        -------
        np.ndarray
            Shifted nodes
        """
        roots = np.where(adj.sum(axis=0) == 0)[0]
        non_roots = np.setdiff1d(list(range(self.d)), roots)
        
        # uniformly choose shifted nodes from non-root nodes
        if len(non_roots) > 0:
            shifted_nodes = np.random.choice(non_roots, min(num_shifted_nodes, len(non_roots)), replace=False)
        else:
            shifted_nodes = np.array([], dtype=int)  # 비어있는 경우 처리
        
        return shifted_nodes
    
    def _assign_shift_types(self, shifted_nodes: np.ndarray) -> None:
        """
        Assign shift types to nodes based on the specified shift case.
        
        Parameters
        ----------
        shifted_nodes : np.ndarray
            Array of shifted node indices
        """
        # Reset shift type lists
        self.func_shift_nodes = []
        self.skew_shift_nodes = []
        self.variance_shift_nodes = []
        
        if len(shifted_nodes) == 0:
            return
        
        # Split shifted nodes into two groups
        n_half = len(shifted_nodes) // 2
        first_half = shifted_nodes[:n_half]
        second_half = shifted_nodes[n_half:]
        
        # Assign shift types based on case
        if self.shift_case == self.SHIFT_CASE1:  # function, noise_std
            self.func_shift_nodes = first_half
            self.variance_shift_nodes = second_half
        elif self.shift_case == self.SHIFT_CASE2:  # function, noise_skew
            self.func_shift_nodes = first_half
            self.skew_shift_nodes = second_half
        elif self.shift_case == self.SHIFT_CASE3:  # function, function
            self.func_shift_nodes = shifted_nodes  # All get function shift
        elif self.shift_case == self.SHIFT_CASE4:  # noise_std, noise_skew
            self.variance_shift_nodes = first_half
            self.skew_shift_nodes = second_half
        elif self.shift_case == self.SHIFT_CASE5:  # noise_std, noise_std
            self.variance_shift_nodes = shifted_nodes  # All get variance shift
        elif self.shift_case == self.SHIFT_CASE6:  # noise_skew, noise_skew
            self.skew_shift_nodes = shifted_nodes  # All get skew shift
        else:
            raise ValueError(f"Unknown shift case: {self.shift_case}")
    
    def _sample_from_same_structs(self, adj: np.ndarray, 
                                  shifted_nodes: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with the same DAG structure for both environments

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        shifted_nodes : np.ndarray
            Set of shifted nodes

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Datasets X and Y
        """
        data_env1, data_env2 = np.zeros((self.n, self.d)), np.zeros((self.n, self.d))
        
        # Generate noise for X (normal distribution)
        for i in range(self.d):
            data_env1[:, i] = self._normal_noise(scale=self.noise_std[i], size=self.n)
        
        # Assign shift types based on the specified case
        self._assign_shift_types(shifted_nodes)
        
        # Generate noise for dataset Y
        for i in range(self.d):
            if i in self.skew_shift_nodes:
                # Skewed normal distribution with alpha=3 for skew-shifted nodes
                data_env2[:, i] = self._skewed_normal_noise(scale=self.noise_std[i], size=self.n, alpha=3.0)
            elif i in self.variance_shift_nodes:
                # Normal distribution with doubled variance (std * sqrt(2))
                data_env2[:, i] = self._normal_noise(scale=self.noise_std[i]*np.sqrt(2.0), size=self.n)
            else:
                # Normal distribution with same variance for non-shifted and functionally shifted nodes
                data_env2[:, i] = self._normal_noise(scale=self.noise_std[i], size=self.n)
        
        # Apply structural equations
        for i in range(self.d):
            parents = np.nonzero(adj[:, i])[0]
            
            # Add individual parent effects for dataset X
            for j in parents:
                data_env1[:, i] += np.sin(data_env1[:, j] ** 2)
            
            # Add pairwise interactions between parents for dataset X
            for j_idx in range(len(parents)):
                for k_idx in range(j_idx + 1, len(parents)):
                    j, k = parents[j_idx], parents[k_idx]
                    data_env1[:, i] += 0.5 * np.sin(data_env1[:, j] * data_env1[:, k])
            
            # Add individual parent effects for dataset Y
            for j in parents:
                if i in self.func_shift_nodes:
                    data_env2[:, i] += 4 * np.cos(2 * data_env2[:, j]**2 - 3 * data_env2[:, j])
                else:
                    data_env2[:, i] += np.sin(data_env2[:, j] ** 2)
            
            # Add pairwise interactions between parents for dataset Y
            for j_idx in range(len(parents)):
                for k_idx in range(j_idx + 1, len(parents)):
                    j, k = parents[j_idx], parents[k_idx]
                    if i in self.func_shift_nodes:
                        data_env2[:, i] += 0.5 * np.cos(data_env2[:, j] * data_env2[:, k])
                    else:
                        data_env2[:, i] += 0.5 * np.sin(data_env2[:, j] * data_env2[:, k])
        return data_env1, data_env2 

    def sample(self, n: int, 
               num_shifted_nodes: int, 
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples two datasets from randomly generated DAGs

        Parameters
        ----------
        n : int
            Number of samples
        num_shifted_nodes : int
            Desired number of shifted nodes. Actual number can be lower.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two datasets with shape [n, d]
            Dataset X has normal noise, while dataset Y has shifts based on the specified case.

        """
        self.n = n
        self.adj_env1 = self._simulate_dag(self.d, self.s0, self.graph_type)
        self.shifted_nodes = self._choose_shifted_nodes(self.adj_env1, num_shifted_nodes)
        self.adj_env2 = self.adj_env1.copy()
        data_env1, data_env2 = self._sample_from_same_structs(self.adj_env1, self.shifted_nodes)
        
        return data_env1, data_env2

    def plot_dag_with_shifts(self, filename: str = "dag_plot.png") -> None:
        """
        Save DAG figure to file, highlighting shifted nodes.
        
        Args:
            filename: Output filename for the plot
        """
        if self.adj is None:
            raise RuntimeError("Call sample() first.")
            
        g = ig.Graph.Adjacency(self.adj.tolist(), mode="directed")
        layout = g.layout("kk")

        colors = []
        labels = []
        for i in range(self.d):
            if i not in self.shifted_nodes:
                c = self.COLOR_NO_SHIFT
            else:
                if self.shift_type.get(i, "") == self.SHIFT_TYPE_FUNCTION:
                    c = self.COLOR_FUNCTION_SHIFT
                else:
                    if self.noise_shift_type.get(i, "") == self.NOISE_SHIFT_STD:
                        c = self.COLOR_NOISE_SHIFT_STD
                    else:
                        c = self.COLOR_NOISE_SHIFT_SKEW
                    
            lbl = str(i)
            colors.append(c)
            labels.append(lbl)

        g.vs["color"] = colors
        g.vs["label"] = labels
        g.vs["size"] = 30

        ig.plot(g,
               target=filename,
               layout=layout,
               bbox=(600, 400),
               margin=50,
               vertex_label_color="black")
        print(f"Saved DAG figure to '{filename}'")
