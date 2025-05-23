from nn import MLP4
import torch
import torch.nn as nn
import numpy as np

class AffineCL(nn.Module):
    """
    Affine Causal Layer for normalizing flows that respects a DAG structure.
    
    This layer applies an affine transformation to each dimension according to
    the causal order defined by a given DAG. For each node, the transformation
    parameters depend only on its parent nodes.
    
    Attributes:
        dim: Number of dimensions/variables
        batch_norm: Whether to use batch normalization
        scale: Whether to use scaling parameters
        shift: Whether to use shift parameters
        dag: Directed acyclic graph represented as adjacency matrix
        parents: Dictionary mapping node indices to their parent nodes
        root_nodes: List of root nodes (nodes without parents)
        processing_order: Topological ordering of nodes
    """
    def __init__(self, dim, dag, net_class=MLP4, nh=24, scale=False, shift=True, batch_norm=True):
        """
        Initialize an affine causal layer.
        
        Args:
            dim: Number of dimensions/variables
            dag: Adjacency matrix representing the DAG
            net_class: Neural network class to use for transformation networks
            nh: Number of hidden units in the transformation networks
            scale: Whether to use scaling parameters
            shift: Whether to use shift parameters
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.dim = dim
        self.batch_norm = batch_norm
        self.scale = scale
        self.shift = shift

        self._parse_dag(dag)
        self.processing_order = self._topological_sort()

        # Parameter dictionaries for storing transformation parameters
        self.params = nn.ParameterDict()
        self.param_networks = nn.ModuleDict()
        
        for i in range(self.dim):
            if i in self.root_nodes:
                # Root nodes with learnable parameters
                if self.scale:
                    self.params[f'node_{i}_s'] = nn.Parameter(torch.randn(1) * 0.01)
                if self.shift:
                    self.params[f'node_{i}_t'] = nn.Parameter(torch.randn(1) * 0.01)
            else:
                # Non-root nodes with networks
                num_parents = len(self.parents[i])
                if num_parents > 0:
                    input_dim = num_parents
                    output_dim = 1  # Each network outputs a single scalar value
                    # Create scale network if scaling is enabled
                    if self.scale:
                        self.param_networks[f'node_{i}_s_net'] = net_class(input_dim, output_dim, nh, batch_norm)
                    # Create shift network if shifting is enabled
                    if self.shift:
                         self.param_networks[f'node_{i}_t_net'] = net_class(input_dim, output_dim, nh, batch_norm)

    def _parse_dag(self, dag):
        """
        Parse DAG adjacency matrix and set up parent-child relationships.
        
        Args:
            dag: Adjacency matrix representing the DAG
        """
        if isinstance(dag, np.ndarray):
            dag = torch.from_numpy(dag)
        if not isinstance(dag, torch.Tensor) or dag.shape != (self.dim, self.dim):
            raise ValueError(f"Invalid dag: expected shape ({self.dim}, {self.dim}), got {dag.shape}")

        self.nodes = list(range(self.dim))
        self.parents = {i: [] for i in self.nodes}
        self.child_nodes = {i: [] for i in self.nodes}
        for i in range(self.dim):
            for j in range(self.dim):
                if dag[i, j] == 1:  # edge i -> j
                    if i == j: 
                        raise ValueError("Self-loops are not allowed")
                    self.parents[j].append(i)
                    self.child_nodes[i].append(j)
        self.root_nodes = sorted([i for i in self.nodes if not self.parents[i]])
        
    def _topological_sort(self):
        """
        Perform topological sort of the DAG to determine processing order.
        
        Returns:
            List of node indices in topological order
        """
        in_degree = {i: len(self.parents[i]) for i in self.nodes}
        queue = list(self.root_nodes)
        sorted_nodes = []
        processed_in_queue = set(queue)

        while queue:
            u = min(queue)
            queue.remove(u)
            processed_in_queue.remove(u)
            sorted_nodes.append(u)

            for v in sorted(self.child_nodes[u]):
                in_degree[v] -= 1
                if in_degree[v] == 0 and v not in processed_in_queue:
                    queue.append(v)
                    processed_in_queue.add(v)
                elif in_degree[v] < 0:
                    raise RuntimeError("Error during topological sort")
                
        if len(sorted_nodes) != self.dim:
            remaining_nodes = set(self.nodes) - set(sorted_nodes)
            cycle_nodes_info = {n: {'parents': self.parents[n], 'in_degree': in_degree[n]} for n in remaining_nodes}
            raise ValueError(f"Graph contains a cycle, cannot perform topological sort. Remaining nodes: {cycle_nodes_info}")
        return sorted_nodes
        
    def forward(self, x):
        """
        Forward pass of the affine causal layer.
        
        Args:
            x: Input tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (transformed tensor, log determinant of Jacobian)
        """
        # try:
        # Handle 1D input tensors
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        base_param = next(iter(self.parameters()))  # Get any parameter for gradient connection
        z = torch.zeros(batch_size, self.dim, device=x.device)
        z = z + 0 * base_param.sum()  # Maintain gradient path
        
        s_total = torch.zeros(batch_size, self.dim, device=x.device)
        s_total = s_total + 0 * base_param.sum()  # Maintain gradient path
        
        for i in self.processing_order:
            node_idx = i
            x_i = x[:, node_idx].unsqueeze(-1)  # [batch_size, 1]

            # Create new tensors to avoid in-place operations
            s_i = torch.zeros(batch_size, 1, device=x.device)
            t_i = torch.zeros(batch_size, 1, device=x.device)

            if node_idx in self.root_nodes:
                # Root node: use stored parameters
                if self.scale and f'node_{node_idx}_s' in self.params:
                    s_param = self.params[f'node_{node_idx}_s']
                    s_i = s_param.expand(batch_size, 1)
                    
                if self.shift and f'node_{node_idx}_t' in self.params:
                    t_param = self.params[f'node_{node_idx}_t']
                    t_i = t_param.expand(batch_size, 1)
            else:
                # Non-root node: use parent z values as input
                parent_indices = self.parents[node_idx]
                if not parent_indices:
                    raise RuntimeError(f"Non-root node {node_idx} has no parents in forward pass.")

                # Get parent values through indexing
                parent_z = z[:, parent_indices]

                # Compute scale parameters if enabled
                if self.scale and f'node_{node_idx}_s_net' in self.param_networks:
                    s_net = self.param_networks[f'node_{node_idx}_s_net']
                    s_i = s_net(parent_z)  # [batch_size, 1]

                # Compute shift parameters if enabled
                if self.shift and f'node_{node_idx}_t_net' in self.param_networks:
                    t_net = self.param_networks[f'node_{node_idx}_t_net']
                    t_i = t_net(parent_z)  # [batch_size, 1]

            # Apply affine transformation without in-place operations
            z_i = torch.exp(s_i) * x_i + t_i
            
            # Update tensors with new values while maintaining gradient tracking
            z_new = z.clone()
            z_new[:, node_idx] = z_i.squeeze(-1)
            z = z_new
            
            s_total_new = s_total.clone()
            s_total_new[:, node_idx] = s_i.squeeze(-1)
            s_total = s_total_new

        # Calculate log determinant of Jacobian
        log_det = s_total.sum(dim=1)
        
        return z, log_det
    

    def backward(self, z):
        """
        Backward pass (inverse transformation) of the affine causal layer.
        
        Args:
            z: Latent tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (reconstructed tensor, log determinant of Jacobian)
        """
        batch_size = z.shape[0]
        x = torch.zeros_like(z)
        s_total = torch.zeros_like(z)  # For log_det calculation

        # Process nodes in the same order as forward
        for i in self.processing_order:
            node_idx = i
            z_i = z[:, node_idx].unsqueeze(-1)  # [batch_size, 1]

            # Initialize with zeros
            s_i = torch.zeros(batch_size, 1, device=z.device)
            t_i = torch.zeros(batch_size, 1, device=z.device)

            # Calculate s_i and t_i (same logic as forward)
            if node_idx in self.root_nodes:
                if self.scale and f'node_{node_idx}_s' in self.params:
                    s_i = self.params[f'node_{node_idx}_s'].expand(batch_size, 1)
                if self.shift and f'node_{node_idx}_t' in self.params:
                    t_i = self.params[f'node_{node_idx}_t'].expand(batch_size, 1)
            else:
                parent_indices = self.parents[node_idx]
                if not parent_indices:
                     raise RuntimeError(f"Non-root node {node_idx} has no parents during inverse.")

                parent_z = z[:, parent_indices]  # Use parent values from input z

                # Calculate scale
                if self.scale and f'node_{node_idx}_s_net' in self.param_networks:
                    s_net = self.param_networks[f'node_{node_idx}_s_net']
                    s_i = s_net(parent_z)

                # Calculate shift
                if self.shift and f'node_{node_idx}_t_net' in self.param_networks:
                    t_net = self.param_networks[f'node_{node_idx}_t_net']
                    t_i = t_net(parent_z)

            # Apply inverse affine transform
            # If scale=False, s_i=0, exp(-s_i)=1; if shift=False, t_i=0
            x_i = (z_i - t_i) * torch.exp(-s_i)
            x[:, node_idx] = x_i.squeeze(-1)  # Store result
            s_total[:, node_idx] = s_i.squeeze(-1)  # Store s values for log_det

        # Calculate log determinant of Jacobian (sum of s values)
        log_det = torch.sum(s_total, dim=1)
        return x, log_det

