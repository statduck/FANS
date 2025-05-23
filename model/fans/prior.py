"""
Prior distributions for normalizing flow models.

This module contains learnable prior distributions that can be used in
normalizing flow models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnableGaussian(nn.Module):
    """
    Gaussian distribution with learnable scale (fixed mean at 0).
    
    This class implements a multivariate Gaussian distribution with fixed mean at 0
    and learnable scales for each dimension.
    
    Attributes:
        dim: Dimensionality of the distribution
        params: ParameterDict containing node-specific scale parameters
    """
    def __init__(self, dim, init_scale=1.0, device=None):
        """
        Initialize learnable Gaussian distribution.
        
        Args:
            dim: Distribution dimensionality
            init_scale: Initial scale value
            device: PyTorch device
        """
        super().__init__()
        self.dim = dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-6
        
        self.params = nn.ParameterDict()
        for i in range(dim):

            init_val = np.log(init_scale) + np.random.uniform(-0.1, 0.1)
            self.params[f'node_{i}_log_scale'] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        
    def log_prob(self, x):
        """
        Calculate log probability of samples.
        
        Args:
            x: Input tensor [batch_size, dim] or [dim]
            
        Returns:
            Log probability tensor
        """
        try:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
                
            batch_size = x.shape[0]
            log_probs = torch.zeros_like(x)
                    
            for i in range(self.dim):
                log_scale_i = self.params[f'node_{i}_log_scale']
                
                # reg_loss += 0.01 * log_scale_i**2
                
                scale_i = torch.exp(log_scale_i)
                
                norm_constant = -0.5 * torch.log(2 * torch.tensor(np.pi, device=x.device))
                                
                log_var = 2 * log_scale_i  # log(scale_i²) = 2*log(scale_i)
                precision = 1.0 / (scale_i**2)  # 1/var_i
                
                normalized_squared_error = x[:, i:i+1]**2 * precision
                log_prob_i = -0.5 * (torch.log(2 * torch.tensor(np.pi, device=x.device)) 
                                    + log_var 
                                    + normalized_squared_error)
                
                log_prob_i = torch.clamp(log_prob_i, min=-20.0)
                
                log_probs[:, i:i+1] = log_prob_i
                                        
            return log_probs
            
        except Exception as e:
            print(f"Error in LearnableGaussian.log_prob: {e}")
            return torch.ones_like(x) * -10.0
    
    def sample(self, size):
        """
        Sample from the distribution.
        
        Args:
            size: Batch size or shape containing batch size
            
        Returns:
            Samples from the distribution
        """
        try:
            # Create sample shape
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            
            # Generate standard normal samples
            eps = torch.randn(sample_shape, device=self.device)
            samples = torch.zeros_like(eps)
            
            for i in range(self.dim):
                log_scale_i = self.params[f'node_{i}_log_scale']
                scale_i = torch.exp(log_scale_i).clamp(min=self.eps)
                samples[..., i] = eps[..., i] * scale_i
            
            return samples
            
        except Exception as e:
            print(f"Error in LearnableGaussian.sample: {e}")
            # Return zeros as fallback
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            return torch.zeros(sample_shape, device=self.device)
    
    def get_params(self):
        """
        Get distribution parameters.
        
        Returns:
            Dictionary of distribution parameters
        """
        scales = torch.zeros(self.dim, device=self.device)
        log_scales = torch.zeros(self.dim, device=self.device)
        
        for i in range(self.dim):
            log_scale_i = self.params[f'node_{i}_log_scale']
            scales[i] = torch.exp(log_scale_i)
            log_scales[i] = log_scale_i
            
        return {
            'scales': scales.detach().cpu().numpy(),
            'log_scales': log_scales.detach().cpu().numpy()
        }

class LearnableLaplace(nn.Module):
    """
    Laplace distribution with learnable scale (fixed mean at 0).
    
    This class implements a multivariate Laplace distribution with fixed mean at 0
    and learnable scales for each dimension.
    
    Attributes:
        dim: Dimensionality of the distribution
        params: ParameterDict containing node-specific scale parameters
    """
    def __init__(self, dim, init_scale=1.0, device=None):
        """
        Initialize learnable Laplace distribution.
        
        Args:
            dim: Distribution dimensionality
            init_scale: Initial scale value
            device: PyTorch device
        """
        super().__init__()
        self.dim = dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-6  # Small constant for numerical stability
        
        self.params = nn.ParameterDict()
        for i in range(dim):
            self.params[f'node_{i}_log_scale'] = nn.Parameter(torch.ones(1) * np.log(init_scale))
        
    def log_prob(self, x):
        """
        Calculate log probability of samples.
        
        Args:
            x: Input tensor [batch_size, dim] or [dim]
            
        Returns:
            Log probability tensor
        """
        try:
            # Handle 1D input tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
            # Move to correct device if needed
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
                
            batch_size = x.shape[0]
            log_probs = torch.zeros_like(x)
            
            requires_grad = torch.is_grad_enabled() and self.training
            print(f"log_prob - requires_grad={requires_grad}, training={self.training}")
            
            for i in range(self.dim):
                log_scale_i = self.params[f'node_{i}_log_scale']
                print(f"node_{i}_log_scale.requires_grad={log_scale_i.requires_grad}")
                
                scale_i = torch.exp(log_scale_i).clamp(min=self.eps)
                
                norm_constant = -torch.log(2 * scale_i)
                log_prob_i = norm_constant - torch.abs(x[:, i:i+1]) / scale_i
                log_probs[:, i:i+1] = log_prob_i
            
            return log_probs
            
        except Exception as e:
            print(f"Error in LearnableLaplace.log_prob: {e}")
            return torch.ones_like(x) * -1e10  # Very low log probability in case of error
    
    def sample(self, size):
        """
        Sample from the distribution.
        
        Args:
            size: Batch size or shape containing batch size
            
        Returns:
            Samples from the distribution
        """
        try:
            # Create sample shape
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            
            # Generate uniform samples for Laplace via inverse CDF
            u = torch.rand(sample_shape, device=self.device) - 0.5
            samples = torch.zeros_like(u)
            
            for i in range(self.dim):
                log_scale_i = self.params[f'node_{i}_log_scale']
                scale_i = torch.exp(log_scale_i).clamp(min=self.eps)
                samples[..., i] = -scale_i * torch.sign(u[..., i]) * torch.log(1 - 2 * torch.abs(u[..., i]))
            
            return samples
            
        except Exception as e:
            print(f"Error in LearnableLaplace.sample: {e}")
            # Return zeros as fallback
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            return torch.zeros(sample_shape, device=self.device)
    
    def get_params(self):
        """
        Get distribution parameters.
        
        Returns:
            Dictionary of distribution parameters
        """
        scales = torch.zeros(self.dim, device=self.device)
        log_scales = torch.zeros(self.dim, device=self.device)
        
        for i in range(self.dim):
            log_scale_i = self.params[f'node_{i}_log_scale']
            scales[i] = torch.exp(log_scale_i)
            log_scales[i] = log_scale_i
            
        return {
            'scales': scales.detach().cpu().numpy(),
            'log_scales': log_scales.detach().cpu().numpy()
        }

class LearnableUniform(nn.Module):
    """
    Uniform distribution with learnable range.
    
    This class implements a multivariate uniform distribution with learnable
    range parameters for each dimension.
    
    Attributes:
        dim: Dimensionality of the distribution
        params: ParameterDict containing node-specific range parameters
    """
    def __init__(self, dim, init_range=2.0, device=None):
        """
        Initialize learnable uniform distribution.
        
        Args:
            dim: Distribution dimensionality
            init_range: Initial range value
            device: PyTorch device
        """
        super().__init__()
        self.dim = dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-6  # Small constant for numerical stability
        
        self.params = nn.ParameterDict()
        for i in range(dim):
            self.params[f'node_{i}_log_range'] = nn.Parameter(torch.ones(1) * np.log(init_range))
        
    def log_prob(self, x):
        """
        Calculate log probability of samples.
        
        Args:
            x: Input tensor [batch_size, dim] or [dim]
            
        Returns:
            Log probability tensor
        """
        try:
            # Handle 1D input tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
            # Move to correct device if needed
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
                
            batch_size = x.shape[0]
            log_probs = torch.zeros_like(x)
            
            requires_grad = torch.is_grad_enabled() and self.training
            # print(f"log_prob - requires_grad={requires_grad}, training={self.training}")
            
            for i in range(self.dim):
            
                log_range_i = self.params[f'node_{i}_log_range']
                # print(f"node_{i}_log_range.requires_grad={log_range_i.requires_grad}")
                range_i = torch.exp(log_range_i).clamp(min=self.eps)
            
                norm_constant = -torch.log(2 * range_i)
                inside_range = (torch.abs(x[:, i:i+1]) <= range_i).float()
                log_prob_i = inside_range * norm_constant + (1 - inside_range) * -1e10
                log_probs[:, i:i+1] = log_prob_i
            
            return log_probs
            
        except Exception as e:
            print(f"Error in LearnableUniform.log_prob: {e}")
            return torch.ones_like(x) * -1e10  # Very low log probability in case of error
    
    def sample(self, size):
        """
        Sample from the distribution.
        
        Args:
            size: Batch size or shape containing batch size
            
        Returns:
            Samples from the distribution
        """
        try:
            # Create sample shape
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            
            # Generate uniform samples
            u = torch.rand(sample_shape, device=self.device) * 2 - 1  # Uniform in [-1, 1]
            samples = torch.zeros_like(u)
            
            for i in range(self.dim):
                log_range_i = self.params[f'node_{i}_log_range']
                range_i = torch.exp(log_range_i).clamp(min=self.eps)
                samples[..., i] = u[..., i] * range_i
            
            return samples
            
        except Exception as e:
            print(f"Error in LearnableUniform.sample: {e}")
            # Return zeros as fallback
            if isinstance(size, tuple):
                sample_shape = size + (self.dim,)
            else:
                sample_shape = (size, self.dim)
            return torch.zeros(sample_shape, device=self.device)
    
    def get_params(self):
        """
        Get distribution parameters.
        
        Returns:
            Dictionary of distribution parameters
        """
        ranges = torch.zeros(self.dim, device=self.device)
        log_ranges = torch.zeros(self.dim, device=self.device)
        
        for i in range(self.dim):
            log_range_i = self.params[f'node_{i}_log_range']
            ranges[i] = torch.exp(log_range_i)
            log_ranges[i] = log_range_i
            
        return {
            'ranges': ranges.detach().cpu().numpy(),
            'log_ranges': log_ranges.detach().cpu().numpy()
        }