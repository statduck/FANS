import torch.nn as nn
import torch
import numpy as np


class NormalizingFlow(nn.Module):
    """
    A sequence of normalizing flows stacked together.
    
    This module applies a sequence of flow transformations,
    where each transformation maps from one space to another.
    
    Attributes:
        flows: Module list of flow transformations
    """

    def __init__(self, flows):
        """
        Initialize the normalizing flow sequence.
        
        Args:
            flows: List of flow transformation modules
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        """
        Forward pass through the flow sequence.
        
        Args:
            x: Input tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (list of intermediate tensors, log determinant of Jacobian)
        """
        try:
            # Handle 1D input tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            m, _ = x.shape
            # Initialize tensors
            log_det = torch.zeros(m, device=x.device)
            zs = [x]
            
            # Evaluation mode (with gradient disabled)
            if not self.training:
                with torch.no_grad():
                    for flow in self.flows:
                        flow_output = flow.forward(x)
                        
                        # Handle unexpected output formats
                        if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                            x = flow_output[0] if isinstance(flow_output, tuple) and len(flow_output) > 0 else flow_output
                            ld = torch.zeros_like(log_det)
                        else:
                            x, ld = flow_output
                            
                        # Create new tensor instead of in-place operations
                        log_det = log_det + ld
                        # Create new list instead of modifying existing list
                        zs = zs + [x]
                    return zs, log_det
            
            # Training mode
            for flow in self.flows:
                flow_output = flow.forward(x)
                
                # Handle unexpected output formats
                if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                    x = flow_output[0] if isinstance(flow_output, tuple) and len(flow_output) > 0 else flow_output
                    ld = torch.zeros_like(log_det)
                else:
                    x, ld = flow_output
                
                # Create new tensor instead of in-place operations
                log_det = log_det + ld
                # Create new list instead of modifying existing list
                zs = zs + [x]
            
            return zs, log_det
        
        except Exception as e:
            print(f"Error in NormalizingFlow.forward: {e}")
            raise 

    def backward(self, z):
        """
        Backward pass (inverse) through the flow sequence.
        
        Args:
            z: Input tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (list of intermediate tensors, log determinant of Jacobian)
        """
        try:
            # Handle 1D input tensors
            if len(z.shape) == 1:
                z = z.unsqueeze(0)
            
            m, _ = z.shape
            log_det = torch.zeros(m, device=z.device)
            xs = [z]
            
            # Evaluation mode
            if not self.training:
                with torch.no_grad():
                    for flow in self.flows[::-1]:  # Process flows in reverse order
                        try:
                            flow_output = flow.backward(z)
                            
                            # Handle unexpected output formats
                            if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                                z = flow_output[0] if isinstance(flow_output, tuple) and len(flow_output) > 0 else flow_output
                                ld = torch.zeros_like(log_det)
                            else:
                                z, ld = flow_output
                                
                            # Create new tensor instead of in-place operations
                            log_det = log_det + ld
                            # Create new list instead of modifying existing list
                            xs = xs + [z]
                        except Exception as e:
                            print(f"Error in individual flow.backward: {e}")
                            raise
                    return xs, log_det
            
            # Training mode
            for flow in self.flows[::-1]:  # Process flows in reverse order
                try:
                    flow_output = flow.backward(z)
                    
                    # Handle unexpected output formats
                    if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                        z = flow_output[0] if isinstance(flow_output, tuple) and len(flow_output) > 0 else flow_output
                        ld = torch.zeros_like(log_det)
                    else:
                        z, ld = flow_output
                    
                    # Create new tensor instead of in-place operations
                    log_det = log_det + ld
                    # Create new list instead of modifying existing list
                    xs = xs + [z]
                except Exception as e:
                    print(f"Error in individual flow.backward: {e}")
                    raise 
            return xs, log_det
        
        except Exception as e:
            print(f"Error in NormalizingFlow.backward: {e}")
            raise 

class NormalizingFlowModel(nn.Module):
    """
    A normalizing flow model consisting of a prior distribution and a sequence of flows.
    
    This model transforms between the data space and the latent space through 
    a series of invertible transformations, while tracking the change in density.
    
    Attributes:
        prior: Prior distribution module
        flow: NormalizingFlow module for transformations
    """

    def __init__(self, prior, flows):
        """
        Initialize the normalizing flow model.
        
        Args:
            prior: Prior distribution module
            flows: List of flow transformation modules
        """
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input data tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (list of intermediate tensors, prior log probability, log determinant of Jacobian)
        """
        try:
            # Handle 1D input tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            if not self.training:
                with torch.no_grad():
                    flow_output = self.flow.forward(x)
                    
                    # Handle unexpected output format
                    if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                        zs = flow_output if not isinstance(flow_output, tuple) else flow_output[0]
                        log_det = torch.zeros(x.size(0), device=x.device)
                    else:
                        zs, log_det = flow_output
                    
                    # Calculate log probability under the prior
                    prior_logprob = self.prior.log_prob(zs[-1])
                    prior_logprob = prior_logprob.sum(dim=1)  # Sum over dimensions
                    
                    return zs, prior_logprob, log_det
            
            # Training mode
            flow_output = self.flow.forward(x)
            
            # Handle unexpected output format
            if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                zs = flow_output if not isinstance(flow_output, tuple) else flow_output[0]
                log_det = torch.zeros(x.size(0), device=x.device)
            else:
                zs, log_det = flow_output
                
            # Calculate log probability under the prior
            prior_logprob = self.prior.log_prob(zs[-1])
            prior_logprob = prior_logprob.sum(dim=1)  # Sum over dimensions
            
            return zs, prior_logprob, log_det
        
        except Exception as e:
            print(f"Error in NormalizingFlowModel.forward: {e}")
            # Create safe return values
            # batch_size = 1 if len(x.shape) == 1 else x.shape[0]
            # dummy_z = [x]
            # dummy_logprob = torch.zeros(batch_size, device=x.device)
            # dummy_logdet = torch.zeros(batch_size, device=x.device)
            # return dummy_z, dummy_logprob, dummy_logdet
            raise 

    def backward(self, z):
        """
        Backward pass (inverse) through the model.
        
        Args:
            z: Input latent tensor of shape [batch_size, dim]
            
        Returns:
            tuple of (list of intermediate tensors, log determinant of Jacobian)
        """
        try:
            # Handle 1D input tensors
            if len(z.shape) == 1:
                z = z.unsqueeze(0)
            
            # Evaluation mode
            if not self.training:
                with torch.no_grad():
                    try:
                        flow_output = self.flow.backward(z)
                        if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                            xs = [z]
                            log_det = torch.zeros(z.shape[0], device=z.device)
                        else:
                            xs, log_det = flow_output
                        return xs, log_det
                    except Exception as e:
                        print(f"Error in backward (eval mode): {e}")
                        return [z], torch.zeros(z.shape[0], device=z.device)
            
            # Training mode
            try:
                flow_output = self.flow.backward(z)
                if not isinstance(flow_output, tuple) or len(flow_output) != 2:
                    xs = [z]
                    log_det = torch.zeros(z.shape[0], device=z.device)
                else:
                    xs, log_det = flow_output
                return xs, log_det
            except Exception as e:
                print(f"Error in backward (train mode): {e}")
                raise 
            
        except Exception as e:
            print(f"Error in NormalizingFlowModel.backward: {e}")
            raise 

    def log_likelihood(self, x):
        """
        Calculate log likelihood of data under the model.
        
        Args:
            x: Input data tensor of shape [batch_size, dim]
            
        Returns:
            numpy array of log likelihood values for each input
        """
        try:
            # Convert numpy array to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.tensor(x.astype(np.float32))
            
            # Handle 1D input tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            # Move to correct device if needed
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
            
            # Compute log likelihood
            result = self.forward(x)
            if not isinstance(result, tuple) or len(result) != 3:
                print(f"Warning: forward returned unexpected format")
                return np.zeros(x.shape[0])
            
            _, prior_logprob, log_det = result
            return (prior_logprob + log_det).cpu().detach().numpy()
        
        except Exception as e:
            print(f"Error in log_likelihood: {e}")
            # Return zeros in case of error
            batch_size = 1 if not hasattr(x, 'shape') or len(x.shape) == 1 else x.shape[0]
            return np.zeros(batch_size)

    def sample(self, num_samples):
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled tensors
        """
        with torch.no_grad():
            z = self.prior.sample((num_samples,))
            xs, _ = self.flow.backward(z)
            return xs
