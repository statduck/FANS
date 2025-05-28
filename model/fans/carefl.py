import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import CustomSyntheticDatasetDensity
from model.fans.AffineCL import AffineCL
from model.fans.prior import LearnableGaussian
from model.fans.nn import MLP4
from model.fans.flow import NormalizingFlowModel

class CAREFL:
    """
    This class implements normalizing flows for causal representation learning,
    with specialized support for selective parameter training and optimization.
    
    Attributes:
        config: Configuration object with hyperparameters
        dim: Data dimensionality
        dag: Directed acyclic graph structure
        flow: Trained normalizing flow model
        parents: Dictionary mapping nodes to their parent nodes
        root_nodes: List of nodes with no parents
    """
    def __init__(self, config):
        """
        Initialize CAREFL model with configuration.
        
        Args:
            config: Configuration object containing model hyperparameters
        """
        self.config = config
        self.n_layers = config.flow.nl
        self.n_hidden = config.flow.nh
        self.epochs = config.training.epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initial state - will be updated after fitting
        self.direction = 'none'
        self.flow_xy = self.flow_yx = self.flow = None
        self._nhxy = self._nhyx = self._nlxy = self._nlyx = None
        self.dag = None 
        self.parents = None
        self.root_nodes = None
        self.dim = None

    def _train_epoch(self, flow, optimizer, train_loader):
        """
        Train model for a single epoch.
        
        Args:
            flow: Flow model to train
            optimizer: PyTorch optimizer
            train_loader: DataLoader for training data
            
        Returns:
            epoch_stats: Dictionary containing training statistics
        """
        flow.train()
        
        param_stats = {
            'prior': {'count': 0, 'grad_sum': 0.0, 'param_sum': 0.0},
            'flow_params': {'count': 0, 'grad_sum': 0.0, 'param_sum': 0.0},
            'flow_nets': {'count': 0, 'grad_sum': 0.0, 'param_sum': 0.0}
        }
        
        epoch_stats = {
            'loss': 0,
            'nll_loss': 0,
            'log_det': 0,
            'prior_logprob': 0,
            'processed_samples': 0,
            'num_batches': 0
        }
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(self.device)
            batch_size = x.shape[0]
            if batch_size <= 1:  # Skip small batches
                continue
            
            # Forward pass
            optimizer.zero_grad()
            zs, prior_logprob, log_det = flow(x)
            nll_loss = -torch.sum(prior_logprob + log_det)
            loss = (nll_loss / batch_size)
            
            if batch_idx == 0:                
                for name, param in flow.named_parameters():
                    if param.requires_grad:
                        if 'prior.params' in name:
                            group = 'prior'
                        elif 'flow' in name and 'param_networks' in name:
                            group = 'flow_nets'
                        else:
                            group = 'flow_params'
                        
                        param_stats[group]['count'] += 1
                        param_stats[group]['param_sum'] += param.data.abs().mean().item()
                        
            # Backward pass
            loss.backward()
            
            # Log post-backward info (첫 배치만)
            if batch_idx == 0:
                missing_grads = []
                
                for name, param in flow.named_parameters():
                    if param.requires_grad:
                        # 그래디언트 정보 수집
                        if param.grad is not None:
                            grad_norm = param.grad.data.abs().mean().item()
                            
                            if 'prior.params' in name:
                                param_stats['prior']['grad_sum'] += grad_norm
                            elif 'flow' in name and 'param_networks' in name:
                                param_stats['flow_nets']['grad_sum'] += grad_norm
                            else:
                                param_stats['flow_params']['grad_sum'] += grad_norm
                                
                        else:
                            missing_grads.append(name)
                
                if missing_grads:
                    print(f"[DEBUG] Missing gradients for {len(missing_grads)}/{len(list(p for p in flow.parameters() if p.requires_grad))} parameters")
            
            # Apply gradients
            optimizer.step()
            
            # Accumulate statistics
            epoch_stats['loss'] += loss.item() * batch_size
            epoch_stats['nll_loss'] += (nll_loss / batch_size).item() * batch_size
            epoch_stats['log_det'] += torch.sum(log_det).item()
            epoch_stats['prior_logprob'] += torch.sum(prior_logprob).item()
            epoch_stats['processed_samples'] += batch_size
            epoch_stats['num_batches'] += 1
        
        return epoch_stats

    def _log_epoch_stats(self, epoch_stats, e, total_epochs, flow_idx):
        """
        Log epoch training statistics.
        
        Args:
            epoch_stats: Dictionary with training statistics
            e: Current epoch number
            total_epochs: Total number of epochs
            flow_idx: Index of the current flow model
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        if epoch_stats['processed_samples'] <= 0:
            print(f"Warning: No batches processed in epoch {e} for flow {flow_idx}")
            return float('nan')
        
        # Calculate average metrics
        avg_loss = epoch_stats['loss'] / epoch_stats['processed_samples']
        avg_nll = epoch_stats['nll_loss'] / epoch_stats['processed_samples']
        avg_log_det = epoch_stats['log_det'] / epoch_stats['processed_samples']
        avg_prior_logprob = epoch_stats['prior_logprob'] / epoch_stats['processed_samples']
        
        if (e % 10 == 0 or e == total_epochs - 1):
            print(f'Flow {flow_idx}, Epoch {e}/{total_epochs} | Loss: {avg_loss:.4f} | NLL: {avg_nll:.4f} | LogDet: {avg_log_det:.4f} | PriorLogP: {avg_prior_logprob:.4f}')
        
        return avg_loss

    def _train(self, dset, trainable_params_selector=None, n_epochs=None):
        """
        Train flow models with early stopping and parameter selection.
        
        Args:
            dset: Dataset for training
            trainable_params_selector: Optional function to select trainable parameters
            n_epochs: Number of epochs to train (uses config value if None)
            
        Returns:
            tuple of (trained_flows, loss_values)
        """
        print(f"Starting training with dataset size: {len(dset)}")
        
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.config.training.batch_size)
        flows = self._get_flow_arch()
        all_loss_vals = []
        
        # Training parameters
        epochs = n_epochs if n_epochs is not None else self.epochs
        
        # Early stopping parameters
        patience = getattr(self.config.training, 'patience', 20)
        min_epochs = getattr(self.config.training, 'min_epochs', 50)
        improvement_threshold = getattr(self.config.training, 'early_stopping_threshold', 0.0005)
            
        # Set all flows to evaluation mode initially
        flows_copy = []
        for flow in flows:
            flow.eval()
            flows_copy.append(flow)
        
        try:    
            for flow_idx, flow in enumerate(flows_copy):
                # Reset all parameters to non-trainable
                for param in flow.parameters():
                    param.requires_grad = False
                
                # Select parameters to train
                if trainable_params_selector is not None:
                    trainable_params = trainable_params_selector(flow)
                else:
                    # Train all parameters
                    for param in flow.parameters():
                        param.requires_grad = True
                    trainable_params = list(flow.parameters())
                
                # Use parameter IDs for safe comparison
                trainable_param_ids = [id(p) for p in trainable_params]
                trainable_param_names = []
                
                for name, param in flow.named_parameters():
                    if id(param) in trainable_param_ids:
                        param.requires_grad = True
                        trainable_param_names.append(name)
                
                print(f"Selected parameters for training ({len(trainable_param_names)}/{len(list(flow.parameters()))}): {trainable_param_names}")
                
                # Setup optimizer and scheduler
                optimizer, scheduler = self._get_optimizer(trainable_params)
                loss_vals = []
                
                # Early stopping variables
                best_loss = float('inf')
                best_epoch = 0
                patience_counter = 0
                best_state = None
                
                for e in range(epochs):
                    # Train for one epoch
                    epoch_stats = self._train_epoch(
                        flow, optimizer, train_loader
                    )
                    
                    # Log statistics and record loss
                    avg_loss = self._log_epoch_stats(epoch_stats, e, epochs, flow_idx)
                    loss_vals.append(avg_loss)
                    
                    # Update learning rate scheduler
                    if self.config.optim.scheduler and scheduler is not None:
                        scheduler.step(avg_loss)
                    
                    # Skip the rest if no samples were processed
                    if epoch_stats['processed_samples'] <= 0:
                        break
                    
                    # Early stopping logic
                    if e >= min_epochs:
                        improvement = best_loss - avg_loss
                        
                        if improvement > improvement_threshold:
                            # Improved significantly - reset patience
                            best_loss = avg_loss
                            best_epoch = e
                            patience_counter = 0
                            best_state = {k: v.cpu().detach().clone() for k, v in flow.state_dict().items()}
                            print(f"  --> New best loss: {best_loss:.4f} (improvement: {improvement:.6f})")
                        else:
                            # No significant improvement - increase patience counter
                            patience_counter += 1
                            print(f"  --> No significant improvement. Current patience: {patience_counter}/{patience} (best: {best_loss:.4f} at epoch {best_epoch})")
                            
                        # Stop if patience exceeded
                        if patience_counter >= patience:
                            print(f'Flow {flow_idx}: Early stopping at epoch {e} (Patience: {patience}, Best epoch: {best_epoch})')
                            if best_state is not None:
                                flow.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                                print(f'  Restored best model state with loss: {best_loss:.4f} from epoch {best_epoch}')
                            else:
                                print(f'  No improvement recorded, stopping with current model.')
                            break
                    else:
                        # Still in minimum epochs phase - just record best
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_epoch = e
                            best_state = {k: v.cpu().detach().clone() for k, v in flow.state_dict().items()}
                            print(f"  --> New best loss: {best_loss:.4f} (min_epochs not reached yet: {e}/{min_epochs})")
                
                # Restore best model if early stopping didn't trigger
                if best_state is not None and patience_counter < patience:
                    flow.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                    print(f'Training completed. Restored best model state with loss: {best_loss:.4f} from epoch {best_epoch}')
                
                all_loss_vals.append(loss_vals)
            
            return flows_copy, all_loss_vals
        except Exception as e:
            print(f"Training failed: {e}")
            raise

    def fit_to_sem(self, data, dag=None, return_scores=False):
        """
        Fit model to Structural Equation Model data.
        
        Args:
            data: Input data (numpy array or PyTorch Dataset)
            dag: Directed acyclic graph adjacency matrix
            return_scores: Whether to return score
            
        Returns:
            score if return_scores=True, otherwise None
        """
        dset, dim = self._get_datasets(data)
        self.dim = dim
        temp_affine_layer = AffineCL(dim, dag)
        self.dag = dag
        self.parents = temp_affine_layer.parents
        self.root_nodes = temp_affine_layer.root_nodes        
        torch.manual_seed(self.config.training.seed)
        flows, all_loss_vals = self._train(dset)
        self.flow, score, self._nlxy, self._nhxy = self._evaluate(flows, dset)
        return score if return_scores else None

    def _get_optimizer(self, parameters):
        """
        Create optimizer and scheduler based on config.
        
        Args:
            parameters: Model parameters to optimize
            
        Returns:
            tuple of (optimizer, scheduler)
        """
        optimizer = optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                               betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        if self.config.optim.scheduler:
            # More aggressive scheduler: reduce LR by factor of 0.3 after 2 epochs without improvement
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=0.3,
                patience=2,
                min_lr=1e-5
            )
        else:
            scheduler = None
        return optimizer, scheduler

    def _get_flow_arch(self):
        """
        Construct flow architecture based on configuration.
        
        Returns:
            List of NormalizingFlowModel instances
        """
        # This method is called after self.dim has been initialized
        dim = self.dim
        
        # Create prior distribution
        if hasattr(self.config.flow, 'prior_dist'):
            if self.config.flow.prior_dist.lower() == 'gaussian':
                prior = LearnableGaussian(dim, device=self.device)
            else:
                raise ValueError(f"Unsupported prior distribution: {self.config.flow.prior_dist}")
        else:
            prior = LearnableGaussian(dim, device=self.device)
            
        net_class = MLP4
        
        # Flow creation function
        def ar_flow(hidden_dim):
            batch_norm = getattr(self.config.flow, 'batch_norm', False)
            return AffineCL(dim=dim, dag=self.dag, nh=hidden_dim, net_class=net_class,
                          scale=self.config.flow.scale, batch_norm=batch_norm)

        # Support varying depths and widths, keep only best
        self.n_layers = self.n_layers if isinstance(self.n_layers, list) else [self.n_layers]
        self.n_hidden = self.n_hidden if isinstance(self.n_hidden, list) else [self.n_hidden]
        normalizing_flows = []
        
        for nl in self.n_layers:
            for nh in self.n_hidden:
                # Construct normalizing flows
                flow_list = [ar_flow(nh) for _ in range(nl)]
                normalizing_flows.append(NormalizingFlowModel(prior, flow_list).to(self.device))
                
        return normalizing_flows

    def _get_params_from_idx(self, idx):
        """Get layer and hidden dimension parameters from index"""
        return self.n_layers[idx // len(self.n_hidden)], self.n_hidden[idx % len(self.n_hidden)]

    def _evaluate(self, flows, dset):
        """
        Evaluate flows on data and select best model.
        
        Args:
            flows: List of flow models to evaluate
            dset: Dataset to evaluate on
            
        Returns:
            tuple of (best_flow, best_score, n_layers, n_hidden)
        """
        with torch.no_grad():
            scores = []
            # Create a DataLoader for the evaluation dataset
            loader = DataLoader(dset, batch_size=self.config.training.batch_size)
            for idx, flow in enumerate(flows):
                flow.eval() # Ensure model is in eval mode
                # Calculate log likelihood score
                # Concatenate likelihoods from all batches in the loader
                log_likelihoods_all_batches = []
                for x_batch in loader:
                    x_batch = x_batch.to(self.device)
                    log_likelihoods_batch = flow.log_likelihood(x_batch)
                    log_likelihoods_all_batches.append(log_likelihoods_batch)
                
                score = np.nanmean(np.concatenate(log_likelihoods_all_batches))
                scores.append(score)
            
            # Find best flow
            try:
                idx = np.nanargmax(scores)
            except ValueError: # Handle cases where all scores might be NaN
                idx = 0 
            
            best_score = scores[idx] if scores else np.nan # Use scores[idx] or handle empty scores
            best_flow = flows[idx] if flows else None # Handle empty flows
            
            # Get architecture parameters
            nl, nh = self._get_params_from_idx(idx)
            
            return best_flow, best_score, nl, nh

    def _get_datasets(self, input_data):
        """
        Create training dataset from input data.
        
        Args:
            input_data: Input data (numpy array, Dataset, or tuple/list)
            
        Returns:
            tuple of (train_dataset, dimension)
        """
        assert isinstance(input_data, (np.ndarray, Dataset, tuple, list))
        if isinstance(input_data, np.ndarray):
            dim = input_data.shape[-1]
            dset = CustomSyntheticDatasetDensity(input_data.astype(np.float32))
            return dset, dim
        if isinstance(input_data, Dataset):
            # Assuming the first element of the dataset can give dimension
            # This might need adjustment based on actual Dataset structure
            dim = input_data[0].shape[-1] if len(input_data) > 0 and hasattr(input_data[0], 'shape') else None
            if dim is None:
                 # Attempt to get dim from the first element if it's a tensor
                try:
                    sample_data = input_data[0]
                    if torch.is_tensor(sample_data):
                        dim = sample_data.shape[-1]
                    else: # If it's a tuple/list of tensors, take the first one
                        dim = sample_data[0].shape[-1]
                except Exception as e:
                    raise ValueError(f"Could not determine dimension from Dataset: {e}")
            return input_data, dim

    def _forward_flow(self, data):
        """
        Apply forward flow transform to data.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data (numpy array)
        """
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def _backward_flow(self, latent):
        """
        Apply backward flow transform to latent.
        
        Args:
            latent: Latent representation
            
        Returns:
            Reconstructed data (numpy array)
        """
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def log_likelihood(self, data):
        """
        Calculate log likelihood of data.
        
        Args:
            data: Input data
            
        Returns:
            Log likelihood values (numpy array)
        """
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
            
        # Convert numpy array to tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data.astype(np.float32)).to(self.device)
            
        # Create DataLoader for batch processing
        dataset = CustomSyntheticDatasetDensity(data.cpu().numpy())
        loader = DataLoader(dataset, batch_size=128)
        
        log_likes = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                _, prior_logprob, log_det = self.flow(batch)
                log_like = prior_logprob + log_det
                log_likes.append(log_like.cpu().numpy())
                
        return np.concatenate(log_likes)
