import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Optional
from .base_auxiliary_loss import AuxiliaryLoss

def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Compute the effective rank of a matrix based on singular values"""
    if matrix.numel() == 0:
        return 0.0
    
    # Reshape matrix to 2D if needed
    if matrix.dim() > 2:
        matrix = matrix.view(-1, matrix.shape[-1])
    
    # Compute SVD
    U, S, V = torch.svd(matrix.float())
    
    # Compute singular values as probabilities
    S_squared = S ** 2
    prob = S_squared / S_squared.sum()
    
    # Filter out very small probabilities to avoid log(0)
    prob = prob[prob > 1e-10]
    
    # Compute entropy
    entropy = -torch.sum(prob * torch.log(prob))
    
    # Effective rank is exp(entropy)
    effective_rank = torch.exp(entropy).item()
    
    return effective_rank

class SimpleCustomLinear(torch.nn.Linear, AuxiliaryLoss):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.metadata: Dict[str, Any] = metadata or {}
        self._aux_loss_config = self.metadata.get('aux_loss_config', {})
        self._aux_loss_enabled = self._aux_loss_config.get('enabled', False)
        self._cached_input = None
        self._cached_output = None

    def record_input_sparsity_and_flops(self, input: torch.Tensor):
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Input last dim {input.shape[-1]} != in_features {self.in_features}")
        
        seq_len_emb_size = [input.shape[1], input.shape[-1]]
        total_elements = input.numel()
        zero_elements = (input == 0).sum().item()
        sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
        flops_per_token = 2 * self.in_features * self.out_features
        sparse_flops_per_token = flops_per_token * (1.0 - sparsity)

        self.metadata.update({
            'input_sparsity': sparsity,
            'sequence_length_emb_size': seq_len_emb_size,
            'flops_per_token_dense': flops_per_token,
            'flops_per_token_sparse': sparse_flops_per_token,
        })

    def compute_statistics(self, input: torch.Tensor, output: torch.Tensor):
        """Compute additional statistics when in eval mode"""
        if self.training:
            return
            
        stats = {}
        
        # Percentage of negative/positive values in input
        input_total = input.numel()
        if input_total > 0:
            input_neg = (input < 0).sum().item()
            input_pos = (input > 0).sum().item()
            stats['input_negative_percentage'] = input_neg / input_total
            stats['input_positive_percentage'] = input_pos / input_total
        
        # Percentage of negative/positive values in output
        output_total = output.numel()
        if output_total > 0:
            output_neg = (output < 0).sum().item()
            output_pos = (output > 0).sum().item()
            stats['output_negative_percentage'] = output_neg / output_total
            stats['output_positive_percentage'] = output_pos / output_total
        
        # Percentage of negative/positive values in weight matrix
        weight_total = self.weight.numel()
        if weight_total > 0:
            weight_neg = (self.weight < 0).sum().item()
            weight_pos = (self.weight > 0).sum().item()
            stats['weight_negative_percentage'] = weight_neg / weight_total
            stats['weight_positive_percentage'] = weight_pos / weight_total
        
        # Bias mean (if bias exists)
        if self.bias is not None:
            stats['bias_mean'] = self.bias.mean().item()
        
        # Input and output effective rank
        # stats['input_effective_rank'] = compute_effective_rank(input)
        # stats['output_effective_rank'] = compute_effective_rank(output)
        # stats['weight_effective_rank'] = compute_effective_rank(self.weight)
        
        # Update metadata with new statistics
        self.metadata.update(stats)

    def forward(self, input: torch.Tensor):
        
        output = F.linear(input, self.weight, self.bias)
        
        # Compute additional statistics in eval mode only
        if not self.training:
            self.record_input_sparsity_and_flops(input)
            self.compute_statistics(input, output)
        
        if self.training and self._aux_loss_enabled:
            self._cached_input = input.detach()
            self._cached_output = output.detach()
        
        return output
    
    def compute_auxiliary_loss(self) -> torch.Tensor:
        """
        Override to add custom loss. Available: self._cached_input, self._cached_output, 
        self.weight, self._aux_loss_config
        
        Example:
            if 'my_loss' in self._aux_loss_config:
                coef = self._aux_loss_config['my_loss']
                loss = torch.abs(self._cached_input).mean() * coef
                return loss
        """
        if not self.training or not self._aux_loss_enabled:
            return torch.tensor(0.0, device=self._get_device())
        
        total_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        if 'l1_activation' in self._aux_loss_config:
            coef = self._aux_loss_config['l1_activation']
            if coef > 0 and self._cached_input is not None:
                l1_loss = torch.abs(self._cached_input).mean() * coef
                total_loss = total_loss + l1_loss
        
        return total_loss
    
    def should_compute_aux_loss(self) -> bool:
        return self.training and self._aux_loss_enabled

    @classmethod
    def from_original(cls, layer: torch.nn.Linear,
                      metadata: Optional[Dict[str, Any]] = None,
                      copy_weights: bool = True) -> "SimpleCustomLinear":
        new_layer = cls(
            layer.in_features,
            layer.out_features,
            bias=(layer.bias is not None),
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            metadata=copy.deepcopy(metadata or {}),
        )
        if copy_weights:
            new_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias.data)
        return new_layer