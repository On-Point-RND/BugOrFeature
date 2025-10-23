import copy
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from .base_auxiliary_loss import AuxiliaryLoss


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

    def forward(self, input: torch.Tensor):
        self.record_input_sparsity_and_flops(input)
        output = F.linear(input, self.weight, self.bias)
        
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