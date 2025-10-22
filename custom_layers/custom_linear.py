import copy
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

class SimpleCustomLinear(torch.nn.Linear):
    """
    Linear layer that records:
      - input sparsity (global over full input tensor)
      - FLOPs per token (i.e., per non-batch element)
      - input shape without batch dimensions
    Stores everything in self.metadata.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.metadata: Dict[str, Any] = metadata or {}

    def record_input_sparsity_and_flops(self, input: torch.Tensor):
        # --- Input shape without batch dims ---
        # Assume last dimension is in_features; everything before is "batch-like"
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Input last dim {input.shape[-1]} != in_features {self.in_features}")
        seq_len_emb_size = [input.shape[1], input.shape[-1]]  # e.g., [seq_len] for (B, seq_len, D)

        # --- Sparsity (computed over entire tensor) ---
        total_elements = input.numel()
        zero_elements = (input == 0).sum().item()
        sparsity = zero_elements / total_elements if total_elements > 0 else 0.0

        # --- FLOPs per token (per vector of size in_features) ---
        flops_per_token = 2 * self.in_features * self.out_features  # dense
        sparse_flops_per_token = flops_per_token * (1.0 - sparsity)

        # --- Update metadata ---
        self.metadata.update({
            'input_sparsity': sparsity,
            'sequence_length_emb_size': seq_len_emb_size,  # e.g., [1024] or [128, 64]
            'flops_per_token_dense': flops_per_token,
            'flops_per_token_sparse': sparse_flops_per_token,
        })

    def forward(self, input: torch.Tensor):
        self.record_input_sparsity_and_flops(input)
        return F.linear(input, self.weight, self.bias)

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