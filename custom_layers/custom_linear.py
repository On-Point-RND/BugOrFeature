import copy
import torch
from typing import Any, Dict, Iterable

class SimpleCustomLinear(torch.nn.Linear):
    """
    Минимальный заменитель Linear: полностью эквивалентен nn.Linear,
    но хранит произвольные параметры из YAML в self.metadata.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 metadata: Dict[str, Any] | None = None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.metadata: Dict[str, Any] = metadata or {}

    @classmethod
    def from_original(cls, layer: torch.nn.Linear,
                      metadata: Dict[str, Any] | None = None,
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
