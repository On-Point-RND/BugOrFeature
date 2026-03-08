from typing import Optional, Type

import torch
import torch.nn as nn


##########################################################################
#                    Decorator for analizing modules                     #
##########################################################################

def analytical_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Decorator to create an analytical version of a given nn.Module class. The resulting class will have additional attributes to store the input and output activations, as well as a debug_info flag to control whether these activations are stored during the forward pass.
    """

    class AnalyticalModule(cls):
        def __init__(
            self,
            *args,
            debug_info: bool = False,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.debug_info = debug_info
            self.in_activation = None
            self.out_activation = None
            
        def forward(self, x):
            if self.debug_info:
                self.in_activation = x
            x = super().forward(x)
            if self.debug_info:
                self.out_activation = x
            return x
        
        def extra_repr(self) -> str:
            return f'debug_info={self.debug_info}, {super().extra_repr()}'
        
    AnalyticalModule.__name__ = f"Analytical{cls.__name__}"
        
    return AnalyticalModule


##########################################################################
#                    Decorator for sparse activations                    #
##########################################################################

def topk_sparse_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Decorator to create a sparse version of a given nn.Module class. The resulting class will have an additional attribute sparsity_level to control the level of sparsity applied to the activations during the forward pass. The sparsity is applied by zeroing out the smallest activations based on the specified sparsity level.
    """

    class SparseModule(cls):
        def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            post_sparsity: bool = True,
            **kwargs
        ):
            super().__init__(*args, **kwargs)

            assert sparsity_level is None or (0.0 < sparsity_level < 1.0), "sparsity_level must be in (0, 1)"

            self.sparsity_level = sparsity_level
            self.post_sparsity = post_sparsity

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.post_sparsity:
                x = super().forward(x)

            if self.sparsity_level is not None:
                # Get batch size while preserving other dimensions
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)  # [batch, rest]
                total_elements = x_flat.size(-1)  # per-sample element count
                n_keep = int((1.0 - self.sparsity_level) * total_elements)
                
                kth_values = torch.kthvalue(x_flat, n_keep, dim=-1).values  # [batch]
                # Reshape kth_values to match x's dimensions for broadcasting
                kth_values = kth_values.view(batch_size, *([1] * (x.dim() - 1)))
                mask = x < kth_values
                x = x.masked_fill(mask, 0.0)

            if not self.post_sparsity:
                x = super().forward(x)

            return x
        
        def extra_repr(self) -> str:
            return f'sparsity_level={self.sparsity_level}, {super().extra_repr()}'
        
    SparseModule.__name__ = f"TopK{cls.__name__}"

    return SparseModule

