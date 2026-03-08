from typing import Optional, Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modifiers.decorators import analytical_module, topk_sparse_module


##########################################################################
#                       Extra activation classes                         #
##########################################################################

# ReLU^2


class ReLUSquaredFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass (needed to compute mask)
        ctx.save_for_backward(input)
        # Compute forward: (max(0, x))^2
        # Note: Intermediate tensors created here are NOT saved in the graph
        output = F.relu(input)
        return torch.square(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Gradient of (ReLU(x))^2 is: 2 * ReLU(x) * (1 if x>0 else 0)
        # Which simplifies to: 2 * ReLU(x)
        # We recompute ReLU(x) from input instead of saving it from forward
        relu_input = F.relu(input)
        grad_input = 2 * relu_input * grad_output
        return grad_input

@analytical_module
class ReLUSquared(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return ReLUSquaredFunction.apply(input)


# B-SiLU

@analytical_module
class BSiLU(nn.SiLU):
    """
    BSiLU is a modified version of the SiLU (Sigmoid Linear Unit) activation function, defined as:
    f(x) = (x + alpha) * sigmoid(x) - alpha / 2
    where alpha is a hyperparameter that controls the shape of the function. The BSiLU activation can provide smoother gradients compared to ReLU and may help with training stability in certain neural network architectures.

    see: https://arxiv.org/html/2505.22074v1 for more details on B-SiLU and its properties.
    """

    def __init__(self, *args, alpha=1.67, **kwargs):
        super().__init__(*args, **kwargs)

        assert getattr(self, 'inplace', False) == False, "Without triton kernel it is impossible to make inplace B-SiLU"

        self.alpha = alpha

    def forward(self, input):
        sigma_x = torch.sigmoid(input)
        return (input + self.alpha) * sigma_x - self.alpha / 2.0
    
    def backward(self, grad_output):
        # Derivative of B-SiLU: sigma(x) + (x + alpha) * sigma(x) * (1 - sigma(x))
        x = self.in_activation
        sigma_x = torch.sigmoid(x)
        b_silu_grad = sigma_x + (x + self.alpha) * sigma_x * (1.0 - sigma_x)
        return grad_output * b_silu_grad


# Sugar B-SiLU

@analytical_module
class SUGARBSiLU(nn.ReLU):
    """
    SUGAR-BSiLU is a variant of the surrogate gradient activation function that combines the properties of ReLU and B-SiLU. It is defined as:

    see: https://arxiv.org/html/2505.22074v1 for more details on SUGAR-BSiLU and its properties.
    """

    def backward(self, grad_output):
        return BSiLU.backward(self, grad_output)


# Noisy ReLU

@analytical_module
class NoisyReLU(nn.ReLU):
    """
    NoisyReLU is a variant of the ReLU activation function that adds noise to the output during training. 
    The noise is generated based on the negative part of the input, and its scale is controlled by a 
    learnable parameter p and a hyperparameter c. The noise can help regularize the model and improve 
    generalization by preventing overfitting to the training data.

    See: https://arxiv.org/pdf/1603.00391 for more details on NoisyReLU and its properties.
    """

    def __init__(
        self,
        *args,
        alpha=1.0,
        c=1.0,
        noise_type='half-normal',
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert getattr(self, 'inplace', False) == False, "Without triton kernel it is impossible to make inplace NoisyReLU"

        self.alpha = alpha
        self.c = c
        self.noise_type = noise_type
        # Registered parameter - will move with module via .to(device) when used correctly
        self.p = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        if not self.training:
            return F.relu(x)
        
        # SAFETY: Ensure p is on same device as input (handles edge cases where .to() wasn't called properly)
        p = self.p.to(x.device)
        
        # Training time with noise
        mask = x < 0
        
        # Create explicit zero tensor on same device/dtype as input
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        delta = torch.where(mask, x, zero)
        
        # Use device-aligned p
        sigma = delta.mul(-p).sigmoid().sub(0.5).square()
        
        # randn_like respects device/dtype of x
        epsilon = torch.randn_like(x)
        if self.noise_type == 'half-normal':
            epsilon.abs_()
        
        # Use explicit zero for masked_fill
        noise = sigma.mul(epsilon.mul_(self.c)).masked_fill_(mask, zero)
        
        x = F.leaky_relu(x, (1 - self.alpha)) + noise
        
        return x
    

# Quantile-based ReLU

@analytical_module
class QuantileReLU(nn.ReLU):
    def __init__(
        self,
        sparsity_level: Optional[float] = None,
        shifted_sparsity: bool = False,
        signed = True,
        continuous = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sparsity_level = sparsity_level
        self.shifted_sparsity = shifted_sparsity
        self.signed = signed
        self.continuous = continuous

    def forward(self, input):
        if not self.signed:
            sign_mask = torch.ones_like(input)
            sign_mask.masked_fill_(input < 0, -1)
            input = input.abs()
        
        if self.sparsity_level is None:
            output = super().forward(input)
        
        elif self.shifted_sparsity or self.continuous:
            # Find k-th value for each batch element
            # FIXME: Calculate quantiles along the batch
            kth_values = torch.quantile(input, self.sparsity_level, dim=0, keepdim=True)
            output = super().forward(input - kth_values)
            if not self.continuous:
                output = output + kth_values
        
        else:
            mask = input >= torch.quantile(input, self.sparsity_level, dim=0, keepdim=True)
            output = input * mask
            
        if not self.signed:
            input = input * sign_mask
        
        return output
    
    def extra_repr(self) -> str:
        return (f'sparsity_level={self.sparsity_level}, shifted_sparsity={self.shifted_sparsity}, '
                f'signed={self.signed}, continuous={self.continuous}, {super().extra_repr()}')
    

##########################################################################
#                           Sparse activations                           #
##########################################################################

@analytical_module
@topk_sparse_module
class TopKSparseGELU(nn.GELU):
    """
    TopKSparseGELU is a variant of the GELU activation function that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


##########################################################################
#          Mapping from string names to activation classes               #
##########################################################################

ACTIVATION_NAMES_MAP = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,

    'AReLU': analytical_module(nn.ReLU),
    'AGELU': analytical_module(nn.GELU),
    'ASiLU': analytical_module(nn.SiLU),

    'ReLUSquared': ReLUSquared,

    'BSiLU': BSiLU,
    'SUGARBSiLU': SUGARBSiLU,
    'NoisyReLU': NoisyReLU,

    'QuantileReLU': QuantileReLU,
    'QuantileReLU-10': partial(QuantileReLU, sparsity_level=0.10),
    'QuantileReLU-25': partial(QuantileReLU, sparsity_level=0.25),
    'QuantileReLU-50': partial(QuantileReLU, sparsity_level=0.50),
    'QuantileReLU-75': partial(QuantileReLU, sparsity_level=0.75),
    'QuantileReLU-90': partial(QuantileReLU, sparsity_level=0.90),

    'TopKSparseGELU': TopKSparseGELU,
    'TopKSparseGELU-10': partial(TopKSparseGELU, sparsity_level=0.10),
    'TopKSparseGELU-25': partial(TopKSparseGELU, sparsity_level=0.25),
    'TopKSparseGELU-50': partial(TopKSparseGELU, sparsity_level=0.50),
    'TopKSparseGELU-75': partial(TopKSparseGELU, sparsity_level=0.75),
    'TopKSparseGELU-90': partial(TopKSparseGELU, sparsity_level=0.90),
}

ActivationClass = Literal[
    'ReLU', 'GELU', 'SiLU',
    'AReLU', 'AGELU', 'ASiLU',
    'ReLUSquared',
    'BSiLU', 'SUGARBSiLU', 'NoisyReLU',
    'QuantileReLU', 'QuantileReLU-10', 'QuantileReLU-25', 'QuantileReLU-50', 'QuantileReLU-75', 'QuantileReLU-90',
    'TopKSparseGELU', 'TopKSparseGELU-10', 'TopKSparseGELU-25', 'TopKSparseGELU-50', 'TopKSparseGELU-75', 'TopKSparseGELU-90',
]
