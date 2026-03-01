# Sparse-Activations

## Modifiers

The `modifiers` package is a collection of drop-in replacements for standard PyTorch activation functions and normalization layers, with a focus on sparsity-inducing variants.

### Activations

Most of these are decorated with `@analytical_module`, which optionally stores input/output tensors on the forward pass.

- **ReLUSquared** - Just ReLU followed by squaring: $f(x) = (\max(0, x))^2$.
- **BSiLU** - A shifted SiLU variant: $f(x) = (x + \alpha) \cdot \sigma(x) - \alpha/2$. Comes from [this paper](https://arxiv.org/html/2505.22074v1). Smoother gradients than ReLU, and `alpha` is configurable.
- **SUGARBSiLU** - Uses ReLU in the forward pass but BSiLU's gradient in the backward pass (surrogate gradient trick). Same paper as above.
- **NoisyReLU** - Adds learnable noise during training based on the negative part of the input. The noise scale is controlled by a parameter `p` and a constant `c`. Based on [this paper](https://arxiv.org/pdf/1603.00391).
- **QuantileReLU** - Zeros out activations below a given quantile threshold instead of just below zero. Supports several modes: shifted sparsity, unsigned, continuous, etc.
- **TopKSparseGELU** - GELU but only the top-k% of activations survive (the rest get zeroed). Uses the `@topk_sparse_module` decorator under the hood.

All activations are accessible by string name (e.g. `'ReLUSquared'`, `'TopKSparseGELU-50'`) through a built-in name map, so you don't have to import classes manually if you don't want to. Presets with common sparsity levels (10%, 25%, 50%, 75%, 90%) are included.

### Normalizations

Custom normalization layers that shift the centering point from the mean to a quantile - this effectively biases the normalization toward sparser outputs.

- **QuantileBatchNorm2d** - Like BatchNorm2d, but uses a quantile of the activations as the mean estimate. Supports global, batchwise, or channelwise quantile computation.
- **QuantileMeanBatchNorm2d** - Same quantile-based mean, but keeps the standard variance calculation. A middle ground.
- **QuantileLayerNorm** - LayerNorm variant with quantile-based centering. Also supports running stats tracking and configurable quantile search modes.

Same deal as activations - string names with sparsity presets are available (e.g. `'QuantileBatchNorm2d-50'`).

### Decorators

Two module decorators that can wrap any `nn.Module`:

- `@analytical_module` - Adds `in_activation` / `out_activation` attributes that capture tensors during forward. Toggle with `debug_info=True/False`.
- `@topk_sparse_module` - Adds top-k sparsity to any activation. Set `sparsity_level` (0 to 1) and choose whether sparsity is applied before or after the base activation with `post_sparsity`.

### Replacing layers in an existing model

```python
from modifiers import replace_activation, replace_normalization

# Swap all GELU activations for ReLUSquared
replace_activation(model, original_activation='GELU', replaced_activation='ReLUSquared')

# Swap BatchNorm2d for quantile-based variant at 50% sparsity
replace_normalization(model, original_normalization='BatchNorm2d', replaced_normalization='QuantileBatchNorm2d-50')
```

Both functions walk the module tree recursively and return a list of the newly created layers, in case you need to track them.
