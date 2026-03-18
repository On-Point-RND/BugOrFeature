import inspect
from typing import List
from functools import partial

import torch.nn as nn

from modifiers.activations import ACTIVATION_NAMES_MAP, ActivationClass
from modifiers.normalizations import NORMALIZATION_NAMES_MAP, NormalizationClass



##########################################################################
#                Function to replace layers in a module                  #
##########################################################################

def replace_activation(
    module: nn.Module,
    original_activation: ActivationClass = 'GELU',
    replaced_activation: ActivationClass | nn.Module = 'ReLU',
    debug_info: bool = True,
) -> List[nn.Module]:
    assert original_activation in ACTIVATION_NAMES_MAP, f"Original activation '{original_activation}' is not supported."
    assert replaced_activation in ACTIVATION_NAMES_MAP or isinstance(replaced_activation, nn.Module), f"Replaced activation '{replaced_activation}' is not supported."

    original_cls = ACTIVATION_NAMES_MAP.get(original_activation)
    replaced_cls = ACTIVATION_NAMES_MAP.get(replaced_activation) if isinstance(replaced_activation, str) else replaced_activation

    print(f"Replacing activation: {original_cls} -> {replaced_cls}")
    
    resulting_layers: List[nn.Module] = []
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if isinstance(child, original_cls):
                if debug_info: 
                    print(child, "->", replaced_cls)

                # default parametes in replaced_csl are not necessarily the same as in original_cls, so we do not pass all parameters by default
                new_activation = replaced_cls()
                # FIXME: add debug_info passing in initialization
                if hasattr(new_activation, 'debug_info'):
                    new_activation.debug_info = debug_info
                setattr(layer, child_name, new_activation)
                resulting_layers.append(new_activation)

    print(f"Replaced {len(resulting_layers)} activation(s): {original_activation} -> {replaced_activation}")
    return resulting_layers

##########################################################################
#                Function to replace layers in a module                  #
##########################################################################

def replace_normalization(
    module: nn.Module,
    original_normalization:  nn.Module = 'RMSNorm',
    replaced_normalization: NormalizationClass |  nn.Module = 'LayerNorm',
) -> List[nn.Module]:
    assert original_normalization in NORMALIZATION_NAMES_MAP, f"Original normalization '{original_normalization}' is not supported."
    assert replaced_normalization in NORMALIZATION_NAMES_MAP or isinstance(replaced_normalization, nn.Module), f"Replaced normalization '{replaced_normalization}' is not supported."

    print(original_normalization, replaced_normalization)
    original_cls = NORMALIZATION_NAMES_MAP.get(original_normalization)
    replaced_cls = NORMALIZATION_NAMES_MAP.get(replaced_normalization) if isinstance(replaced_normalization, str) else replaced_normalization

    # Parameters that should NOT be passed to constructor (managed by nn.Module)
    module_attributes = {'training', 'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked', 'running_layer_mean'}
    
    resulting_layers: List[nn.Module] = []
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if child.__class__.__name__ == original_cls.__name__:
                # Get device from original layer
                original_device = next(child.parameters()).device if list(child.parameters()) else torch.device('cpu')
                
                # Filter out private PyTorch attributes and module-managed attributes
                valid_kwargs = {
                    k: v for k, v in child.__dict__.items() 
                    if not k.startswith('_') and k not in module_attributes
                }
                # Filter out arguments that couldn't be passed to constructor
                # var_names = set()
                # has_kwarg = False
                # for k, v in inspect.signature(replaced_cls).parameters.items():
                #     if v.kind == 4: # <_ParameterKind.VAR_KEYWORD: 4>
                #         has_kwarg = True
                #     elif v.kind == 1 or v.kind == 3: # <_ParameterKind.POSITIONAL_OR_KEYWORD: 1> | <_ParameterKind.KEYWORD_ONLY: 3>
                #         var_names.add(k)
                # if not has_kwarg:
                #     valid_kwargs = {
                #         k: v for k, v in valid_kwargs.items() 
                #         if k in var_names
                #     }
                
                new_normalization = replaced_cls(**valid_kwargs)
                
                # Copy weight if it exists in original and is not None
                if hasattr(child, 'weight'):
                    weight = getattr(child, 'weight')
                    if weight is not None:
                        # Delete old parameter first, then register new one
                        del new_normalization.weight
                        new_normalization.register_parameter('weight', weight)
                
                # Copy bias if it exists in original and is not None
                if hasattr(child, 'bias'):
                    bias = getattr(child, 'bias')
                    if bias is not None:
                        if hasattr(new_normalization, 'bias') and new_normalization.bias is not None:
                            del new_normalization.bias
                        new_normalization.register_parameter('bias', bias)
                else:
                    # Original has no bias - remove bias from new layer to avoid device mismatch
                    if hasattr(new_normalization, 'bias') and new_normalization.bias is not None:
                        del new_normalization.bias
                        new_normalization.register_parameter('bias', None)
                    
                setattr(layer, child_name, new_normalization)
                resulting_layers.append(new_normalization)

    print(f"Replaced {len(resulting_layers)} normalization(s): {original_normalization} -> {replaced_normalization}")
    return resulting_layers


relufiaction = partial(replace_activation, replaced_activation='ReLU')