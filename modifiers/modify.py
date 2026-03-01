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


def replace_normalization(
    module: nn.Module,
    original_normalization: NormalizationClass = 'BatchNorm2d',
    replaced_normalization: NormalizationClass | nn.Module = 'SparseBatchNorm2dQuantile50',
) -> List[nn.Module]:
    assert original_normalization in NORMALIZATION_NAMES_MAP, f"Original normalization '{original_normalization}' is not supported."
    assert replaced_normalization in NORMALIZATION_NAMES_MAP or isinstance(replaced_normalization, nn.Module), f"Replaced normalization '{replaced_normalization}' is not supported."

    original_cls = NORMALIZATION_NAMES_MAP.get(original_normalization)
    replaced_cls = NORMALIZATION_NAMES_MAP.get(replaced_normalization) if isinstance(replaced_normalization, str) else replaced_normalization

    resulting_layers: List[nn.Module] = []
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if isinstance(child, original_cls):
                new_activation = replaced_cls(**child.__dict__)
                setattr(layer, child_name, new_activation)
                resulting_layers.append(new_activation)

    print(f"Replaced {len(resulting_layers)} normalization(s): {original_normalization} -> {replaced_normalization}")
    return resulting_layers


relufiaction = partial(replace_activation, replaced_activation='ReLU')