import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class AuxiliaryLoss:
    def should_compute_aux_loss(self) -> bool:
        if not hasattr(self, 'training'):
            return False
        return self.training
    
    def compute_auxiliary_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self._get_device())
    
    def _get_device(self) -> torch.device:
        if hasattr(self, 'weight') and self.weight is not None:
            return self.weight.device
        return torch.device('cpu')
    
    def get_aux_loss_weight(self) -> float:
        if hasattr(self, '_aux_loss_config'):
            return self._aux_loss_config.get('weight', 1.0)
        return 1.0


def collect_auxiliary_losses(
    model: nn.Module, 
    weighted: bool = True,
    return_details: bool = False
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
    total_loss = None
    details = {}
    count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'compute_auxiliary_loss') and callable(module.compute_auxiliary_loss):
            try:
                aux_loss = module.compute_auxiliary_loss()
                
                if aux_loss is None or (torch.is_tensor(aux_loss) and aux_loss.item() == 0):
                    continue
                
                weight = 1.0
                if weighted and hasattr(module, 'get_aux_loss_weight'):
                    weight = module.get_aux_loss_weight()
                
                weighted_loss = aux_loss * weight
                
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss
                
                if return_details:
                    details[name] = {
                        'loss': aux_loss.item() if torch.is_tensor(aux_loss) else float(aux_loss),
                        'weight': weight,
                        'weighted_loss': weighted_loss.item() if torch.is_tensor(weighted_loss) else float(weighted_loss),
                    }
                
                count += 1
                
            except Exception as e:
                if return_details:
                    details[name] = {'error': str(e)}
    
    if total_loss is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if return_details:
        details['_summary'] = {
            'total_loss': total_loss.item(),
            'num_layers': count,
        }
        return total_loss, details
    
    return total_loss


def enable_auxiliary_losses(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, '_aux_loss_enabled'):
            module._aux_loss_enabled = True
            count += 1
    return count


def disable_auxiliary_losses(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, '_aux_loss_enabled'):
            module._aux_loss_enabled = False
            count += 1
    return count

