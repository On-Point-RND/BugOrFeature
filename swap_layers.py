import torch
from typing import Any, Dict

from custom_layers.custom_linear import SimpleCustomLinear

def _module_dict(root: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return {name: module for name, module in root.named_modules()}


def apply_simple_linear_swaps(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    logging: Any = print,
) -> int:
    """
    Делает замену nn.Linear -> SimpleCustomLinear по правилам из YAML.

    Ожидаемая секция в cfg:
      model:
        layer_swap_simple:
          enabled: true
          copy_weights: true
          include_leaf_names: []              # опционально: если не пусто — менять только эти имена "листьев"
          exclude_substrings: ["attn", "q_", "k_", "v_", "up", "down", "proj", "attention"]
          default_custom_layer_config: { ... }  # дефолтный конфиг для всех замен
          per_layer_config:                   # переопределения (по полному пути или по листу)
            "lm_head": {alpha: 0.1}
            "transformer.h.0.mlp.fc_out": {beta: 0.2}
    """
    section = (cfg.get("model") or {}).get("layer_swap_simple")
    if not section or not section.get("enabled", False):
        logging("[layer_swap_simple] disabled/no section — skip")
        return 0

    copy_weights = bool(section.get("copy_weights", True))
    include_leaf = set(section.get("include_leaf_names") or [])
    exclude_substrings = [s.lower() for s in (section.get("exclude_substrings") or [])]
    default_cfg: Dict[str, Any] = section.get("default_custom_layer_config") or {}
    per_layer_cfg: Dict[str, Dict[str, Any]] = section.get("per_layer_config") or {}

    replaced = 0
    mdict = _module_dict(model)

    for full_name, module in list(mdict.items()):
        if not isinstance(module, torch.nn.Linear):
            continue

        leaf = full_name.split(".")[-1]
        lname = full_name.lower()

        if any(substr in lname for substr in exclude_substrings):
            continue

        if include_leaf and (leaf not in include_leaf and full_name not in include_leaf):
            continue

        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        parent = mdict[parent_name]

        effective_cfg = dict(default_cfg)
        if leaf in per_layer_cfg:
            effective_cfg.update(per_layer_cfg[leaf])
        if full_name in per_layer_cfg:
            effective_cfg.update(per_layer_cfg[full_name])

        new_linear = SimpleCustomLinear.from_original(
            module,
            metadata=effective_cfg,
            copy_weights=copy_weights,
        )
        setattr(parent, leaf, new_linear)
        replaced += 1
        logging(f"[layer_swap_simple] replaced {full_name} -> SimpleCustomLinear")

    logging(f"[layer_swap_simple] total replaced: {replaced}")
    return replaced
