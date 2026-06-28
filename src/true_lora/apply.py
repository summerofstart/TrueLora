from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from true_lora.adapter import LoraTensorSpec


def lora_delta(a: torch.Tensor, b: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("LoRA A and B tensors must be matrices")
    if a.shape[0] != b.shape[1]:
        raise ValueError(f"rank mismatch: A has rank {a.shape[0]}, B has rank {b.shape[1]}")
    rank = float(a.shape[0])
    scale = rank if alpha is None else float(alpha)
    return (b.float() @ a.float()) * (scale / rank)


def named_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def merge_lora_into_linear(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    strict: bool = True,
) -> list[str]:
    modules = named_linear_modules(model)
    applied: list[str] = []

    with torch.no_grad():
        for spec in specs:
            if spec.name not in modules:
                if strict:
                    raise KeyError(f"Model has no nn.Linear module named {spec.name!r}")
                continue
            module = modules[spec.name]
            a_name = f"{spec.name}.lora_A.weight"
            b_name = f"{spec.name}.lora_B.weight"
            if a_name not in state_dict or b_name not in state_dict:
                if strict:
                    raise KeyError(f"Missing LoRA tensors for {spec.name!r}")
                continue

            delta = lora_delta(state_dict[a_name], state_dict[b_name], alpha=spec.alpha)
            if tuple(delta.shape) != tuple(module.weight.shape):
                raise ValueError(
                    f"{spec.name} delta shape {tuple(delta.shape)} does not match "
                    f"module weight shape {tuple(module.weight.shape)}"
                )
            module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
            applied.append(spec.name)

    return applied


@contextmanager
def temporary_lora(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    strict: bool = True,
) -> Iterator[list[str]]:
    # Only clone weights for modules that will be modified (memory optimization)
    target_names = {spec.name for spec in specs}
    all_modules = dict(model.named_modules())
    originals = {
        name: module.weight.detach().clone()
        for name, module in all_modules.items()
        if name in target_names
    }
    applied = merge_lora_into_linear(model, state_dict, specs, strict=strict)
    try:
        yield applied
    finally:
        # Restore only modified weights (faster iteration)
        with torch.no_grad():
            for name, original_weight in originals.items():
                if name in all_modules:
                    all_modules[name].weight.copy_(original_weight)
