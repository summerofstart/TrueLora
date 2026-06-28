from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from true_lora.adapter import LoraTensorSpec, filter_lora_tensors, infer_lora_tensor_specs, load_torch_state_dict


def has_peft() -> bool:
    return importlib.util.find_spec("peft") is not None


def has_safetensors() -> bool:
    return importlib.util.find_spec("safetensors") is not None


def load_peft_directory(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    safe_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safe_path.exists():
        if not has_safetensors():
            raise RuntimeError("safetensors is required to load adapter_model.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(str(safe_path), device="cpu")
    elif bin_path.exists():
        state_dict = load_torch_state_dict(bin_path)
    else:
        raise FileNotFoundError(f"{path} has no adapter_model.safetensors or adapter_model.bin")

    return filter_lora_tensors(state_dict), config


def specs_from_peft_config(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> list[LoraTensorSpec]:
    true_lora_specs = config.get("true_lora_tensor_specs")
    if isinstance(true_lora_specs, list) and true_lora_specs:
        specs: list[LoraTensorSpec] = []
        for row in true_lora_specs:
            if not isinstance(row, dict):
                continue
            specs.append(
                LoraTensorSpec(
                    str(row["name"]),
                    out_features=int(row["out_features"]),
                    in_features=int(row["in_features"]),
                    rank=int(row["rank"]),
                    alpha=float(row.get("alpha", 1.0)),
                )
            )
        if specs:
            return specs

    inferred = infer_lora_tensor_specs(state_dict)
    rank = int(config.get("r", 0) or 0)
    alpha = float(config.get("lora_alpha", 0.0) or 0.0)
    if rank <= 0 and alpha <= 0:
        return inferred

    specs: list[LoraTensorSpec] = []
    for spec in inferred:
        specs.append(
            LoraTensorSpec(
                spec.name,
                out_features=spec.out_features,
                in_features=spec.in_features,
                rank=rank or spec.rank,
                alpha=alpha or spec.alpha,
            )
        )
    return specs


def inspect_peft_directory(path: Path) -> dict[str, Any]:
    state_dict, config = load_peft_directory(path)
    specs = specs_from_peft_config(state_dict, config)
    return {
        "path": str(path),
        "peft_type": config.get("peft_type"),
        "base_model_name_or_path": config.get("base_model_name_or_path", ""),
        "target_modules": config.get("target_modules", []),
        "tensor_count": len(state_dict),
        "spec_count": len(specs),
        "specs": [spec.__dict__ for spec in specs],
    }


def load_peft_model(base_model: Any, adapter_dir: Path, **kwargs: Any) -> Any:
    if not has_peft():
        raise RuntimeError("peft is not installed; install peft to load adapters into Hugging Face models")
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, str(adapter_dir), **kwargs)
