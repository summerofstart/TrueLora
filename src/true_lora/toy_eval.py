from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from true_lora.adapter import LoraTensorSpec, infer_lora_tensor_specs
from true_lora.apply import temporary_lora


class ToyClassifier(nn.Module):
    def __init__(self, in_features: int = 4, classes: int = 2) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, classes, bias=False)
        nn.init.zeros_(self.layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


@dataclass(frozen=True)
class ToyTask:
    x: torch.Tensor
    y: torch.Tensor
    specs: list[LoraTensorSpec]


def make_sign_task(
    samples: int = 128,
    in_features: int = 4,
    out_features: int = 2,
    rank: int = 2,
    seed: int = 13,
) -> ToyTask:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, in_features, generator=generator)
    y = (x[:, 0] + x[:, 1] > 0).long()
    specs = [LoraTensorSpec("layer", out_features=out_features, in_features=in_features, rank=rank, alpha=float(rank))]
    return ToyTask(x=x, y=y, specs=specs)


def adapter_for_sign_task(in_features: int = 4) -> dict[str, torch.Tensor]:
    a = torch.zeros(2, in_features)
    a[0, 0] = 1.0
    a[1, 1] = 1.0
    b = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    return {"layer.lora_A.weight": a, "layer.lora_B.weight": b}


def accuracy_with_adapter(state_dict: dict[str, torch.Tensor], task: ToyTask | None = None) -> dict[str, float]:
    specs = infer_lora_tensor_specs(state_dict)
    layer_spec = next((spec for spec in specs if spec.name == "layer"), None)
    if layer_spec is None:
        raise ValueError("Toy evaluation requires LoRA tensors named layer.lora_A.weight/layer.lora_B.weight")

    task = task or make_sign_task(
        in_features=layer_spec.in_features,
        out_features=layer_spec.out_features,
        rank=layer_spec.rank,
    )
    model = ToyClassifier(in_features=task.x.shape[1], classes=layer_spec.out_features)
    baseline = _accuracy(model, task.x, task.y)
    with temporary_lora(model, state_dict, task.specs):
        adapted = _accuracy(model, task.x, task.y)
    restored = _accuracy(model, task.x, task.y)
    return {"baseline_accuracy": baseline, "adapted_accuracy": adapted, "restored_accuracy": restored}


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        return float((pred == y).float().mean())
