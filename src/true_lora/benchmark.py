from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn

from true_lora.adapter import LoraTensorSpec, infer_lora_tensor_specs
from true_lora.apply import temporary_lora


@dataclass(frozen=True)
class ClassificationBenchmark:
    x: torch.Tensor
    y: torch.Tensor
    module_name: str
    classes: int
    specs: list[LoraTensorSpec]


class LinearBenchmarkModel(nn.Module):
    def __init__(self, in_features: int, classes: int, module_name: str = "layer") -> None:
        super().__init__()
        if module_name != "layer":
            raise ValueError("LinearBenchmarkModel currently exposes one module named 'layer'")
        self.layer = nn.Linear(in_features, classes, bias=False)
        nn.init.zeros_(self.layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def load_classification_jsonl(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    module_name: str = "layer",
) -> ClassificationBenchmark:
    specs = infer_lora_tensor_specs(state_dict)
    spec = next((item for item in specs if item.name == module_name), None)
    if spec is None:
        raise ValueError(f"Adapter has no LoRA tensors for module {module_name!r}")

    features: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            vector = row.get("features")
            label = row.get("label")
            if not isinstance(vector, list) or not isinstance(label, int):
                raise ValueError(f"{path}:{line_number} requires features:list and label:int")
            features.append([float(value) for value in vector])
            labels.append(label)

    if not features:
        raise ValueError(f"{path} did not contain benchmark examples")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise ValueError(f"{path} contains inconsistent feature widths")
    if width != spec.in_features:
        raise ValueError(f"Benchmark width {width} does not match adapter in_features {spec.in_features}")
    if max(labels) >= spec.out_features or min(labels) < 0:
        raise ValueError("Benchmark labels are outside adapter output range")

    return ClassificationBenchmark(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        module_name=module_name,
        classes=spec.out_features,
        specs=[spec],
    )


def evaluate_classification(
    state_dict: dict[str, torch.Tensor],
    benchmark: ClassificationBenchmark,
) -> dict[str, float]:
    model = LinearBenchmarkModel(
        in_features=benchmark.x.shape[1],
        classes=benchmark.classes,
        module_name=benchmark.module_name,
    )
    baseline = _accuracy(model, benchmark.x, benchmark.y)
    with temporary_lora(model, state_dict, benchmark.specs):
        adapted = _accuracy(model, benchmark.x, benchmark.y)
    restored = _accuracy(model, benchmark.x, benchmark.y)
    return {
        "baseline_accuracy": baseline,
        "adapted_accuracy": adapted,
        "restored_accuracy": restored,
        "examples": float(benchmark.x.shape[0]),
        "classes": float(benchmark.classes),
    }


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        return float((pred == y).float().mean())
