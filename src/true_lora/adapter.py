from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from true_lora.text import HashingTextEncoder


@dataclass(frozen=True)
class LoraTensorSpec:
    name: str
    out_features: int
    in_features: int
    rank: int
    alpha: float = 1.0

    @property
    def a_shape(self) -> tuple[int, int]:
        return (self.rank, self.in_features)

    @property
    def b_shape(self) -> tuple[int, int]:
        return (self.out_features, self.rank)


@dataclass
class AdapterSpec:
    description: str
    embedding: torch.Tensor
    tensors: dict[str, torch.Tensor]
    metrics: dict[str, float] | None = None
    source: str | None = None
    fingerprint: str | None = None


class AdapterBank:
    def __init__(self, adapters: Iterable[AdapterSpec]) -> None:
        self.adapters = list(adapters)
        if not self.adapters:
            raise ValueError("AdapterBank requires at least one adapter")

        # Pre-normalize embeddings once at construction time
        embeddings = [F.normalize(a.embedding.float(), dim=0) for a in self.adapters]
        self.embeddings = torch.stack(embeddings)
        self._metric_prior_cache: dict[str, torch.Tensor] = {}

    def score(
        self,
        query_embedding: torch.Tensor,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> torch.Tensor:
        if metric_weight < 0:
            raise ValueError("metric_weight must be non-negative")
        # Embeddings are already normalized in __init__, only normalize query
        query = F.normalize(query_embedding.float(), dim=0)
        scores = self.embeddings @ query
        if metric and metric_weight > 0:
            scores = scores + self._get_metric_prior(metric) * metric_weight
        return scores

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[list[AdapterSpec], torch.Tensor]:
        if k <= 0:
            raise ValueError("k must be positive")

        scores = self.score(query_embedding, metric=metric, metric_weight=metric_weight)
        count = min(k, len(self.adapters))
        values, indices = torch.topk(scores, count)
        weights = torch.softmax(values, dim=0)
        return [self.adapters[int(i)] for i in indices], weights

    def retrieve_with_max_score(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[list[AdapterSpec], torch.Tensor, torch.Tensor]:
        """Retrieve top-k adapters and return all scores (avoids double computation)."""
        if k <= 0:
            raise ValueError("k must be positive")

        scores = self.score(query_embedding, metric=metric, metric_weight=metric_weight)
        max_score = scores.max()
        count = min(k, len(self.adapters))
        values, indices = torch.topk(scores, count)
        weights = torch.softmax(values, dim=0)
        return [self.adapters[int(i)] for i in indices], weights, scores

    def interpolate(
        self,
        query_embedding: torch.Tensor,
        k: int = 4,
        metric: str | None = None,
        metric_weight: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], float]:
        adapters, weights = self.retrieve(query_embedding, k=k, metric=metric, metric_weight=metric_weight)
        return self.interpolate_retrieved(adapters, weights)

    def interpolate_retrieved(
        self,
        adapters: list[AdapterSpec],
        weights: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], float]:
        merged: dict[str, torch.Tensor] = {}

        # Collect all tensor names and pre-stack for batch computation
        all_names: list[str] = []
        seen: set[str] = set()
        for adapter in adapters:
            for name in adapter.tensors:
                if name not in seen:
                    all_names.append(name)
                    seen.add(name)

        for name in all_names:
            # Stack tensors from all adapters that have this name, then batch multiply
            tensors = []
            valid_weights = []
            for adapter, weight in zip(adapters, weights):
                if name in adapter.tensors:
                    tensors.append(adapter.tensors[name].float())
                    valid_weights.append(weight)

            if len(tensors) == 1:
                merged[name] = tensors[0] * valid_weights[0]
            elif len(tensors) > 1:
                stacked = torch.stack(tensors)  # (k, *shape)
                w = torch.stack(valid_weights)  # (k,)
                # Reshape weights for broadcasting: (k, 1, 1, ...)
                for _ in range(stacked.dim() - 1):
                    w = w.unsqueeze(-1)
                merged[name] = (stacked * w).sum(dim=0)

        # Fast entropy computation using math for small k
        import math
        if len(weights) <= 1:
            uncertainty = 0.0
        else:
            log_weights = []
            for w in weights:
                w_val = max(float(w), 1e-8)
                log_weights.append(w_val * math.log(w_val))
            entropy = -sum(log_weights)
            max_entropy = math.log(len(weights))
            uncertainty = entropy / max_entropy
        return merged, uncertainty

    @staticmethod
    def retrieval_provenance(adapters: list[AdapterSpec], weights: torch.Tensor) -> list[dict[object]]:
        rows: list[dict[object]] = []
        for rank, (adapter, weight) in enumerate(zip(adapters, weights), start=1):
            rows.append(
                {
                    "rank": rank,
                    "description": adapter.description,
                    "weight": float(weight.detach()),
                    "source": adapter.source or "",
                    "fingerprint": adapter.fingerprint or adapter_fingerprint(adapter.tensors),
                    "metrics": adapter.metrics or {},
                }
            )
        return rows

    def _get_metric_prior(self, metric: str) -> torch.Tensor:
        """Cached metric prior computation."""
        if metric in self._metric_prior_cache:
            return self._metric_prior_cache[metric]

        values = []
        for adapter in self.adapters:
            value = 0.0
            if adapter.metrics and metric in adapter.metrics:
                value = float(adapter.metrics[metric])
            values.append(value)
        tensor = torch.tensor(values, dtype=torch.float32)
        span = tensor.max() - tensor.min()
        if float(span) <= 1e-8:
            result = torch.zeros_like(tensor)
        else:
            result = (tensor - tensor.mean()) / span
        self._metric_prior_cache[metric] = result
        return result


def infer_lora_tensor_specs(state_dict: dict[str, torch.Tensor]) -> list[LoraTensorSpec]:
    specs: list[LoraTensorSpec] = []
    for name, tensor in sorted(state_dict.items()):
        if not name.endswith(".lora_A.weight"):
            continue
        prefix = name[: -len(".lora_A.weight")]
        b_name = f"{prefix}.lora_B.weight"
        if b_name not in state_dict:
            continue

        a = state_dict[name]
        b = state_dict[b_name]
        if a.ndim != 2 or b.ndim != 2:
            continue
        rank, in_features = int(a.shape[0]), int(a.shape[1])
        out_features, b_rank = int(b.shape[0]), int(b.shape[1])
        if rank != b_rank:
            continue
        specs.append(LoraTensorSpec(prefix, out_features=out_features, in_features=in_features, rank=rank))

    if not specs:
        raise ValueError("No matching LoRA A/B tensors found")
    return specs


def filter_lora_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().float()
        for name, tensor in state_dict.items()
        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")
    }


def adapter_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous().float()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def load_torch_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not contain a state dict")
    return {str(name): tensor for name, tensor in obj.items() if isinstance(tensor, torch.Tensor)}


def load_adapter_report(path: Path) -> dict[str, float]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        report = obj.get("true_lora_report") or obj.get("report") or {}
        if isinstance(report, dict):
            return {str(name): float(value) for name, value in report.items() if isinstance(value, int | float)}
    return {}


def save_peft_adapter(path: Path, state_dict: dict[str, torch.Tensor], report: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    metadata = dict(report or {})
    metadata.setdefault("adapter_fingerprint", adapter_fingerprint(clean))
    torch.save({"state_dict": clean, "true_lora_report": metadata}, path)


def save_peft_directory(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    specs: list[LoraTensorSpec],
    report: dict[str, object] | None = None,
    base_model_name_or_path: str = "",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    clean = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    metadata = dict(report or {})
    metadata.setdefault("adapter_fingerprint", adapter_fingerprint(clean))
    torch.save({"state_dict": clean, "true_lora_report": metadata}, path / "adapter_model.bin")
    target_modules = sorted({spec.name.split(".")[-1] for spec in specs})
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name_or_path,
        "r": max(spec.rank for spec in specs),
        "lora_alpha": max(spec.alpha for spec in specs),
        "target_modules": target_modules,
        "true_lora_tensor_specs": [asdict(spec) for spec in specs],
        "true_lora_report": metadata,
        "true_lora_adapter_fingerprint": metadata["adapter_fingerprint"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def load_adapter_manifest(path: Path, encoder) -> tuple[list[LoraTensorSpec], AdapterBank, list[AdapterSpec]]:
    adapters: list[AdapterSpec] = []
    inferred_specs: list[LoraTensorSpec] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            description = row.get("description")
            adapter_path = row.get("path")
            if not description or not adapter_path:
                raise ValueError(f"{path}:{line_number} requires description and path")

            resolved = (path.parent / adapter_path).resolve()
            tensors = filter_lora_tensors(load_torch_state_dict(resolved))
            fingerprint = adapter_fingerprint(tensors)
            specs = infer_lora_tensor_specs(tensors)
            if inferred_specs is None:
                inferred_specs = specs
            elif [s.name for s in inferred_specs] != [s.name for s in specs]:
                raise ValueError(f"{resolved} has incompatible LoRA tensor names")

            adapters.append(
                AdapterSpec(
                    description=description,
                    embedding=encoder.encode(description),
                    tensors=tensors,
                    metrics=row.get("metrics"),
                    source=str(resolved),
                    fingerprint=fingerprint,
                )
            )

    if inferred_specs is None:
        raise ValueError(f"{path} did not contain any adapters")
    return inferred_specs, AdapterBank(adapters), adapters


def validate_adapter_manifest(
    path: Path,
    required_metrics: list[str] | None = None,
    duplicate_similarity_threshold: float = 0.98,
    text_dim: int = 256,
) -> dict:
    required_metrics = required_metrics or []
    rows = []
    errors = []
    warnings = []
    expected_names: list[str] | None = None
    expected_shapes: dict[str, tuple[int, ...]] | None = None
    encoder = HashingTextEncoder(dim=text_dim)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue

            description = row.get("description")
            adapter_path = row.get("path")
            metrics = row.get("metrics") or {}
            if not description:
                errors.append(f"{path}:{line_number}: missing description")
            if not adapter_path:
                errors.append(f"{path}:{line_number}: missing path")
                continue
            for metric in required_metrics:
                if metric not in metrics:
                    warnings.append(f"{path}:{line_number}: missing metric {metric!r}")

            resolved = (path.parent / adapter_path).resolve()
            try:
                tensors = filter_lora_tensors(load_torch_state_dict(resolved))
                specs = infer_lora_tensor_specs(tensors)
                fingerprint = adapter_fingerprint(tensors)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {resolved}: {exc}")
                continue

            names = sorted(tensors)
            shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
            if expected_names is None:
                expected_names = names
                expected_shapes = shapes
            else:
                if names != expected_names:
                    errors.append(f"{path}:{line_number}: LoRA tensor names differ from first adapter")
                elif expected_shapes is not None:
                    for name, shape in shapes.items():
                        if expected_shapes.get(name) != shape:
                            errors.append(f"{path}:{line_number}: tensor {name} shape {shape} differs from {expected_shapes.get(name)}")

            rows.append(
                {
                    "line": line_number,
                    "description": description,
                    "embedding": encoder.encode(str(description)) if description else None,
                    "path": str(resolved),
                    "fingerprint": fingerprint,
                    "metrics": metrics,
                    "tensor_count": len(tensors),
                    "specs": [asdict(spec) for spec in specs],
                }
            )

    duplicate_pairs = _description_duplicate_pairs(rows, duplicate_similarity_threshold)
    for pair in duplicate_pairs:
        warnings.append(
            f"{path}: descriptions on lines {pair['left_line']} and {pair['right_line']} are near-duplicates "
            f"(similarity={pair['similarity']:.4f})"
        )

    clean_rows = []
    for row in rows:
        clean = dict(row)
        clean.pop("embedding", None)
        clean_rows.append(clean)

    return {
        "ok": not errors,
        "adapter_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "duplicate_similarity_threshold": duplicate_similarity_threshold,
        "duplicate_pairs": duplicate_pairs,
        "rows": clean_rows,
    }


def _description_duplicate_pairs(rows: list[dict], threshold: float) -> list[dict[str, float | int | str]]:
    pairs: list[dict[str, float | int | str]] = []
    if threshold > 1.0:
        return pairs

    # Collect valid embeddings and metadata
    valid = [
        (row, row["embedding"].float())
        for row in rows
        if isinstance(row.get("embedding"), torch.Tensor) and row.get("description")
    ]
    if len(valid) < 2:
        return pairs

    # Batched pairwise similarity via matrix multiply
    matrix = torch.stack([emb for _, emb in valid])  # (n, dim)
    sim_matrix = matrix @ matrix.T  # (n, n)

    for i, (left, _) in enumerate(valid):
        for j in range(i + 1, len(valid)):
            right = valid[j][0]
            sim = float(sim_matrix[i, j])
            left_desc = str(left["description"])
            right_desc = str(right["description"])
            if sim >= threshold or left_desc.strip().lower() == right_desc.strip().lower():
                pairs.append(
                    {
                        "left_line": int(left["line"]),
                        "right_line": int(right["line"]),
                        "similarity": sim,
                        "left_description": left_desc,
                        "right_description": right_desc,
                    }
                )
    return pairs
