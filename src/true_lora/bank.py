from __future__ import annotations

import itertools

import torch

from true_lora.adapter import AdapterSpec


def adapter_bank_summary(adapters: list[AdapterSpec]) -> dict[str, object]:
    if not adapters:
        raise ValueError("Cannot summarize an empty adapter bank")

    fingerprints = [adapter.fingerprint or "" for adapter in adapters]
    metric_names = sorted({name for adapter in adapters for name in (adapter.metrics or {})})

    # Batched tensor norms using torch.cat
    all_tensors = [tensor.detach().float() for adapter in adapters for tensor in adapter.tensors.values()]
    if all_tensors:
        concatenated = torch.cat([t.flatten() for t in all_tensors])
        tensor_norms = [float(t.norm()) for t in all_tensors]
    else:
        tensor_norms = []

    # Batched pairwise similarity via matrix multiply
    if len(adapters) >= 2:
        embeddings = torch.stack([a.embedding.float() for a in adapters])  # (n, dim)
        sim_matrix = embeddings @ embeddings.T  # (n, n)
        similarities = [float(sim_matrix[i, j]) for i, j in itertools.combinations(range(len(adapters)), 2)]
    else:
        similarities = []

    return {
        "adapter_count": float(len(adapters)),
        "unique_fingerprints": float(len(set(fingerprints))),
        "duplicate_fingerprints": float(len(fingerprints) - len(set(fingerprints))),
        "metric_names": metric_names,
        "metric_coverage": {
            name: sum(1 for adapter in adapters if adapter.metrics and name in adapter.metrics) / len(adapters)
            for name in metric_names
        },
        "description_similarity": _summary_stats(similarities),
        "tensor_norms": _summary_stats(tensor_norms),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }
