from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Mapping

import torch

from true_lora.generator import TrueLoraGenerator


PromptGroups = dict[str, list[str]]


def load_prompt_groups(path: Path) -> PromptGroups:
    groups: PromptGroups = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            group = row.get("group", "default")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty string prompt")
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{path}:{line_number} requires group to be a non-empty string when provided")
            groups.setdefault(group, []).append(prompt)
    if not groups:
        raise ValueError(f"{path} did not contain any prompts")
    return groups


def adapter_pair_mse(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    names = sorted(set(left) & set(right))
    if not names:
        raise ValueError("adapters do not share tensor names")

    # Batched MSE: concatenate all differences and compute in one pass
    diffs = []
    for name in names:
        if left[name].shape != right[name].shape:
            raise ValueError(f"adapter tensor shape mismatch for {name}: {left[name].shape} vs {right[name].shape}")
        # Use float() only if needed, skip .detach() since tensors are already detached
        l = left[name].float() if not left[name].is_floating_point() else left[name]
        r = right[name].float() if not right[name].is_floating_point() else right[name]
        diffs.append((l - r).flatten())

    if not diffs:
        return 0.0
    concatenated = torch.cat(diffs)
    return float(torch.mean(concatenated * concatenated).item())


def prompt_consistency_report(
    model: TrueLoraGenerator,
    prompt_groups: PromptGroups,
    *,
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
) -> dict[str, object]:
    group_reports: list[dict[str, float | str]] = []
    weighted_distance = 0.0
    weighted_pairs = 0
    total_prompts = 0

    for group, prompts in prompt_groups.items():
        generations = [
            model.generate(
                prompt,
                retrieval_k=retrieval_k,
                retrieval_metric=retrieval_metric,
                metric_weight=metric_weight,
                min_retrieval_score=min_retrieval_score,
            )
            for prompt in prompts
        ]
        states = [state for state, _ in generations]
        reports = [report for _, report in generations]
        distances = [adapter_pair_mse(left, right) for left, right in itertools.combinations(states, 2)]
        pair_count = len(distances)
        mean_pairwise_mse = sum(distances) / pair_count if pair_count else 0.0
        mean_uncertainty = sum(float(report["uncertainty"]) for report in reports) / len(reports)
        mean_abstained = sum(float(report.get("abstained", 0.0)) for report in reports) / len(reports)

        group_reports.append(
            {
                "group": group,
                "prompts": float(len(prompts)),
                "pairs": float(pair_count),
                "mean_pairwise_mse": mean_pairwise_mse,
                "mean_uncertainty": mean_uncertainty,
                "mean_abstained": mean_abstained,
            }
        )
        weighted_distance += mean_pairwise_mse * pair_count
        weighted_pairs += pair_count
        total_prompts += len(prompts)

    return {
        "groups": group_reports,
        "group_count": float(len(group_reports)),
        "examples": float(total_prompts),
        "pairs": float(weighted_pairs),
        "mean_pairwise_mse": weighted_distance / weighted_pairs if weighted_pairs else 0.0,
        "retrieval_k": float(retrieval_k),
        "metric_weight": float(metric_weight),
    }
