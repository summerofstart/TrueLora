from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from true_lora.consistency import adapter_pair_mse
from true_lora.generator import TrueLoraGenerator


@dataclass(frozen=True)
class PromptContrast:
    group: str
    aligned: str
    control: str


def load_prompt_contrasts(path: Path) -> list[PromptContrast]:
    contrasts: list[PromptContrast] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            group = row.get("group", f"row-{line_number}")
            aligned = row.get("aligned")
            control = row.get("control")
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{path}:{line_number} requires group to be a non-empty string when provided")
            if not isinstance(aligned, str) or not aligned.strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty string aligned prompt")
            if not isinstance(control, str) or not control.strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty string control prompt")
            contrasts.append(PromptContrast(group=group, aligned=aligned, control=control))
    if not contrasts:
        raise ValueError(f"{path} did not contain any prompt contrasts")
    return contrasts


def prompt_sensitivity_report(
    model: TrueLoraGenerator,
    contrasts: list[PromptContrast],
    *,
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
) -> dict[str, object]:
    rows: list[dict[str, float | str]] = []
    total_mse = 0.0
    total_delta = 0.0
    total_aligned_score = 0.0
    total_control_score = 0.0

    for contrast in contrasts:
        aligned_state, aligned_report = model.generate(
            contrast.aligned,
            retrieval_k=retrieval_k,
            retrieval_metric=retrieval_metric,
            metric_weight=metric_weight,
            min_retrieval_score=min_retrieval_score,
        )
        control_state, control_report = model.generate(
            contrast.control,
            retrieval_k=retrieval_k,
            retrieval_metric=retrieval_metric,
            metric_weight=metric_weight,
            min_retrieval_score=min_retrieval_score,
        )
        control_mse = adapter_pair_mse(aligned_state, control_state)
        aligned_score = float(aligned_report["max_retrieval_score"])
        control_score = float(control_report["max_retrieval_score"])
        score_delta = aligned_score - control_score
        rows.append(
            {
                "group": contrast.group,
                "control_mse": control_mse,
                "aligned_retrieval_score": aligned_score,
                "control_retrieval_score": control_score,
                "retrieval_score_delta": score_delta,
                "aligned_uncertainty": float(aligned_report["uncertainty"]),
                "control_uncertainty": float(control_report["uncertainty"]),
            }
        )
        total_mse += control_mse
        total_delta += score_delta
        total_aligned_score += aligned_score
        total_control_score += control_score

    count = len(contrasts)
    return {
        "contrasts": rows,
        "examples": float(count),
        "mean_control_mse": total_mse / count,
        "mean_retrieval_score_delta": total_delta / count,
        "mean_aligned_retrieval_score": total_aligned_score / count,
        "mean_control_retrieval_score": total_control_score / count,
        "retrieval_k": float(retrieval_k),
        "metric_weight": float(metric_weight),
    }
