from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QualityGate:
    min_accuracy_delta: float = 0.0
    max_uncertainty: float = 0.8
    max_tensor_norm: float = 8.0
    max_consistency_mse: float | None = None
    min_prompt_sensitivity_mse: float | None = None
    min_retrieval_score_delta: float | None = None
    max_ece: float | None = None
    max_aurc: float | None = None
    max_selective_risk: float | None = None
    selective_risk_coverage: str = "coverage_0.8"


def tensor_norm_report(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    if not state_dict:
        raise ValueError("Cannot score an empty adapter")

    # Batched norm computation
    tensors = [tensor.detach().float() for tensor in state_dict.values()]
    norms = [float(t.norm()) for t in tensors]

    return {
        "max_tensor_norm": max(norms),
        "mean_tensor_norm": sum(norms) / len(norms),
        "tensor_count": float(len(norms)),
    }


def gate_adapter(
    state_dict: dict[str, torch.Tensor],
    eval_report: dict[str, float],
    generation_report: dict[str, float] | None = None,
    consistency_report: dict[str, float] | None = None,
    sensitivity_report: dict[str, float] | None = None,
    reliability_report: dict[str, object] | None = None,
    gate: QualityGate | None = None,
) -> dict[str, float | bool | str]:
    gate = gate or QualityGate()
    norms = tensor_norm_report(state_dict)
    uncertainty = float((generation_report or {}).get("uncertainty", 0.0))
    accuracy_delta = float(eval_report.get("adapted_accuracy", 0.0) - eval_report.get("baseline_accuracy", 0.0))
    consistency_mse = float((consistency_report or {}).get("mean_pairwise_mse", 0.0))
    sensitivity_mse = float((sensitivity_report or {}).get("mean_control_mse", 0.0))
    retrieval_score_delta = float((sensitivity_report or {}).get("mean_retrieval_score_delta", 0.0))
    reliability = reliability_report or {}
    ece = float(reliability.get("ece", 0.0))
    aurc = float(reliability.get("aurc", 0.0))
    selective = reliability.get("selective_risk", {}) or {}
    selective_risk = float(selective.get(gate.selective_risk_coverage, 0.0)) if isinstance(selective, dict) else 0.0

    failures: list[str] = []
    if accuracy_delta < gate.min_accuracy_delta:
        failures.append("accuracy_delta")
    if uncertainty > gate.max_uncertainty:
        failures.append("uncertainty")
    if norms["max_tensor_norm"] > gate.max_tensor_norm:
        failures.append("tensor_norm")
    if gate.max_consistency_mse is not None and consistency_mse > gate.max_consistency_mse:
        failures.append("consistency_mse")
    if gate.min_prompt_sensitivity_mse is not None and sensitivity_mse < gate.min_prompt_sensitivity_mse:
        failures.append("prompt_sensitivity_mse")
    if gate.min_retrieval_score_delta is not None and retrieval_score_delta < gate.min_retrieval_score_delta:
        failures.append("retrieval_score_delta")
    if gate.max_ece is not None and ece > gate.max_ece:
        failures.append("ece")
    if gate.max_aurc is not None and aurc > gate.max_aurc:
        failures.append("aurc")
    if gate.max_selective_risk is not None and selective_risk > gate.max_selective_risk:
        failures.append("selective_risk")

    return {
        "accepted": not failures,
        "failures": ",".join(failures),
        "accuracy_delta": accuracy_delta,
        "uncertainty": uncertainty,
        "mean_pairwise_mse": consistency_mse,
        "mean_control_mse": sensitivity_mse,
        "mean_retrieval_score_delta": retrieval_score_delta,
        "ece": ece,
        "aurc": aurc,
        "selective_risk": selective_risk,
        **norms,
    }
