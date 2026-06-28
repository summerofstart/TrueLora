from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_json_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


AUDIT_PROFILE_KEYS = {
    "min_accuracy_delta",
    "max_uncertainty",
    "max_consistency_mse",
    "min_prompt_sensitivity_mse",
    "min_retrieval_score_delta",
    "max_duplicate_pairs",
    "max_duplicate_fingerprints",
    "max_description_similarity",
    "min_adapter_count",
    "max_bank_tensor_norm",
    "min_metric_coverage",
    "require_ablation_not_worse",
    "require_fingerprint_match",
}


def load_audit_profile(path: Path) -> dict[str, Any]:
    profile = load_json_report(path)
    if not isinstance(profile, dict):
        raise ValueError(f"{path} must contain a JSON object")
    unknown = sorted(set(profile) - AUDIT_PROFILE_KEYS)
    if unknown:
        raise ValueError(f"{path} contains unknown audit profile keys: {', '.join(unknown)}")
    coverage = profile.get("min_metric_coverage")
    if coverage is not None:
        if not isinstance(coverage, dict):
            raise ValueError("min_metric_coverage must be an object")
        profile["min_metric_coverage"] = {str(name): float(value) for name, value in coverage.items()}
    for key in ("require_ablation_not_worse", "require_fingerprint_match"):
        if key in profile:
            profile[key] = bool(profile[key])
    return profile


def compare_reports(paths: list[Path], metric: str = "accuracy_delta", descending: bool = True) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        report = load_json_report(path)
        flat = _flatten(report)
        rows.append(
            {
                "path": str(path),
                "command": report.get("command", ""),
                metric: flat.get(metric),
                "accepted": flat.get("accepted"),
                "accuracy_delta": flat.get("accuracy_delta"),
                "adapted_accuracy": flat.get("adapted_accuracy"),
                "adapted_exact_match": flat.get("adapted_exact_match"),
                "uncertainty": flat.get("uncertainty"),
                "abstained": flat.get("abstained"),
                "max_tensor_norm": flat.get("max_tensor_norm"),
                "mean_pairwise_mse": flat.get("mean_pairwise_mse"),
                "mean_control_mse": flat.get("mean_control_mse"),
                "mean_retrieval_score_delta": flat.get("mean_retrieval_score_delta"),
            }
        )

    def key(row: dict[str, Any]) -> tuple[bool, float]:
        value = row.get(metric)
        if isinstance(value, int | float):
            return False, float(value)
        return True, float("-inf")

    return sorted(rows, key=key, reverse=descending)


def audit_reports(
    paths: list[Path],
    *,
    min_accuracy_delta: float | None = None,
    max_uncertainty: float | None = None,
    max_consistency_mse: float | None = None,
    min_prompt_sensitivity_mse: float | None = None,
    min_retrieval_score_delta: float | None = None,
    max_duplicate_pairs: int | None = None,
    max_duplicate_fingerprints: float | None = None,
    max_description_similarity: float | None = None,
    min_adapter_count: float | None = None,
    max_bank_tensor_norm: float | None = None,
    min_metric_coverage: dict[str, float] | None = None,
    require_ablation_not_worse: bool = False,
    require_fingerprint_match: bool = False,
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    reports = []
    payloads = []
    for path in paths:
        payload = load_json_report(path)
        payloads.append(payload)
        reports.append({"path": str(path), "command": payload.get("command", "")})
        flat.update(_flatten(payload))

    failures: list[str] = []
    _check_min(flat, failures, "accuracy_delta", min_accuracy_delta)
    _check_max(flat, failures, "uncertainty", max_uncertainty)
    _check_max(flat, failures, "mean_pairwise_mse", max_consistency_mse)
    _check_min(flat, failures, "mean_control_mse", min_prompt_sensitivity_mse)
    _check_min(flat, failures, "mean_retrieval_score_delta", min_retrieval_score_delta)
    _check_max(flat, failures, "duplicate_fingerprints", max_duplicate_fingerprints)
    _check_min(flat, failures, "adapter_count", min_adapter_count)
    description_similarity_max = _first_nested_number(payloads, ["report", "description_similarity", "max"])
    bank_tensor_norm_max = _first_nested_number(payloads, ["report", "tensor_norms", "max"])
    if max_description_similarity is not None:
        if description_similarity_max is None or description_similarity_max > max_description_similarity:
            failures.append("description_similarity")
    if max_bank_tensor_norm is not None:
        if bank_tensor_norm_max is None or bank_tensor_norm_max > max_bank_tensor_norm:
            failures.append("bank_tensor_norm")
    coverage_failures = _metric_coverage_failures(payloads, min_metric_coverage or {})
    failures.extend(coverage_failures)

    duplicate_pairs = flat.get("duplicate_pairs")
    if max_duplicate_pairs is not None:
        count = len(duplicate_pairs) if isinstance(duplicate_pairs, list) else 0
        if count > max_duplicate_pairs:
            failures.append("duplicate_pairs")

    if require_ablation_not_worse:
        blended = flat.get("mean_blended_mse")
        retrieval = flat.get("mean_retrieval_mse")
        if isinstance(blended, int | float) and isinstance(retrieval, int | float):
            if float(blended) > float(retrieval):
                failures.append("ablation_blended_worse_than_retrieval")
        else:
            failures.append("missing_ablation")

    fingerprint_mismatches = _fingerprint_mismatches(payloads) if require_fingerprint_match else []
    if fingerprint_mismatches:
        failures.append("fingerprint_mismatch")

    return {
        "accepted": not failures,
        "failures": failures,
        "reports": reports,
        "fingerprint_mismatches": fingerprint_mismatches,
        "metrics": {
            "accuracy_delta": flat.get("accuracy_delta"),
            "uncertainty": flat.get("uncertainty"),
            "mean_pairwise_mse": flat.get("mean_pairwise_mse"),
            "mean_control_mse": flat.get("mean_control_mse"),
            "mean_retrieval_score_delta": flat.get("mean_retrieval_score_delta"),
            "mean_blended_mse": flat.get("mean_blended_mse"),
            "mean_retrieval_mse": flat.get("mean_retrieval_mse"),
            "duplicate_pairs": len(duplicate_pairs) if isinstance(duplicate_pairs, list) else 0,
            "adapter_count": flat.get("adapter_count"),
            "duplicate_fingerprints": flat.get("duplicate_fingerprints"),
            "max_description_similarity": description_similarity_max,
            "max_bank_tensor_norm": bank_tensor_norm_max,
            "metric_coverage_failures": coverage_failures,
        },
    }


def _first_nested_number(payloads: list[dict[str, Any]], path: list[str]) -> float | None:
    for payload in payloads:
        value = _nested_get(payload, path)
        if isinstance(value, int | float):
            return float(value)
    return None


def parse_metric_coverage_thresholds(items: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"metric coverage threshold must be NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError(f"metric coverage threshold requires a metric name: {item!r}")
        thresholds[name] = float(value)
    return thresholds


def _metric_coverage_failures(payloads: list[dict[str, Any]], thresholds: dict[str, float]) -> list[str]:
    if not thresholds:
        return []
    coverage: dict[str, Any] = {}
    for payload in payloads:
        item = _nested_get(payload, ["report", "metric_coverage"])
        if isinstance(item, dict):
            coverage.update(item)

    failures = []
    for name, threshold in thresholds.items():
        value = coverage.get(name)
        if not isinstance(value, int | float) or float(value) < threshold:
            failures.append(f"metric_coverage:{name}")
    return failures


def _fingerprint_mismatches(payloads: list[dict[str, Any]]) -> list[dict[str, str]]:
    manifest_fingerprints: dict[str, str] = {}
    missing_sources = 0
    for payload in payloads:
        rows = _nested_get(payload, ["report", "rows"])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("path")
            fingerprint = row.get("fingerprint")
            if isinstance(source, str) and isinstance(fingerprint, str):
                manifest_fingerprints[source] = fingerprint

    mismatches: list[dict[str, str]] = []
    for payload in payloads:
        retrieved = _nested_get(payload, ["generation", "retrieved_adapters"])
        if not isinstance(retrieved, list):
            continue
        for adapter in retrieved:
            if not isinstance(adapter, dict):
                continue
            source = adapter.get("source")
            fingerprint = adapter.get("fingerprint")
            if not isinstance(source, str) or not isinstance(fingerprint, str):
                continue
            expected = manifest_fingerprints.get(source)
            if expected is None:
                missing_sources += 1
            elif expected != fingerprint:
                mismatches.append({"source": source, "expected": expected, "actual": fingerprint})

    if not manifest_fingerprints and any(_nested_get(payload, ["generation", "retrieved_adapters"]) for payload in payloads):
        mismatches.append({"source": "", "expected": "manifest_fingerprints", "actual": "missing"})
    if missing_sources:
        mismatches.append({"source": "", "expected": "all_retrieved_sources_in_manifest", "actual": str(missing_sources)})
    return mismatches


def _nested_get(value: dict[str, Any], path: list[str]) -> Any:
    item: Any = value
    for key in path:
        if not isinstance(item, dict):
            return None
        item = item.get(key)
    return item


def _check_min(flat: dict[str, Any], failures: list[str], key: str, threshold: float | None) -> None:
    if threshold is None:
        return
    value = flat.get(key)
    if not isinstance(value, int | float) or float(value) < threshold:
        failures.append(key)


def _check_max(flat: dict[str, Any], failures: list[str], key: str, threshold: float | None) -> None:
    if threshold is None:
        return
    value = flat.get(key)
    if not isinstance(value, int | float) or float(value) > threshold:
        failures.append(key)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str) or value is None:
        return value
    return str(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(_flatten(item, name))
        else:
            flat[name] = item
            flat[str(key)] = item
    return flat
