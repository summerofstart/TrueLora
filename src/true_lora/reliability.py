"""Reliability metrics: calibration (ECE), selective prediction (risk-coverage),
and OOD abstention.

Text-to-LoRA hypernetworks always emit *something*; when the task description is
out of distribution they silently produce a low-quality adapter. True-LoRA's
differentiator is knowing -- and reporting -- when it does not know: a calibrated
confidence, a risk-coverage trade-off for selective generation, and an abstention
path. This module provides the pure metric functions plus a bridge that scores a
generator over an adapter bank.

All functions are dependency-light (pure Python + torch) so they run offline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from true_lora.adapter import AdapterSpec
    from true_lora.generator import TrueLoraGenerator


def _as_floats(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


def expected_calibration_error(
    confidences: Sequence[float],
    corrects: Sequence[float],
    n_bins: int = 10,
) -> dict[str, object]:
    """Expected/maximum calibration error with a reliability-diagram table.

    ``confidences`` are in ``[0, 1]`` and ``corrects`` are 0/1 outcomes. ECE is the
    sample-weighted average gap between confidence and accuracy across equal-width
    confidence bins; MCE is the worst bin gap.
    """
    conf = [min(max(c, 0.0), 1.0) for c in _as_floats(confidences)]
    corr = _as_floats(corrects)
    if len(conf) != len(corr):
        raise ValueError("confidences and corrects must have the same length")
    n = len(conf)
    if n == 0:
        raise ValueError("need at least one record")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_cnt = [0] * n_bins
    for c, y in zip(conf, corr):
        b = min(n_bins - 1, int(c * n_bins))
        bin_conf[b] += c
        bin_acc[b] += y
        bin_cnt[b] += 1

    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        if bin_cnt[b] == 0:
            bins.append({"bin": float(b), "count": 0.0, "avg_confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        avg_conf = bin_conf[b] / bin_cnt[b]
        acc = bin_acc[b] / bin_cnt[b]
        gap = abs(acc - avg_conf)
        ece += (bin_cnt[b] / n) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": float(b),
                "count": float(bin_cnt[b]),
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )
    return {"ece": ece, "mce": mce, "bins": bins, "examples": float(n)}


def risk_coverage_points(confidences: Sequence[float], losses: Sequence[float]) -> list[dict[str, float]]:
    """Risk-coverage curve: answer the most confident first, track mean risk.

    Returns one point per coverage step ``k/N`` (k most confident answered), where
    ``risk`` is the mean loss among the answered subset.
    """
    conf = _as_floats(confidences)
    loss = _as_floats(losses)
    if len(conf) != len(loss):
        raise ValueError("confidences and losses must have the same length")
    n = len(conf)
    if n == 0:
        raise ValueError("need at least one record")

    order = sorted(range(n), key=lambda i: conf[i], reverse=True)
    points: list[dict[str, float]] = []
    cumulative = 0.0
    for k, i in enumerate(order, start=1):
        cumulative += loss[i]
        points.append({"coverage": k / n, "risk": cumulative / k, "confidence_threshold": conf[i]})
    return points


def area_under_risk_coverage(confidences: Sequence[float], losses: Sequence[float]) -> float:
    """Discrete AURC: mean selective risk across all coverage levels (lower is better)."""
    points = risk_coverage_points(confidences, losses)
    return sum(p["risk"] for p in points) / len(points)


def selective_risk_at_coverage(confidences: Sequence[float], losses: Sequence[float], coverage: float) -> float:
    """Mean loss among the top ``coverage`` fraction ranked by confidence."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    conf = _as_floats(confidences)
    loss = _as_floats(losses)
    if len(conf) != len(loss):
        raise ValueError("confidences and losses must have the same length")
    n = len(conf)
    if n == 0:
        raise ValueError("need at least one record")
    order = sorted(range(n), key=lambda i: conf[i], reverse=True)
    k = max(1, math.ceil(coverage * n))
    chosen = order[:k]
    return sum(loss[i] for i in chosen) / k


class HistogramBinningCalibrator:
    """Monotone histogram-binning calibrator: maps confidence to empirical accuracy.

    Fit a confidence->accuracy table on a held-out set, then ``transform`` raw
    confidences into calibrated probabilities. Simple, robust, and sklearn-free.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.bin_accuracy: list[float] | None = None

    def fit(self, confidences: Sequence[float], corrects: Sequence[float]) -> "HistogramBinningCalibrator":
        conf = [min(max(c, 0.0), 1.0) for c in _as_floats(confidences)]
        corr = _as_floats(corrects)
        acc = [0.0] * self.n_bins
        cnt = [0] * self.n_bins
        for c, y in zip(conf, corr):
            b = min(self.n_bins - 1, int(c * self.n_bins))
            acc[b] += y
            cnt[b] += 1
        # Empty bins fall back to their midpoint confidence (identity-ish prior).
        self.bin_accuracy = [
            (acc[b] / cnt[b]) if cnt[b] > 0 else (b + 0.5) / self.n_bins for b in range(self.n_bins)
        ]
        return self

    def transform(self, confidences: Sequence[float]) -> list[float]:
        if self.bin_accuracy is None:
            raise RuntimeError("calibrator must be fit before transform")
        out: list[float] = []
        for c in _as_floats(confidences):
            cc = min(max(c, 0.0), 1.0)
            b = min(self.n_bins - 1, int(cc * self.n_bins))
            out.append(self.bin_accuracy[b])
        return out


def reliability_report(
    confidences: Sequence[float],
    corrects: Sequence[float],
    losses: Sequence[float] | None = None,
    *,
    n_bins: int = 10,
    coverages: Sequence[float] = (0.5, 0.8, 1.0),
    abstained: Sequence[float] | None = None,
    calibrate: bool = False,
) -> dict[str, object]:
    """Bundle calibration + selective-prediction metrics into one report.

    ``losses`` supplies the per-sample risk for the risk-coverage analysis; when
    omitted, the 0/1 error ``1 - correct`` is used. When ``abstained`` is given, the
    report contrasts the risk of answered vs abstained samples to validate that the
    abstention path actually catches the bad ones.
    """
    conf = _as_floats(confidences)
    corr = _as_floats(corrects)
    if not conf:
        raise ValueError("need at least one record")

    ece = expected_calibration_error(conf, corr, n_bins=n_bins)
    risk_source = _as_floats(losses) if losses is not None else [1.0 - c for c in corr]

    report: dict[str, object] = {
        "ece": ece["ece"],
        "mce": ece["mce"],
        "reliability_bins": ece["bins"],
        "examples": ece["examples"],
        "mean_accuracy": sum(corr) / len(corr),
        "mean_confidence": sum(conf) / len(conf),
        "aurc": area_under_risk_coverage(conf, risk_source),
        "risk_at_full_coverage": sum(risk_source) / len(risk_source),
        "selective_risk": {
            f"coverage_{cov}": selective_risk_at_coverage(conf, risk_source, cov) for cov in coverages
        },
        "risk_coverage_curve": risk_coverage_points(conf, risk_source),
    }

    if calibrate:
        calibrator = HistogramBinningCalibrator(n_bins=n_bins).fit(conf, corr)
        calibrated_conf = calibrator.transform(conf)
        report["calibrated_ece"] = expected_calibration_error(calibrated_conf, corr, n_bins=n_bins)["ece"]

    if abstained is not None:
        ab = _as_floats(abstained)
        answered = [risk_source[i] for i in range(len(ab)) if ab[i] < 0.5]
        skipped = [risk_source[i] for i in range(len(ab)) if ab[i] >= 0.5]
        report["abstention"] = {
            "abstain_rate": sum(ab) / len(ab),
            "answered": float(len(answered)),
            "abstained": float(len(skipped)),
            "answered_risk": (sum(answered) / len(answered)) if answered else 0.0,
            "abstained_risk": (sum(skipped) / len(skipped)) if skipped else 0.0,
        }

    return report


def collect_generation_records(
    model: "TrueLoraGenerator",
    adapters: list["AdapterSpec"],
    *,
    tolerance: float = 0.05,
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
    ensemble: int = 1,
    ensemble_noise: float = 0.05,
    ensemble_seed: int = 0,
) -> list[dict[str, object]]:
    """Score the generator over a bank, producing per-adapter reliability records.

    Confidence is ``1 - uncertainty``; risk is the reconstruction MSE between the
    generated and target adapter; an example counts as ``correct`` when its MSE is
    within ``tolerance``. Set ``ensemble > 1`` to score with test-time ensemble
    generation, whose disagreement-based epistemic signal feeds the reported
    confidence.
    """
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for adapter in adapters:
            predicted, report = model.generate(
                adapter.description,
                retrieval_k=retrieval_k,
                retrieval_metric=retrieval_metric,
                metric_weight=metric_weight,
                min_retrieval_score=min_retrieval_score,
                ensemble=ensemble,
                ensemble_noise=ensemble_noise,
                ensemble_seed=ensemble_seed,
            )
            total = 0.0
            count = 0
            for name, target in adapter.tensors.items():
                if name not in predicted:
                    continue
                total += float(F.mse_loss(predicted[name].detach(), target.float()))
                count += 1
            if count == 0:
                continue
            mse = total / count
            uncertainty = float(report["uncertainty"])
            records.append(
                {
                    "description": adapter.description,
                    "source": adapter.source,
                    "confidence": 1.0 - uncertainty,
                    "uncertainty": uncertainty,
                    "epistemic": float(report.get("epistemic", 0.0)),
                    "loss": mse,
                    "correct": 1.0 if mse <= tolerance else 0.0,
                    "abstained": float(report.get("abstained", 0.0)),
                    "max_retrieval_score": float(report.get("max_retrieval_score", 0.0)),
                }
            )
    if not records:
        raise ValueError("No comparable adapter tensors found")
    return records


def reliability_report_for_adapters(
    model: "TrueLoraGenerator",
    adapters: list["AdapterSpec"],
    *,
    tolerance: float = 0.05,
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
    n_bins: int = 10,
    coverages: Sequence[float] = (0.5, 0.8, 1.0),
    calibrate: bool = True,
) -> dict[str, object]:
    """End-to-end reliability report for a generator over an adapter bank."""
    records = collect_generation_records(
        model,
        adapters,
        tolerance=tolerance,
        retrieval_k=retrieval_k,
        retrieval_metric=retrieval_metric,
        metric_weight=metric_weight,
        min_retrieval_score=min_retrieval_score,
    )
    report = reliability_report(
        [r["confidence"] for r in records],
        [r["correct"] for r in records],
        losses=[r["loss"] for r in records],
        n_bins=n_bins,
        coverages=coverages,
        abstained=[r["abstained"] for r in records],
        calibrate=calibrate,
    )
    report["tolerance"] = float(tolerance)
    report["records"] = records
    return report
