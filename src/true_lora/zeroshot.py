"""Zero-shot generalization benchmark with calibration linkage.

Text-to-LoRA's headline claim is *zero-shot generalization*: describe a task that
was never seen during training and still get a working adapter. The honest way to
measure that is a held-out split -- train the hypernetwork on one set of task
descriptions and evaluate on disjoint, unseen descriptions -- and report the
**generalization gap** between the two.

True-LoRA goes one step past T2L here. A generator that merely reports an accuracy
number cannot tell you *which* unseen tasks it actually handles. This module
measures whether the model's own confidence **tracks** the generalization gap:

* **calibration linkage** -- on held-out tasks, does higher confidence really
  predict lower loss? (Pearson correlation of confidence vs. ``-loss``.)
* **honesty gap** -- does the model lower its confidence on unseen descriptions
  relative to seen ones, i.e. does it *know* they are harder?
* **selective generalization** -- if we answer only the most confident held-out
  tasks (and abstain on the rest), does the residual risk drop? This ties the
  abstention path directly to generalization.

Everything is dependency-light (pure Python + torch) and reuses the per-adapter
scoring in :mod:`true_lora.reliability`, so it runs offline on CPU.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Sequence

from true_lora.reliability import (
    area_under_risk_coverage,
    collect_generation_records,
    expected_calibration_error,
    reliability_report,
    selective_risk_at_coverage,
)

if TYPE_CHECKING:
    from true_lora.adapter import AdapterSpec
    from true_lora.generator import TrueLoraGenerator


def split_adapters_by_description(
    adapters: Sequence["AdapterSpec"],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list["AdapterSpec"], list["AdapterSpec"]]:
    """Split adapters into ``(train, heldout)`` by description.

    Deduplicates by description first so the same task never lands in both splits
    (that would leak the held-out answer into training). At least one adapter is
    kept in each split when there are two or more distinct descriptions.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")

    # Stable de-duplication by description, then a deterministic shuffle.
    seen: dict[str, "AdapterSpec"] = {}
    for adapter in adapters:
        seen.setdefault(adapter.description, adapter)
    unique = list(seen.values())
    if len(unique) < 2:
        raise ValueError("need at least two distinct descriptions to hold one out")

    rng = _Lcg(seed)
    order = list(range(len(unique)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(i + 1)
        order[i], order[j] = order[j], order[i]

    n_holdout = max(1, min(len(unique) - 1, round(holdout_fraction * len(unique))))
    holdout_idx = set(order[:n_holdout])
    train = [unique[i] for i in range(len(unique)) if i not in holdout_idx]
    heldout = [unique[i] for i in range(len(unique)) if i in holdout_idx]
    return train, heldout


class _Lcg:
    """Tiny deterministic RNG so the split needs no torch/numpy global-state churn."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 2862933555777941757 + 3037000493) & ((1 << 63) - 1)

    def randint(self, bound: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        return self.state % bound


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; returns ``nan`` for <2 points or zero variance."""
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    if len(x) != len(y):
        raise ValueError("xs and ys must have the same length")
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _split_summary(records: list[dict[str, object]]) -> dict[str, float]:
    losses = [float(r["loss"]) for r in records]
    confidences = [float(r["confidence"]) for r in records]
    corrects = [float(r["correct"]) for r in records]
    n = len(records)
    return {
        "n": float(n),
        "mean_loss": sum(losses) / n,
        "mean_confidence": sum(confidences) / n,
        "accuracy": sum(corrects) / n,
    }


def zero_shot_benchmark(
    model: "TrueLoraGenerator",
    train_adapters: Sequence["AdapterSpec"],
    heldout_adapters: Sequence["AdapterSpec"],
    *,
    tolerance: float = 0.05,
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
    n_bins: int = 10,
    coverages: Sequence[float] = (0.5, 0.8, 1.0),
    calibrate: bool = True,
    ensemble: int = 1,
    ensemble_noise: float = 0.05,
    ensemble_seed: int = 0,
) -> dict[str, object]:
    """Score a *already-trained* generator on seen vs. unseen descriptions.

    The model must have been trained on ``train_adapters`` only;
    ``heldout_adapters`` are the unseen descriptions. Returns the generalization
    gap plus the calibration-linkage metrics that say whether confidence predicts
    that gap. Set ``ensemble > 1`` to score with test-time ensemble generation: its
    disagreement-based epistemic signal typically lifts the calibration linkage far
    above the single-forward learned-variance head.
    """
    if not train_adapters or not heldout_adapters:
        raise ValueError("need both train and heldout adapters")

    score = lambda batch: collect_generation_records(  # noqa: E731 - small local helper
        model,
        list(batch),
        tolerance=tolerance,
        retrieval_k=retrieval_k,
        retrieval_metric=retrieval_metric,
        metric_weight=metric_weight,
        min_retrieval_score=min_retrieval_score,
        ensemble=ensemble,
        ensemble_noise=ensemble_noise,
        ensemble_seed=ensemble_seed,
    )
    train_records = score(train_adapters)
    heldout_records = score(heldout_adapters)
    for r in train_records:
        r["split"] = "train"
    for r in heldout_records:
        r["split"] = "heldout"

    train_summary = _split_summary(train_records)
    heldout_summary = _split_summary(heldout_records)

    hc = [float(r["confidence"]) for r in heldout_records]
    hl = [float(r["loss"]) for r in heldout_records]

    # Calibration linkage: higher confidence should predict lower loss on unseen
    # tasks. Reported so that larger (toward +1) is better.
    calibration_linkage = pearson_correlation(hc, [-v for v in hl])
    # Honesty gap: a model that knows unseen tasks are harder lowers its confidence.
    honesty_gap = train_summary["mean_confidence"] - heldout_summary["mean_confidence"]
    generalization_gap = heldout_summary["mean_loss"] - train_summary["mean_loss"]

    # Selective generalization on the held-out split: answer the most confident
    # fraction, abstain on the rest, and watch the residual risk fall.
    selective_generalization = {
        f"coverage_{cov}": selective_risk_at_coverage(hc, hl, cov) for cov in coverages
    }

    reliability = reliability_report(
        hc,
        [float(r["correct"]) for r in heldout_records],
        losses=hl,
        n_bins=n_bins,
        coverages=coverages,
        abstained=[float(r["abstained"]) for r in heldout_records],
        calibrate=calibrate,
    )

    linkage_ok = (not math.isnan(calibration_linkage)) and calibration_linkage > 0.0
    return {
        "train": train_summary,
        "heldout": heldout_summary,
        "generalization_gap": generalization_gap,
        "calibration_linkage": calibration_linkage,
        "honesty_gap": honesty_gap,
        "heldout_aurc": area_under_risk_coverage(hc, hl),
        "selective_generalization": selective_generalization,
        # The model is "honest" about unseen tasks when confidence both drops on
        # them and still ranks the survivors by quality.
        "honest": bool(linkage_ok and honesty_gap >= 0.0),
        "reliability": reliability,
        "tolerance": float(tolerance),
        "records": train_records + heldout_records,
    }


def run_zero_shot_benchmark(
    model: "TrueLoraGenerator",
    adapters: Sequence["AdapterSpec"],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 0,
    train_steps: int = 200,
    lr: float = 1e-2,
    train_fn: Callable[..., list[float]] | None = None,
    set_anchors_from_train: bool = True,
    **benchmark_kwargs: object,
) -> dict[str, object]:
    """Split, train on the seen split only, then run the zero-shot benchmark.

    ``train_fn`` defaults to :func:`true_lora.train.train_on_adapter_bank` and is
    called as ``train_fn(model, train_adapters, steps=train_steps, lr=lr)``. The
    held-out descriptions never touch training, so the reported gap is a genuine
    zero-shot measurement. The split is echoed back under ``"split"`` for
    reproducibility.

    When ``set_anchors_from_train`` is set (and the model supports
    :meth:`~true_lora.generator.TrueLoraGenerator.set_distribution_anchors`), the
    seen prompts are registered as distribution anchors so the model can lower its
    confidence on unseen descriptions -- this is what lets the benchmark's
    calibration-linkage and honesty metrics measure something real.
    """
    train_adapters, heldout_adapters = split_adapters_by_description(
        adapters, holdout_fraction=holdout_fraction, seed=seed
    )
    if train_fn is None:
        from true_lora.train import train_on_adapter_bank

        train_fn = train_on_adapter_bank
    train_losses = train_fn(model, train_adapters, steps=train_steps, lr=lr)

    if set_anchors_from_train and hasattr(model, "set_distribution_anchors"):
        model.set_distribution_anchors(train_adapters)

    report = zero_shot_benchmark(model, train_adapters, heldout_adapters, **benchmark_kwargs)  # type: ignore[arg-type]
    report["split"] = {
        "train_descriptions": [a.description for a in train_adapters],
        "heldout_descriptions": [a.description for a in heldout_adapters],
        "holdout_fraction": float(holdout_fraction),
        "seed": int(seed),
    }
    report["train_losses"] = [float(v) for v in train_losses]
    return report
