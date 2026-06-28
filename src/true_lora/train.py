from __future__ import annotations

import torch
from torch import nn

from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec
from true_lora.generator import TrueLoraGenerator
from true_lora.repro import set_seed


def train_on_adapter_bank(
    model: TrueLoraGenerator,
    adapters: list[AdapterSpec],
    steps: int = 200,
    lr: float = 1e-3,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.hyper.parameters(), lr=lr)
    losses: list[float] = []

    # Pre-convert all adapter tensors to float32 to avoid repeated .float() calls
    float_targets: list[dict[str, torch.Tensor]] = []
    float_embeddings: list[torch.Tensor] = []
    for adapter in adapters:
        float_targets.append({name: t.float() for name, t in adapter.tensors.items()})
        float_embeddings.append(adapter.embedding)

    for step in range(steps):
        idx = step % len(adapters)
        predicted, _ = model.hyper(float_embeddings[idx])
        target = float_targets[idx]

        # Batch MSE loss across all tensor names
        total_loss = torch.tensor(0.0, requires_grad=True)
        for name, t in target.items():
            total_loss = total_loss + nn.functional.mse_loss(predicted[name], t)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach()))

    return losses


def reconstruction_report(
    model: TrueLoraGenerator,
    adapters: list[AdapterSpec],
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
) -> dict[str, float]:
    total_loss = 0.0
    total_uncertainty = 0.0
    count = 0

    with torch.no_grad():
        for adapter in adapters:
            predicted, report = model.generate(
                adapter.description,
                retrieval_k=retrieval_k,
                retrieval_metric=retrieval_metric,
                metric_weight=metric_weight,
                min_retrieval_score=min_retrieval_score,
            )
            loss = torch.zeros((), dtype=torch.float32)
            tensor_count = 0
            for name, target in adapter.tensors.items():
                if name not in predicted:
                    continue
                loss = loss + nn.functional.mse_loss(predicted[name], target.float())
                tensor_count += 1
            if tensor_count == 0:
                continue
            total_loss += float((loss / tensor_count).detach())
            total_uncertainty += report["uncertainty"]
            count += 1

    if count == 0:
        raise ValueError("No comparable adapter tensors found")
    return {
        "mean_reconstruction_mse": total_loss / count,
        "mean_uncertainty": total_uncertainty / count,
        "examples": float(count),
    }


def ablation_report(
    model: TrueLoraGenerator,
    adapters: list[AdapterSpec],
    retrieval_k: int = 4,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
) -> dict[str, object]:
    totals = {"blended": 0.0, "retrieval": 0.0, "generated": 0.0}
    wins = {"blended": 0, "retrieval": 0, "generated": 0}
    rows = []
    count = 0

    with torch.no_grad():
        for adapter in adapters:
            components, report = model.generate_components(
                adapter.description,
                retrieval_k=retrieval_k,
                retrieval_metric=retrieval_metric,
                metric_weight=metric_weight,
                min_retrieval_score=min_retrieval_score,
            )
            losses = {name: _adapter_mse(state, adapter) for name, state in components.items()}
            best = min(losses, key=losses.get)
            wins[best] += 1
            for name, value in losses.items():
                totals[name] += value
            rows.append(
                {
                    "description": adapter.description,
                    "source": adapter.source,
                    "best_component": best,
                    "blended_mse": losses["blended"],
                    "retrieval_mse": losses["retrieval"],
                    "generated_mse": losses["generated"],
                    "uncertainty": report["uncertainty"],
                    "generated_weight": report["generated_weight"],
                    "max_retrieval_score": report["max_retrieval_score"],
                }
            )
            count += 1

    if count == 0:
        raise ValueError("No comparable adapter tensors found")

    return {
        "examples": float(count),
        "mean_blended_mse": totals["blended"] / count,
        "mean_retrieval_mse": totals["retrieval"] / count,
        "mean_generated_mse": totals["generated"] / count,
        "blended_wins": float(wins["blended"]),
        "retrieval_wins": float(wins["retrieval"]),
        "generated_wins": float(wins["generated"]),
        "rows": rows,
    }


def leave_one_out_report(
    tensor_specs: list[LoraTensorSpec],
    adapters: list[AdapterSpec],
    text_dim: int = 256,
    hidden_dim: int = 512,
    steps: int = 100,
    lr: float = 1e-3,
    retrieval_k: int = 4,
    max_tensor_norm: float = 4.0,
    ood_shrink_factor: float = 0.25,
    retrieval_metric: str | None = None,
    metric_weight: float = 0.0,
    min_retrieval_score: float | None = None,
    seed: int | None = None,
) -> dict:
    if len(adapters) < 2:
        raise ValueError("leave-one-out evaluation requires at least two adapters")

    folds = []
    total_mse = 0.0
    total_uncertainty = 0.0

    for index, target in enumerate(adapters):
        set_seed(seed + index if seed is not None else None)
        train_adapters = [adapter for i, adapter in enumerate(adapters) if i != index]
        model = TrueLoraGenerator(
            tensor_specs,
            AdapterBank(train_adapters),
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            max_tensor_norm=max_tensor_norm,
            ood_shrink_factor=ood_shrink_factor,
        )
        if steps > 0:
            train_on_adapter_bank(model, train_adapters, steps=steps, lr=lr)

        predicted, report = model.generate(
            target.description,
            retrieval_k=retrieval_k,
            retrieval_metric=retrieval_metric,
            metric_weight=metric_weight,
            min_retrieval_score=min_retrieval_score,
        )
        mse = _adapter_mse(predicted, target)
        total_mse += mse
        total_uncertainty += report["uncertainty"]
        folds.append(
            {
                "held_out": target.description,
                "source": target.source,
                "mse": mse,
                "uncertainty": report["uncertainty"],
                "abstained": report["abstained"],
                "max_retrieval_score": report["max_retrieval_score"],
            }
        )

    return {
        "mean_leave_one_out_mse": total_mse / len(folds),
        "mean_uncertainty": total_uncertainty / len(folds),
        "folds": folds,
        "examples": float(len(folds)),
        "seed": seed,
    }


def _adapter_mse(predicted: dict[str, torch.Tensor], target: AdapterSpec) -> float:
    # Use pre-float conversion and batch MSE
    total_loss = 0.0
    tensor_count = 0
    for name, tensor in target.tensors.items():
        if name not in predicted:
            continue
        # Both tensors should already be float32 from generate()
        total_loss += float(nn.functional.mse_loss(predicted[name].detach(), tensor.detach()))
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError("No comparable adapter tensors found")
    return total_loss / tensor_count
