"""Head-to-head: single-forward Text-to-LoRA vs. test-time ensemble epistemic.

Text-to-LoRA (T2L) generates a LoRA in one forward pass and reports a learned
variance head as its confidence. The README's own claim -- borne out below -- is
that this head is nearly constant and *cannot* tell which unseen tasks it handles.

This script trains one bankless, conditioned hypernetwork on a seen split, then
scores the held-out (zero-shot) split two ways with the *same* trained weights:

    * single   -- one forward pass, learned-variance confidence (the T2L baseline)
    * ensemble -- K members from perturbed prompt embeddings, averaged, with the
                  cross-member disagreement as an epistemic confidence

We report, averaged over several seeds:

    held_MSE   -- mean reconstruction MSE on unseen tasks (adapter quality)
    linkage    -- Pearson(confidence, -loss): does confidence predict the gap? (->+1)
    risk@50%   -- selective risk answering only the most confident half (lower better)
    risk@100%  -- mean risk at full coverage (the no-abstention baseline)

The honest takeaway: the ensemble does not change adapter quality (held_MSE is
flat) -- the win is a confidence that finally ranks unseen tasks by quality, so
selective generation works. Run: ``python experiments/ensemble_epistemic.py``.
"""

from __future__ import annotations

import math
import statistics

import torch

from true_lora import (
    AdapterSpec,
    HashingTextEncoder,
    LoraTensorSpec,
    TrueLoraGenerator,
    split_adapters_by_description,
    train_on_adapter_bank,
    zero_shot_benchmark,
)


def make_gqa_specs(layers: list[int], rank: int = 4) -> list[LoraTensorSpec]:
    specs: list[LoraTensorSpec] = []
    for li in layers:
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.q_proj", 16, 16, rank))
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.v_proj", 4, 16, rank))
    return specs


def make_distinct_adapters(specs, encoder, descriptions, seed: int) -> list[AdapterSpec]:
    gen = torch.Generator().manual_seed(seed)
    adapters: list[AdapterSpec] = []
    for idx, desc in enumerate(descriptions):
        scale = 0.1 + 0.05 * idx
        tensors = {}
        for spec in specs:
            tensors[f"{spec.name}.lora_A.weight"] = torch.randn(spec.a_shape, generator=gen) * scale
            tensors[f"{spec.name}.lora_B.weight"] = torch.randn(spec.b_shape, generator=gen) * scale
        adapters.append(AdapterSpec(desc, encoder.encode(desc), tensors, metrics={"score": 0.5}))
    return adapters


def run_seed(seed: int, ensemble: int, noise: float) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    encoder = HashingTextEncoder(dim=64)
    specs = make_gqa_specs([0, 6, 12])
    descriptions = [f"task alpha {i}" for i in range(6)] + [f"domain beta {i}" for i in range(6)]
    adapters = make_distinct_adapters(specs, encoder, descriptions, seed=seed)
    train, heldout = split_adapters_by_description(adapters, holdout_fraction=0.34, seed=seed)

    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=64, hidden_dim=64,
        max_tensor_norm=8.0, encoder=encoder, hyper_kind="conditioned",
    )
    train_on_adapter_bank(model, train, steps=300, lr=1e-2)

    out = {}
    out["single"] = zero_shot_benchmark(model, train, heldout, tolerance=0.05, ensemble=1)
    out["ensemble"] = zero_shot_benchmark(
        model, train, heldout, tolerance=0.05, ensemble=ensemble, ensemble_noise=noise,
    )
    return out


def main(seeds: int = 10, ensemble: int = 9, noise: float = 0.05) -> None:
    agg: dict[str, dict[str, list[float]]] = {}
    for seed in range(seeds):
        results = run_seed(seed, ensemble=ensemble, noise=noise)
        for label, report in results.items():
            a = agg.setdefault(label, {"mse": [], "link": [], "sr50": [], "sr100": []})
            a["mse"].append(report["heldout"]["mean_loss"])
            a["sr50"].append(report["selective_generalization"]["coverage_0.5"])
            a["sr100"].append(report["selective_generalization"]["coverage_1.0"])
            link = report["calibration_linkage"]
            if not math.isnan(link):
                a["link"].append(link)

    print(f"Text-to-LoRA zero-shot, averaged over {seeds} seeds (ensemble={ensemble}, noise={noise})\n")
    header = f"{'variant':10} {'held_MSE':>10} {'linkage':>9} {'risk@50%':>10} {'risk@100%':>10}"
    print(header)
    print("-" * len(header))
    for label in ("single", "ensemble"):
        d = agg[label]
        link = statistics.mean(d["link"]) if d["link"] else float("nan")
        print(
            f"{label:10} {statistics.mean(d['mse']):10.5f} {link:9.3f} "
            f"{statistics.mean(d['sr50']):10.5f} {statistics.mean(d['sr100']):10.5f}"
        )


if __name__ == "__main__":
    main()
