from __future__ import annotations

import math

import torch
from torch import nn

from true_lora.adapter import AdapterBank, LoraTensorSpec
from true_lora.text import HashingTextEncoder


class HyperAdapter(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, tensor_specs: list[LoraTensorSpec]) -> None:
        super().__init__()
        self.tensor_specs = tensor_specs
        total = sum(math.prod(spec.a_shape) + math.prod(spec.b_shape) for spec in tensor_specs)
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, total * 2),
        )
        # Pre-compute offset table for fast tensor splitting
        self._offsets: list[int] = []
        self._a_shapes: list[tuple[int, ...]] = []
        self._b_shapes: list[tuple[int, ...]] = []
        offset = 0
        for spec in tensor_specs:
            self._offsets.append(offset)
            self._a_shapes.append(spec.a_shape)
            self._b_shapes.append(spec.b_shape)
            offset += math.prod(spec.a_shape) + math.prod(spec.b_shape)

    def forward(self, embedding: torch.Tensor) -> tuple[dict[str, torch.Tensor], float]:
        raw = self.net(embedding.float())
        mean, log_var = raw.chunk(2, dim=-1)
        uncertainty = float(torch.sigmoid(log_var.mean()).detach())

        # Use pre-computed offsets for faster splitting
        flat = mean
        tensors: dict[str, torch.Tensor] = {}
        for i, spec in enumerate(self.tensor_specs):
            a_size = math.prod(self._a_shapes[i])
            offset = self._offsets[i]
            a = flat[offset : offset + a_size].reshape(self._a_shapes[i])
            b = flat[offset + a_size : offset + a_size + math.prod(self._b_shapes[i])].reshape(self._b_shapes[i])
            tensors[f"{spec.name}.lora_A.weight"] = a
            tensors[f"{spec.name}.lora_B.weight"] = b

        return tensors, uncertainty


class TrueLoraGenerator:
    def __init__(
        self,
        tensor_specs: list[LoraTensorSpec],
        adapter_bank: AdapterBank,
        text_dim: int = 256,
        hidden_dim: int = 512,
        max_tensor_norm: float = 1.0,
        ood_shrink_factor: float = 0.25,
    ) -> None:
        self.encoder = HashingTextEncoder(dim=text_dim)
        self.hyper = HyperAdapter(text_dim, hidden_dim, tensor_specs)
        self.adapter_bank = adapter_bank
        self.max_tensor_norm = max_tensor_norm
        self.ood_shrink_factor = ood_shrink_factor

    def generate(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        components, report = self.generate_components(
            prompt,
            retrieval_k=retrieval_k,
            retrieval_metric=retrieval_metric,
            metric_weight=metric_weight,
            min_retrieval_score=min_retrieval_score,
        )
        return components["blended"], report

    def generate_components(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
        embedding = self.encoder.encode(prompt)
        # Use single score computation (avoids double scoring)
        retrieved_adapters, retrieval_weights, all_scores = self.adapter_bank.retrieve_with_max_score(
            embedding,
            k=retrieval_k,
            metric=retrieval_metric,
            metric_weight=metric_weight,
        )
        max_retrieval_score = float(all_scores.max())
        retrieved, retrieval_uncertainty = self.adapter_bank.interpolate_retrieved(retrieved_adapters, retrieval_weights)
        generated, generator_uncertainty = self.hyper(embedding)

        uncertainty = min(1.0, 0.5 * retrieval_uncertainty + 0.5 * generator_uncertainty)
        generated_weight = 1.0 - uncertainty
        abstained = min_retrieval_score is not None and max_retrieval_score < min_retrieval_score
        shrink = self.ood_shrink_factor if abstained else 1.0

        names = set(retrieved) | set(generated)
        blended: dict[str, torch.Tensor] = {}
        retrieval_only: dict[str, torch.Tensor] = {}
        generated_only: dict[str, torch.Tensor] = {}
        for name in names:
            base = retrieved.get(name)
            delta = generated.get(name)
            if base is None:
                merged = delta * generated_weight
            elif delta is None:
                merged = base * uncertainty
            else:
                merged = base * uncertainty + delta * generated_weight
            if base is not None:
                retrieval_only[name] = self._clip_norm(base * shrink)
            if delta is not None:
                generated_only[name] = self._clip_norm(delta * shrink)
            blended[name] = self._clip_norm(merged * shrink)

        return {"blended": blended, "retrieval": retrieval_only, "generated": generated_only}, {
            "uncertainty": uncertainty,
            "retrieval_uncertainty": retrieval_uncertainty,
            "generator_uncertainty": generator_uncertainty,
            "generated_weight": generated_weight,
            "metric_weight": metric_weight,
            "max_retrieval_score": max_retrieval_score,
            "min_retrieval_score": float(min_retrieval_score) if min_retrieval_score is not None else float("nan"),
            "abstained": float(abstained),
            "shrink_factor": shrink,
            "retrieved_adapters": self.adapter_bank.retrieval_provenance(retrieved_adapters, retrieval_weights),
        }

    def _clip_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = tensor.norm().item()  # Scalar comparison avoids full tensor ops
        if norm <= self.max_tensor_norm:
            return tensor
        return tensor * (self.max_tensor_norm / max(norm, 1e-8))


def load_true_lora_checkpoint(
    path,
    adapter_bank: AdapterBank,
    expected_specs: list[LoraTensorSpec] | None = None,
    ood_shrink_factor: float = 0.25,
) -> tuple[TrueLoraGenerator, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    specs = [
        LoraTensorSpec(
            row["name"],
            out_features=int(row["out_features"]),
            in_features=int(row["in_features"]),
            rank=int(row["rank"]),
            alpha=float(row.get("alpha", 1.0)),
        )
        for row in checkpoint["tensor_specs"]
    ]
    if expected_specs is not None and _spec_signature(specs) != _spec_signature(expected_specs):
        raise ValueError("Checkpoint tensor specs do not match manifest tensor specs")

    model = TrueLoraGenerator(
        specs,
        adapter_bank,
        text_dim=int(checkpoint.get("text_dim", 256)),
        hidden_dim=int(checkpoint.get("hidden_dim", 512)),
        max_tensor_norm=float(checkpoint.get("max_tensor_norm", 1.0)),
        ood_shrink_factor=ood_shrink_factor,
    )
    model.hyper.load_state_dict(checkpoint["hyper_state_dict"])
    return model, checkpoint


def _spec_signature(specs: list[LoraTensorSpec]) -> list[tuple]:
    return [(spec.name, spec.out_features, spec.in_features, spec.rank, float(spec.alpha)) for spec in specs]
