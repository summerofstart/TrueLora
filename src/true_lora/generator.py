from __future__ import annotations

import math
import re

import torch
from torch import nn

from true_lora.adapter import AdapterBank, LoraTensorSpec
from true_lora.text import HashingTextEncoder


_DIGIT_RUN_RE = re.compile(r"\d+")


def module_key(name: str) -> str:
    """Collapse the layer index in a module name so the same module type shares a key.

    ``model.layers.0.self_attn.q_proj`` and ``model.layers.18.self_attn.q_proj`` both
    map to ``model.layers.{}.self_attn.q_proj`` -- they share an output head and a
    module-type embedding, while differing only via their per-layer embedding.
    """
    return _DIGIT_RUN_RE.sub("{}", name)


def layer_index(name: str) -> int:
    """Extract the first integer in a module name as its layer index (0 if absent)."""
    match = _DIGIT_RUN_RE.search(name)
    return int(match.group()) if match else 0


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


class ConditionedHyperAdapter(nn.Module):
    """Text-to-LoRA hypernetwork conditioned on (task, layer, module).

    Unlike :class:`HyperAdapter`, which emits every LoRA tensor from a single dense
    output layer (parameters grow with the *total* adapter size), this module follows
    the Text-to-LoRA design: a shared trunk is conditioned on a task embedding plus
    learned (layer index, module type) embeddings, and a small per-module-type head
    decodes one block at a time. Parameter count therefore scales with the number of
    *module types*, not the number of layers, so it stays compact on deep models and
    shares statistics across layers.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        tensor_specs: list[LoraTensorSpec],
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.tensor_specs = tensor_specs

        layer_ids = sorted({layer_index(spec.name) for spec in tensor_specs})
        module_keys = sorted({module_key(spec.name) for spec in tensor_specs})
        self.layer_to_idx = {layer: i for i, layer in enumerate(layer_ids)}
        self.module_to_idx = {key: i for i, key in enumerate(module_keys)}

        self.layer_emb = nn.Embedding(len(layer_ids), cond_dim)
        self.module_emb = nn.Embedding(len(module_keys), cond_dim)
        self.trunk = nn.Sequential(
            nn.Linear(text_dim + 2 * cond_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Each module key has a single, consistent A/B shape -> one head per key.
        key_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for spec in tensor_specs:
            key = module_key(spec.name)
            shape = (spec.a_shape, spec.b_shape)
            if key in key_shapes and key_shapes[key] != shape:
                raise ValueError(
                    f"module key {key!r} maps to inconsistent shapes "
                    f"{key_shapes[key]} and {shape}"
                )
            key_shapes[key] = shape

        self.heads = nn.ModuleDict()
        for key, (a_shape, b_shape) in key_shapes.items():
            out = math.prod(a_shape) + math.prod(b_shape)
            self.heads[self._safe(key)] = nn.Linear(hidden_dim, out * 2)

        self.register_buffer(
            "_layer_idx",
            torch.tensor([self.layer_to_idx[layer_index(s.name)] for s in tensor_specs], dtype=torch.long),
        )
        self.register_buffer(
            "_module_idx",
            torch.tensor([self.module_to_idx[module_key(s.name)] for s in tensor_specs], dtype=torch.long),
        )

        # Group spec indices by their decode head. Every spec sharing a module key
        # shares both the head *and* a single A/B shape (enforced above), so each head
        # can decode all of its layers in one batched matmul instead of one-row-at-a-
        # time. On a deep model this collapses hundreds of per-spec Linear calls into a
        # handful of (#module-types) batched calls. Index buffers are registered so they
        # ride along with ``.to(device)``.
        groups: dict[str, list[int]] = {}
        for i, spec in enumerate(tensor_specs):
            groups.setdefault(self._safe(module_key(spec.name)), []).append(i)
        self._head_groups: list[tuple[str, str, tuple[int, ...], tuple[int, ...], int, int]] = []
        for gi, (safe, idxs) in enumerate(groups.items()):
            first = tensor_specs[idxs[0]]
            a_shape, b_shape = first.a_shape, first.b_shape
            buf_name = f"_group_rows_{gi}"
            self.register_buffer(buf_name, torch.tensor(idxs, dtype=torch.long))
            self._head_groups.append(
                (safe, buf_name, a_shape, b_shape, math.prod(a_shape), math.prod(b_shape))
            )
        # Spec names indexed exactly as the gathered rows, so decoded blocks map back
        # to their canonical ``{name}.lora_{A,B}.weight`` keys.
        self._group_names: list[list[str]] = [
            [tensor_specs[i].name for i in idxs] for idxs in groups.values()
        ]

    @staticmethod
    def _safe(key: str) -> str:
        # ModuleDict keys cannot contain '.'; keep them readable and collision-free.
        return key.replace(".", "__").replace("{}", "L")

    def forward(self, embedding: torch.Tensor) -> tuple[dict[str, torch.Tensor], float]:
        device = self._layer_idx.device
        emb = embedding.float().to(device)

        count = len(self.tensor_specs)
        layer_vec = self.layer_emb(self._layer_idx)          # (S, cond)
        module_vec = self.module_emb(self._module_idx)       # (S, cond)
        task_vec = emb.unsqueeze(0).expand(count, -1)        # (S, text_dim)
        conditioned = torch.cat([task_vec, layer_vec, module_vec], dim=-1)
        latent = self.trunk(conditioned)                     # (S, hidden)

        # Decode one head at a time, but batched over all layers that share it: a
        # single (rows, hidden) @ headᵀ matmul replaces the per-spec Python loop while
        # producing numerically identical blocks.
        tensors: dict[str, torch.Tensor] = {}
        log_var_means: list[torch.Tensor] = []
        for gi, (safe, buf_name, a_shape, b_shape, a_numel, b_numel) in enumerate(self._head_groups):
            rows = getattr(self, buf_name)                   # (G,) long indices
            raw = self.heads[safe](latent.index_select(0, rows))  # (G, out*2)
            mean, log_var = raw.chunk(2, dim=-1)
            a_blocks = mean[:, :a_numel].reshape(-1, *a_shape)
            b_blocks = mean[:, a_numel : a_numel + b_numel].reshape(-1, *b_shape)
            for j, name in enumerate(self._group_names[gi]):
                tensors[f"{name}.lora_A.weight"] = a_blocks[j]
                tensors[f"{name}.lora_B.weight"] = b_blocks[j]
            log_var_means.append(log_var.mean(dim=-1))       # (G,) one scalar per spec

        # Unweighted mean of every spec's log-variance (matches the per-spec form).
        uncertainty = float(torch.sigmoid(torch.cat(log_var_means).mean()).detach())
        return tensors, uncertainty


class TrueLoraGenerator:
    def __init__(
        self,
        tensor_specs: list[LoraTensorSpec],
        adapter_bank: AdapterBank | None = None,
        text_dim: int = 256,
        hidden_dim: int = 512,
        max_tensor_norm: float = 1.0,
        ood_shrink_factor: float = 0.25,
        encoder=None,
        hyper_kind: str = "flat",
        cond_dim: int = 64,
        novelty_weight: float = 1.0,
        epistemic_scale: float = 50.0,
    ) -> None:
        # An explicit encoder (e.g. SemanticTextEncoder) overrides the hashing default
        # and dictates the hypernetwork input width via its reported ``dim``.
        self.encoder = encoder if encoder is not None else HashingTextEncoder(dim=text_dim)
        resolved_text_dim = int(getattr(self.encoder, "dim", text_dim))
        if hyper_kind == "conditioned":
            self.hyper: nn.Module = ConditionedHyperAdapter(
                resolved_text_dim, hidden_dim, tensor_specs, cond_dim=cond_dim
            )
        elif hyper_kind == "flat":
            self.hyper = HyperAdapter(resolved_text_dim, hidden_dim, tensor_specs)
        else:
            raise ValueError(f"unknown hyper_kind {hyper_kind!r}; use 'flat' or 'conditioned'")
        self.hyper_kind = hyper_kind
        self.adapter_bank = adapter_bank
        self.max_tensor_norm = max_tensor_norm
        self.ood_shrink_factor = ood_shrink_factor
        # Optional distribution anchors (the seen training prompts). When set, a
        # training-free novelty signal -- distance from the nearest seen prompt --
        # raises the reported uncertainty so the model knows when a prompt is OOD.
        self.novelty_weight = novelty_weight
        self.anchors: torch.Tensor | None = None
        # Maps raw ensemble disagreement (a small mean variance) to an epistemic
        # uncertainty in [0, 1). Larger -> the same disagreement reads as less certain.
        self.epistemic_scale = epistemic_scale

    def set_distribution_anchors(self, items) -> None:
        """Record the seen-prompt distribution for the novelty / OOD signal.

        ``items`` may be embedding tensors, objects with an ``.embedding`` attribute
        (e.g. :class:`~true_lora.adapter.AdapterSpec`), or raw description strings
        (encoded on the fly). Anchors are L2-normalized so novelty is a cosine
        distance. Pass an empty iterable or ``None`` to clear them.
        """
        if not items:
            self.anchors = None
            return
        vecs: list[torch.Tensor] = []
        for item in items:
            if isinstance(item, torch.Tensor):
                vec = item
            elif hasattr(item, "embedding"):
                vec = item.embedding
            else:
                vec = self.encoder.encode(str(item))
            vec = vec.float()
            vecs.append(vec / max(float(vec.norm()), 1e-8))
        self.anchors = torch.stack(vecs) if vecs else None

    def _novelty(self, embedding: torch.Tensor) -> tuple[float, float]:
        """Return ``(novelty, max_anchor_similarity)`` for a prompt embedding.

        Novelty is ``1 - max cosine similarity`` to the seen-prompt anchors, clamped
        to ``[0, 1]``. With no anchors set it is ``0`` (``nan`` similarity), so the
        reported uncertainty is unchanged -- preserving the bankless/retrieval
        behavior for callers that never register a distribution.
        """
        if self.anchors is None or self.anchors.numel() == 0:
            return 0.0, float("nan")
        vec = embedding.float()
        vec = vec / max(float(vec.norm()), 1e-8)
        sims = self.anchors @ vec
        max_sim = float(sims.max())
        novelty = max(0.0, min(1.0, 1.0 - max_sim))
        return novelty, max_sim

    def _apply_novelty(self, base_uncertainty: float, novelty: float) -> float:
        """Push uncertainty toward 1 as novelty rises (monotone, bounded)."""
        return min(1.0, base_uncertainty + (1.0 - base_uncertainty) * novelty * self.novelty_weight)

    def _apply_epistemic(self, base_uncertainty: float, epistemic: float) -> float:
        """Push uncertainty toward 1 as ensemble disagreement rises (monotone, bounded)."""
        return min(1.0, base_uncertainty + (1.0 - base_uncertainty) * epistemic)

    def _epistemic_uncertainty(self, disagreement: float) -> float:
        """Map raw ensemble disagreement (>= 0) to an epistemic uncertainty in [0, 1)."""
        if disagreement <= 0.0:
            return 0.0
        return 1.0 - 1.0 / (1.0 + self.epistemic_scale * disagreement)

    def _hyper_ensemble(
        self,
        embedding: torch.Tensor,
        ensemble: int,
        noise: float,
        seed: int,
    ) -> tuple[dict[str, torch.Tensor], float, float]:
        """Deep-ensemble-style test-time generation for the hypernetwork component.

        A single forward through a Text-to-LoRA hypernetwork yields a point estimate
        whose learned variance head is nearly constant -- it cannot say *which* unseen
        prompts it actually handles. Here we draw ``ensemble`` members by perturbing
        the prompt embedding with small Gaussian noise (each perturbed prompt is
        renormalized back onto the encoder's unit sphere), average their LoRA tensors,
        and read the **cross-member disagreement** as an epistemic uncertainty signal.

        Returns ``(mean_tensors, generator_uncertainty, disagreement)``. With
        ``ensemble <= 1`` or ``noise <= 0`` it is exactly the single forward pass and a
        zero disagreement, so the default path is byte-for-byte unchanged. The draw is
        deterministic in ``seed`` for reproducibility.
        """
        base_tensors, base_uncertainty = self.hyper(embedding)
        if ensemble is None or ensemble <= 1 or noise <= 0.0:
            return base_tensors, base_uncertainty, 0.0

        sums = {name: tensor.clone() for name, tensor in base_tensors.items()}
        sq_sums = {name: tensor * tensor for name, tensor in base_tensors.items()}
        gen = torch.Generator().manual_seed(int(seed))
        emb = embedding.float()
        norm = float(emb.norm())
        for _ in range(ensemble - 1):
            perturbed = emb + torch.randn(emb.shape, generator=gen) * noise
            if norm > 0.0:
                # Keep each member on the same unit sphere the encoder emits, so the
                # perturbation explores prompt-space rather than rescaling it.
                perturbed = perturbed / max(float(perturbed.norm()), 1e-8) * norm
            member, _ = self.hyper(perturbed)
            for name in sums:
                sums[name] = sums[name] + member[name]
                sq_sums[name] = sq_sums[name] + member[name] * member[name]

        mean = {name: total / ensemble for name, total in sums.items()}
        total_var = 0.0
        for name in sums:
            var = (sq_sums[name] / ensemble - mean[name] * mean[name]).clamp_min(0.0).mean()
            total_var += float(var.detach())
        disagreement = total_var / len(sums) if sums else 0.0
        return mean, base_uncertainty, disagreement

    def generate(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
        ensemble: int = 1,
        ensemble_noise: float = 0.05,
        ensemble_seed: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        components, report = self.generate_components(
            prompt,
            retrieval_k=retrieval_k,
            retrieval_metric=retrieval_metric,
            metric_weight=metric_weight,
            min_retrieval_score=min_retrieval_score,
            ensemble=ensemble,
            ensemble_noise=ensemble_noise,
            ensemble_seed=ensemble_seed,
        )
        return components["blended"], report

    def generate_components(
        self,
        prompt: str,
        retrieval_k: int = 4,
        retrieval_metric: str | None = None,
        metric_weight: float = 0.0,
        min_retrieval_score: float | None = None,
        ensemble: int = 1,
        ensemble_noise: float = 0.05,
        ensemble_seed: int = 0,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
        embedding = self.encoder.encode(prompt)

        # Bankless (pure text-to-LoRA) mode: no retrieval database, the hypernetwork
        # alone maps the prompt to a LoRA. Used when no adapter bank is provided.
        if self.adapter_bank is None:
            generated, generator_uncertainty, disagreement = self._hyper_ensemble(
                embedding, ensemble, ensemble_noise, ensemble_seed
            )
            epistemic = self._epistemic_uncertainty(disagreement)
            novelty, max_anchor_similarity = self._novelty(embedding)
            # Epistemic disagreement and novelty both inform the *reported* confidence;
            # neither rescales the produced adapter (the mean is the adapter).
            base_uncertainty = self._apply_epistemic(min(1.0, generator_uncertainty), epistemic)
            uncertainty = self._apply_novelty(base_uncertainty, novelty)
            generated_only = {name: self._clip_norm(delta) for name, delta in generated.items()}
            report = {
                "uncertainty": uncertainty,
                "retrieval_uncertainty": 0.0,
                "generator_uncertainty": generator_uncertainty,
                "epistemic": epistemic,
                "ensemble_size": float(max(1, ensemble)),
                "novelty": novelty,
                "max_anchor_similarity": max_anchor_similarity,
                "generated_weight": 1.0,
                "metric_weight": metric_weight,
                "max_retrieval_score": float("nan"),
                "min_retrieval_score": float(min_retrieval_score) if min_retrieval_score is not None else float("nan"),
                "abstained": 0.0,
                "shrink_factor": 1.0,
                "retrieved_adapters": [],
            }
            return {"blended": dict(generated_only), "retrieval": {}, "generated": generated_only}, report

        # Use single score computation (avoids double scoring)
        retrieved_adapters, retrieval_weights, all_scores = self.adapter_bank.retrieve_with_max_score(
            embedding,
            k=retrieval_k,
            metric=retrieval_metric,
            metric_weight=metric_weight,
        )
        max_retrieval_score = float(all_scores.max())
        retrieved, retrieval_uncertainty = self.adapter_bank.interpolate_retrieved(retrieved_adapters, retrieval_weights)
        generated, generator_uncertainty, disagreement = self._hyper_ensemble(
            embedding, ensemble, ensemble_noise, ensemble_seed
        )
        epistemic = self._epistemic_uncertainty(disagreement)

        # The novelty and epistemic signals inform the *reported* confidence
        # (calibration / abstain) but must NOT change the produced adapter -- folding
        # them into the blend weight shifts retrieval-vs-generation and can lower
        # accuracy on prompts that merely sit a little farther from the anchors. So the
        # blend uses the model-internal uncertainty only; novelty and epistemic
        # disagreement are applied to the reported uncertainty afterwards.
        novelty, max_anchor_similarity = self._novelty(embedding)
        blend_uncertainty = min(1.0, 0.5 * retrieval_uncertainty + 0.5 * generator_uncertainty)
        reported_base = self._apply_epistemic(blend_uncertainty, epistemic)
        uncertainty = self._apply_novelty(reported_base, novelty)
        generated_weight = 1.0 - blend_uncertainty
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
                merged = base * blend_uncertainty
            else:
                merged = base * blend_uncertainty + delta * generated_weight
            if base is not None:
                retrieval_only[name] = self._clip_norm(base * shrink)
            if delta is not None:
                generated_only[name] = self._clip_norm(delta * shrink)
            blended[name] = self._clip_norm(merged * shrink)

        return {"blended": blended, "retrieval": retrieval_only, "generated": generated_only}, {
            "uncertainty": uncertainty,
            "blend_uncertainty": blend_uncertainty,
            "retrieval_uncertainty": retrieval_uncertainty,
            "generator_uncertainty": generator_uncertainty,
            "epistemic": epistemic,
            "ensemble_size": float(max(1, ensemble)),
            "novelty": novelty,
            "max_anchor_similarity": max_anchor_similarity,
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
