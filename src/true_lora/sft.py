"""End-to-end SFT for the text-to-LoRA hypernetwork.

The reconstruction objective (:func:`true_lora.train.train_on_adapter_bank`)
teaches the hypernetwork to *copy* a library of target LoRA tensors. The stronger
Text-to-LoRA objective is end-to-end SFT: generate a LoRA from the prompt, apply
it to a frozen base model, compute the downstream loss, and backpropagate through
the hypernetwork. The network then learns to produce LoRAs that actually *solve*
the task rather than merely reconstruct example weights.

The key enabler is a *differentiable* LoRA application. The in-place merge in
:mod:`true_lora.apply` runs under ``no_grad`` and cannot carry gradients back to
the generated tensors, so this module instead attaches forward hooks that add
``(x @ Aᵀ) @ Bᵀ · (alpha / rank)`` to each target linear's output, keeping the
hypernetwork-produced ``A``/``B`` in the autograd graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from true_lora.adapter import LoraTensorSpec

if TYPE_CHECKING:
    from true_lora.generator import TrueLoraGenerator


class LoraSFTModel:
    """Attach differentiable LoRA forward-hooks to a frozen base model.

    Target ``nn.Linear`` modules (named by ``spec.name``) get a forward hook that
    adds the LoRA delta computed from externally supplied, gradient-carrying
    ``A``/``B`` tensors. Call :meth:`set_adapter` with a freshly generated state
    dict before each forward pass; gradients flow from the loss back into whatever
    produced those tensors (the hypernetwork).

    Usable as a context manager so the hooks are always removed::

        with LoraSFTModel(base_model, specs) as sft:
            sft.set_adapter(generated)
            loss = loss_fn(base_model, batch)
    """

    def __init__(self, model: nn.Module, specs: Sequence[LoraTensorSpec]) -> None:
        self.model = model
        self.specs = {spec.name: spec for spec in specs}
        modules = {name: module for name, module in model.named_modules() if name in self.specs}
        if not modules:
            raise KeyError(
                "none of the LoRA target names match nn.Module names in the base model; "
                f"wanted any of {sorted(self.specs)}"
            )
        self._modules = modules
        self._active: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    @property
    def matched_targets(self) -> list[str]:
        return sorted(self._modules)

    def _make_hook(self, name: str) -> Callable:
        spec = self.specs[name]
        scale = spec.alpha / spec.rank

        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            pair = self._active.get(name)
            if pair is None:
                return output
            a, b = pair  # A: (rank, in), B: (out, rank)
            x = inputs[0]
            delta = F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))  # (x @ Aᵀ) @ Bᵀ
            return output + delta * scale

        return hook

    def set_adapter(self, state_dict: dict[str, torch.Tensor]) -> None:
        active: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in self._modules:
            a = state_dict.get(f"{name}.lora_A.weight")
            b = state_dict.get(f"{name}.lora_B.weight")
            if a is not None and b is not None:
                active[name] = (a, b)
        self._active = active

    def clear(self) -> None:
        self._active = {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "LoraSFTModel":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def sft_train_hypernetwork(
    generator: "TrueLoraGenerator",
    base_model: nn.Module,
    examples: list[tuple[str, object]],
    loss_fn: Callable[[nn.Module, object], torch.Tensor],
    *,
    steps: int = 200,
    lr: float = 1e-3,
    freeze_base: bool = True,
    grad_clip: float | None = 1.0,
) -> list[float]:
    """Train the hypernetwork end-to-end on a downstream loss.

    Each step: encode a prompt, generate a LoRA with the hypernetwork, apply it to
    ``base_model`` via differentiable hooks, evaluate ``loss_fn(base_model, payload)``,
    and backpropagate into the hypernetwork only.

    Args:
        generator: provides ``.encoder`` and ``.hyper`` (the trainable hypernetwork).
        base_model: frozen base model exposing the LoRA target ``nn.Linear`` modules.
        examples: ``(prompt, payload)`` pairs; ``payload`` is whatever ``loss_fn`` needs.
        loss_fn: maps ``(base_model, payload)`` to a scalar loss to minimize.
        steps, lr: optimization budget for AdamW over the hypernetwork parameters.
        freeze_base: set ``requires_grad=False`` on the base model (recommended).
        grad_clip: optional gradient-norm clip on the hypernetwork parameters.

    Returns:
        Per-step loss values.
    """
    if not examples:
        raise ValueError("sft_train_hypernetwork requires at least one example")

    if freeze_base:
        base_model.requires_grad_(False)

    specs = list(generator.hyper.tensor_specs)
    optimizer = torch.optim.AdamW(generator.hyper.parameters(), lr=lr)
    losses: list[float] = []

    with LoraSFTModel(base_model, specs) as sft:
        for step in range(steps):
            prompt, payload = examples[step % len(examples)]
            embedding = generator.encoder.encode(prompt)
            generated, _ = generator.hyper(embedding)
            sft.set_adapter(generated)

            loss = loss_fn(base_model, payload)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(generator.hyper.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))

    return losses


def causal_lm_loss(base_model: nn.Module, batch: dict) -> torch.Tensor:
    """Convenience ``loss_fn`` for HuggingFace causal LMs.

    ``batch`` is forwarded as keyword arguments and must include ``labels`` so the
    model returns a language-modeling ``loss`` (e.g. ``{"input_ids", "attention_mask",
    "labels"}``).
    """
    output = base_model(**batch)
    return output.loss
