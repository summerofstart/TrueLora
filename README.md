# True-LoRA

**Reliable, Uncertainty-Aware Text-to-LoRA Generation**

Developed by [MARVserver](https://github.com/MARVserver)

True-LoRA is a framework that generates LoRA (Low-Rank Adaptation) adapters directly from text prompts, enabling on-the-fly model customization without per-task fine-tuning. It runs as pure text-to-LoRA or with optional retrieval grounding, trains by reconstruction or end-to-end SFT, and reports calibrated confidence so you know when to trust — or abstain from — a generated adapter.

## Overview

Traditional LoRA adapters require expensive fine-tuning for each task. True-LoRA
generates a LoRA from a text prompt in a single forward pass. The hypernetwork can
run **bankless** (pure text-to-LoRA, no retrieval database) or **retrieval-grounded**
(blend in the nearest adapters from a bank), and it can be trained two ways:

1. **Semantic encoding**: Embed the prompt with a multilingual sentence encoder (with an offline hashing fallback)
2. **Generation**: A `(task, layer, module)`-conditioned hypernetwork emits the LoRA directly
3. **(Optional) Retrieval grounding**: Blend in the nearest adapters from a bank, weighted by similarity
4. **Uncertainty & reliability**: Estimate confidence, calibrate it (ECE), and abstain / gate on it
5. **Training**: Reconstruction (copy example LoRAs) **or** end-to-end SFT (apply the LoRA to a frozen base model and backpropagate the downstream loss)

This enables instant adaptation of large language models to new tasks without
per-task fine-tuning.

## Architecture

```
                         Text Prompt
                              │
                  ┌───────────▼────────────┐
                  │ SemanticTextEncoder     │  (multilingual; hashing fallback)
                  └───────────┬────────────┘
              ┌───────────────┴───────────────┐
              ▼                                ▼
  ┌─────────────────────┐        ┌──────────────────────────┐
  │ ConditionedHyper-   │        │ Adapter Bank (optional)   │
  │ Adapter (task,layer,│        │  Retrieval → Interpolation│
  │ module)             │        │  (skipped when bankless)  │
  └──────────┬──────────┘        └─────────────┬────────────┘
             └───────────────┬─────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Uncertainty-weighted blend   │
              │ + norm clip (+ OOD abstain)  │
              └──────────────┬───────────────┘
                             ▼
                     ┌───────────────┐
                     │  LoRA Adapter │  (PEFT-compatible)
                     └───────────────┘

Training: reconstruction (copy example LoRAs)  │  end-to-end SFT
          via train_on_adapter_bank            │  via sft_train_hypernetwork
                                               │  (differentiable LoRA application,
                                               │   downstream loss → hypernetwork)
```

## Key Features

- **Zero-Shot Adaptation**: Generate LoRA adapters for new tasks without fine-tuning
- **Bankless or Retrieval-Grounded**: Run as pure text-to-LoRA (`adapter_bank=None`) or blend in retrieved neighbors from an adapter bank
- **Semantic Text Encoder**: Multilingual sentence embeddings (with an offline hashing fallback) so cross-lingual descriptions like `"binary search"` and `"二分探索"` land near each other
- **Conditioned Hypernetwork**: A shared trunk conditioned on `(task, layer, module)` whose parameters scale with the number of *module types*, not the number of layers — ~28× smaller than a flat generator on a 28-layer model
- **End-to-End SFT**: Train the hypernetwork through a differentiable LoRA application and a real downstream loss, not just weight reconstruction
- **Calibrated Reliability**: ECE/MCE, risk-coverage/AURC selective prediction, and an OOD abstain path — know when to trust or abstain
- **Honest Zero-Shot Generalization**: a held-out benchmark that measures the generalization gap *and* whether confidence actually predicts it (calibration linkage), backed by a training-free novelty signal that lowers confidence on unseen prompts
- **Quality Gating**: Automatically accept/reject adapters based on confidence and reliability thresholds
- **PEFT Compatible**: Export to standard HuggingFace PEFT format
- **Reproducibility**: Deterministic generation with seed control

### Semantic Encoder + Conditioned Hypernetwork

The default `HashingTextEncoder` + flat `HyperAdapter` remain the backward-compatible
baseline. To opt into the stronger Text-to-LoRA-style stack:

```python
from true_lora import SemanticTextEncoder, TrueLoraGenerator

encoder = SemanticTextEncoder()  # multilingual SBERT; falls back to hashing if offline

# Bankless: pure text-to-LoRA, no retrieval database.
model = TrueLoraGenerator(
    specs,
    adapter_bank=None,         # omit the bank entirely
    hidden_dim=512,
    encoder=encoder,           # semantic embeddings drive generation
    hyper_kind="conditioned",  # shared, (task, layer, module)-conditioned hypernetwork
)
state_dict, report = model.generate("write a fast vectorized numpy function")

# Retrieval-grounded: pass an AdapterBank built with the SAME encoder
#   AdapterSpec(desc, encoder.encode(desc), tensors, ...)
# model = TrueLoraGenerator(specs, bank, encoder=encoder, hyper_kind="conditioned")
```

The conditioned hypernetwork parses each `LoraTensorSpec` name into a `(layer index,
module type)` pair: the same module type at different layers shares an output head and
a module embedding, differing only via a per-layer embedding. This is what keeps the
parameter count flat as model depth grows.

## Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- transformers (optional, for HuggingFace model evaluation)
- sentence-transformers (optional, for the multilingual semantic encoder)

## Quick Start

### Bankless generation (Python)

The simplest path — no adapter bank, no manifest. Train the hypernetwork on a few
`(description → LoRA)` pairs, then generate from any prompt. See
[`notebooks/matutake_coding_lora_tutorial.ipynb`](notebooks/matutake_coding_lora_tutorial.ipynb)
for a full, runnable Colab playground, and
[`notebooks/cot_think_compare.ipynb`](notebooks/cot_think_compare.ipynb) for `<think>`-tag
chain-of-thought with a base-vs-adapted comparison (toggling a generated LoRA on the same model).

```python
from true_lora import TrueLoraGenerator, SemanticTextEncoder, train_on_adapter_bank

encoder = SemanticTextEncoder()
model = TrueLoraGenerator(specs, adapter_bank=None, encoder=encoder, hyper_kind="conditioned")
train_on_adapter_bank(model, adapters, steps=200)         # (description -> LoRA) pairs
state_dict, report = model.generate("write a fast vectorized numpy function")
```

### 1. Demo Command

Generate a LoRA adapter from a text prompt:

```bash
true-lora demo --prompt "python code generation debugging" --out adapter.pt
```

### 2. Train on Adapter Bank

Train the hypernetwork on a collection of LoRA adapters:

```bash
true-lora train \
  --manifest adapters.jsonl \
  --out checkpoint.pt \
  --steps 500
```

### 3. Generate Adapter

Generate a new LoRA adapter using a trained checkpoint:

```bash
true-lora generate \
  --manifest adapters.jsonl \
  --checkpoint checkpoint.pt \
  --prompt "creative writing storytelling" \
  --out generated.pt
```

### 4. Evaluate on HuggingFace Model

Evaluate the generated adapter on GPT-2 or other models:

```bash
true-lora gate \
  --adapter generated.pt \
  --hf-generation-benchmark benchmark.jsonl \
  --model gpt2
```

## Adapter Manifest Format

The manifest file (`adapters.jsonl`) lists available adapters with descriptions and metrics:

```json
{"description": "code generation python", "path": "adapters/adapter_00.pt", "metrics": {"score": 0.72}}
{"description": "japanese translation", "path": "adapters/adapter_01.pt", "metrics": {"score": 0.48}}
{"description": "creative writing", "path": "adapters/adapter_02.pt", "metrics": {"score": 0.61}}
```

## Performance Optimizations

True-LoRA includes several performance optimizations:

### 1. Cached Metric Prior
- Pre-computes and caches metric priors for adapter scoring
- Eliminates redundant tensor operations on repeated queries

### 2. Batched Pairwise Similarity
- Uses matrix multiplication for O(n²) similarity computations
- Reduces Python loop overhead by 10-100x

### 3. Single-Pass Retrieval
- Combines score computation and top-k retrieval
- Avoids double normalization and dot product calculations

### 4. Targeted Weight Cloning
- Only clones weights for modules that will be modified
- Reduces memory usage by 90%+ on large models

### 5. Token Hash Cache
- Caches blake2b hash results for repeated tokens
- Speeds up text encoding by 5-10x for common prompts

### 6. Batched Tensor Operations
- Uses `torch.stack` and matrix multiplication for adapter blending
- Eliminates per-tensor Python loop overhead

## Benchmark Results

Performance on GPT-2 (124M parameters):

| Metric | Value |
|--------|-------|
| Training Loss (200 steps) | 1.38 → 0.0008 |
| Mean Blended MSE | 0.008242 |
| Mean Retrieval MSE | 0.010435 |
| Mean Generated MSE | 0.006293 |
| Generation Time (per adapter) | ~2ms |

## Advanced Usage

### Custom Tensor Specs

Define LoRA targets for specific model architectures:

```python
from true_lora.adapter import LoraTensorSpec

specs = [
    LoraTensorSpec("transformer.h.0.attn.c_attn", out_features=2304, in_features=768, rank=4),
    LoraTensorSpec("transformer.h.0.attn.c_proj", out_features=768, in_features=768, rank=4),
]
```

### Quality Gating

Automatically accept/reject adapters based on quality criteria:

```python
from true_lora.quality import QualityGate, gate_adapter

gate = QualityGate(
    min_accuracy_delta=0.1,
    max_uncertainty=0.8,
    max_tensor_norm=4.0,
)

report = gate_adapter(state_dict, eval_report, gate=gate)
if report["accepted"]:
    # Adapter is ready for deployment
    pass
```

### Reliability: Calibration, Selective Prediction & Abstention

A text-to-LoRA hypernetwork always emits *something* — when the task description is
out of distribution it silently produces a low-quality adapter. True-LoRA's
differentiator is reporting **when it does not know**:

- **Calibration (ECE/MCE)** — is a confidence of 0.8 actually right 80% of the time?
  A `HistogramBinningCalibrator` re-maps raw confidence to empirical accuracy.
- **Selective prediction (risk-coverage / AURC)** — answer only the most confident
  fraction and measure the residual risk; a good confidence signal yields low risk
  at high coverage.
- **Abstention** — contrast the risk of answered vs abstained samples to confirm the
  OOD abstain path catches the bad ones.

```python
from true_lora.reliability import reliability_report_for_adapters

report = reliability_report_for_adapters(model, adapters, tolerance=0.02, calibrate=True)
print(report["ece"], report["calibrated_ece"])         # raw vs calibrated
print(report["aurc"], report["selective_risk"])        # selective generation
print(report["abstention"])                            # answered vs abstained risk
```

From the CLI, generate a reliability report and gate on it:

```bash
true-lora reliability --manifest adapters.jsonl --checkpoint ckpt.pt --report-out rel.json
true-lora gate --adapter generated.pt --reliability-report rel.json \
  --max-ece 0.1 --max-aurc 0.05 --max-selective-risk 0.02
```

### Zero-Shot Generalization Benchmark (with calibration linkage)

Text-to-LoRA's headline claim is *zero-shot generalization*: describe a task never
seen in training and still get a working adapter. The honest way to measure that is
a held-out split — train on one set of task descriptions, evaluate on disjoint,
unseen ones — and report the **generalization gap** between them.

True-LoRA goes one step further than just reporting a number. It measures whether
the model's own confidence **tracks** that gap:

- **calibration linkage** — on unseen tasks, does higher confidence really predict
  lower loss? (Pearson correlation of confidence vs. `-loss`; toward +1 is better.)
- **honesty gap** — does the model lower its confidence on unseen descriptions
  relative to seen ones, i.e. does it *know* they are harder?
- **selective generalization** — answer only the most confident unseen tasks and
  watch the residual risk fall.

A learned variance head alone is nearly constant and cannot do this, so the
generator carries a **training-free novelty signal**: register the seen prompts as
distribution anchors and the reported uncertainty rises with cosine distance from
the nearest anchor. Unseen-but-near prompts stay confident; genuinely out-of-distribution
prompts get low confidence — which is exactly what makes the linkage positive.

```python
from true_lora import TrueLoraGenerator, run_zero_shot_benchmark

model = TrueLoraGenerator(specs, adapter_bank=None, hyper_kind="conditioned")
report = run_zero_shot_benchmark(model, adapters, holdout_fraction=0.3, train_steps=300)

print(report["generalization_gap"])    # seen vs. unseen loss gap
print(report["calibration_linkage"])   # does confidence predict the gap? (toward +1)
print(report["honesty_gap"])           # confidence drop on unseen tasks (>0 = honest)
print(report["honest"])                # True when both signals line up
```

`run_zero_shot_benchmark` splits the tasks, trains on the seen split only, registers
the seen prompts as anchors, then scores both splits. For a pre-split, pre-trained
model use `zero_shot_benchmark(model, train_adapters, heldout_adapters)` directly,
and `model.set_distribution_anchors(seen_prompts)` to enable the novelty signal.

From the CLI:

```bash
true-lora zero-shot --manifest adapters.jsonl --holdout-fraction 0.3 \
  --train-steps 300 --report-out zeroshot.json
```

### End-to-End SFT (train through a real downstream loss)

Reconstruction (`train_on_adapter_bank`) teaches the hypernetwork to *copy* example
LoRA weights. The stronger objective is end-to-end SFT: generate a LoRA, apply it to
a frozen base model, compute the downstream loss, and backpropagate through the
hypernetwork so it learns to produce LoRAs that actually *solve* the task.

This needs a **differentiable** LoRA application (the in-place merge runs under
`no_grad`). `LoraSFTModel` attaches forward hooks that add `(x @ Aᵀ) @ Bᵀ · (α/r)`
to each target linear, keeping the generated `A`/`B` in the autograd graph.

```python
from true_lora.sft import sft_train_hypernetwork, causal_lm_loss

# examples: list of (prompt, payload); payload is whatever your loss_fn consumes.
losses = sft_train_hypernetwork(
    generator,                 # provides .encoder and .hyper (trained in place)
    base_model,                # frozen; exposes the LoRA target nn.Linear modules
    examples=[("write a haiku", batch), ...],
    loss_fn=causal_lm_loss,    # HF causal LM: returns model(**batch).loss
    steps=300, lr=1e-3,
)
```

For lower-level control, `LoraSFTModel(base_model, specs)` is a context manager:
`set_adapter(generated_state_dict)` before each forward, then read `base_model(...)`.

### Prompt Consistency Analysis

Evaluate how consistent the adapter generation is across similar prompts:

```python
from true_lora.consistency import prompt_consistency_report, load_prompt_groups

groups = load_prompt_groups("prompts.jsonl")
report = prompt_consistency_report(model, groups, retrieval_k=4)
print(f"Mean pairwise MSE: {report['mean_pairwise_mse']:.4f}")
```

### PEFT Export

Export generated adapters to HuggingFace PEFT format:

```bash
true-lora export-peft \
  --adapter generated.pt \
  --out peft_output/ \
  --base-model gpt2
```

## Project Structure

```
true-lora/
├── src/true_lora/
│   ├── __init__.py          # Package exports
│   ├── adapter.py           # Core adapter classes and functions
│   ├── apply.py             # LoRA application utilities
│   ├── bank.py              # Adapter bank summary
│   ├── benchmark.py         # Benchmarking utilities
│   ├── cli.py               # Command-line interface
│   ├── consistency.py       # Prompt consistency analysis
│   ├── generator.py         # TrueLoraGenerator + conditioned hypernetwork (core)
│   ├── hf_eval.py           # HuggingFace model evaluation
│   ├── peft_io.py           # PEFT format I/O
│   ├── quality.py           # Quality gating
│   ├── reliability.py       # Calibration (ECE), selective prediction, abstention
│   ├── reporting.py         # JSON report utilities
│   ├── repro.py             # Reproducibility (seed control)
│   ├── sensitivity.py       # Prompt sensitivity analysis
│   ├── sft.py               # End-to-end SFT (differentiable LoRA application)
│   ├── text.py              # Text encoding (hashing + semantic encoder)
│   ├── toy_eval.py          # Toy evaluation tasks
│   ├── train.py             # Reconstruction training loop and evaluation
│   └── zeroshot.py          # Zero-shot generalization benchmark + calibration linkage
├── tests/
│   ├── test_true_lora.py    # Unit tests
│   └── smoke.py             # Integration smoke tests
├── experiments/
│   └── gpt2/                # GPT-2 experiment scripts
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Citation

If you use True-LoRA in your research, please cite:

```bibtex
@software{truelora2024,
  title={True-LoRA: Retrieval-Grounded Text-to-LoRA Generation},
  author={MARVGAME},
  year={2024},
  url={https://github.com/MARVserver/TrueLora}
}
```

## License

MIT License

---

Developed by [MARVserver](https://github.com/MARVserver)
