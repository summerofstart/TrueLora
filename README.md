# True-LoRA

**Retrieval-Grounded, Uncertainty-Aware Text-to-LoRA Generation**

Developed by [MARVGAME](https://github.com/MARVserver)

True-LoRA is a framework that generates LoRA (Low-Rank Adaptation) adapters directly from text prompts, enabling on-the-fly model customization without fine-tuning. It combines retrieval-based adapter blending with neural generation, providing uncertainty estimates for quality control.

## Overview

Traditional LoRA adapters require expensive fine-tuning for each task. True-LoRA takes a different approach:

1. **Adapter Bank**: Store pre-trained LoRA adapters with text descriptions
2. **Retrieval**: Given a new prompt, find the most relevant adapters via semantic search
3. **Blending**: Combine retrieved adapters using learned interpolation weights
4. **Generation**: Generate new adapter tensors via a hypernetwork conditioned on the prompt
5. **Uncertainty**: Estimate confidence in the generated adapter for quality gating

This enables instant adaptation of large language models to new tasks without any fine-tuning.

## Architecture

```
Text Prompt
    │
    ▼
┌─────────────────┐
│ Text Encoder    │  (Feature Hashing)
│ (HashingText)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Adapter Bank    │────▶│ Retrieval       │
│ (Pre-trained)   │     │ (Top-K Search)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ HyperAdapter    │     │ Interpolation   │
│ (Neural Net)    │     │ (Weighted Blend)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│            Blending Layer               │
│  (Uncertainty-Weighted Combination)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  LoRA Adapter │
         │  (Output)     │
         └───────────────┘
```

## Key Features

- **Zero-Shot Adaptation**: Generate LoRA adapters for new tasks without fine-tuning
- **Semantic Text Encoder**: Multilingual sentence embeddings (with an offline hashing fallback) so cross-lingual descriptions like `"binary search"` and `"二分探索"` retrieve the same adapters
- **Conditioned Hypernetwork**: A shared trunk conditioned on `(task, layer, module)` whose parameters scale with the number of *module types*, not the number of layers — ~28× smaller than a flat generator on a 28-layer model
- **Uncertainty Estimation**: Know when the generated adapter is reliable
- **Quality Gating**: Automatically accept/reject adapters based on confidence
- **PEFT Compatible**: Export to standard HuggingFace PEFT format
- **Batch Processing**: Optimized tensor operations for high throughput
- **Reproducibility**: Deterministic generation with seed control

### Semantic Encoder + Conditioned Hypernetwork

The default `HashingTextEncoder` + flat `HyperAdapter` remain the backward-compatible
baseline. To opt into the stronger Text-to-LoRA-style stack:

```python
from true_lora import SemanticTextEncoder, TrueLoraGenerator

encoder = SemanticTextEncoder()  # multilingual SBERT; falls back to hashing if offline
# Build the AdapterBank with the SAME encoder so retrieval embeddings match:
#   AdapterSpec(desc, encoder.encode(desc), tensors, ...)

model = TrueLoraGenerator(
    specs,
    bank,
    hidden_dim=512,
    encoder=encoder,           # semantic embeddings drive both retrieval and generation
    hyper_kind="conditioned",  # shared, (task, layer, module)-conditioned hypernetwork
)
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

## Quick Start

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
│   ├── generator.py         # TrueLoraGenerator (core)
│   ├── hf_eval.py           # HuggingFace model evaluation
│   ├── peft_io.py           # PEFT format I/O
│   ├── quality.py           # Quality gating
│   ├── reporting.py         # JSON report utilities
│   ├── repro.py             # Reproducibility (seed control)
│   ├── sensitivity.py       # Prompt sensitivity analysis
│   ├── text.py              # Text encoding (feature hashing)
│   ├── toy_eval.py          # Toy evaluation tasks
│   └── train.py             # Training loop and evaluation
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
