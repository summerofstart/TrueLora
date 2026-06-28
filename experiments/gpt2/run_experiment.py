"""
GPT-2 True-LoRA Experiment
===========================
Creates GPT-2 compatible LoRA adapters, trains the TrueLoraGenerator,
generates new adapters from text prompts, and evaluates them on GPT-2.
"""

from pathlib import Path
import json
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from true_lora.adapter import (
    AdapterBank,
    AdapterSpec,
    LoraTensorSpec,
    save_peft_adapter,
)
from true_lora.generator import TrueLoraGenerator
from true_lora.repro import set_seed
from true_lora.text import HashingTextEncoder
from true_lora.apply import temporary_lora
from true_lora.adapter import infer_lora_tensor_specs
from true_lora.train import train_on_adapter_bank, ablation_report

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
MANIFEST = HERE / "manifest.jsonl"

# ── GPT-2 small module names ──────────────────────────────────────────────
# Target attention projections in the first 2 transformer blocks
GPT2_LAYERS = ["transformer.h.0.attn.c_attn", "transformer.h.0.attn.c_proj"]
GPT2_HIDDEN = 768
GPT2_RANK = 4

SPECS = [
    LoraTensorSpec("transformer.h.0.attn.c_attn", out_features=2304, in_features=768, rank=GPT2_RANK),
    LoraTensorSpec("transformer.h.0.attn.c_proj", out_features=768, in_features=768, rank=GPT2_RANK),
]


def make_gpt2_tensors(scale_a: float, scale_b: float) -> dict[str, torch.Tensor]:
    """Create LoRA tensors matching GPT-2 dimensions."""
    tensors = {}
    for spec in SPECS:
        tensors[f"{spec.name}.lora_A.weight"] = torch.full(spec.a_shape, scale_a)
        tensors[f"{spec.name}.lora_B.weight"] = torch.full(spec.b_shape, scale_b)
    return tensors


def create_manifest(encoder: HashingTextEncoder) -> list[AdapterSpec]:
    """Create and save training adapters, return AdapterSpec list."""
    adapters_out = OUTPUTS / "training_adapters"
    adapters_out.mkdir(parents=True, exist_ok=True)

    examples = [
        # (description, scale_A, scale_B, score)
        ("code generation python javascript typescript", 0.15, 0.08, 0.72),
        ("japanese translation polite casual business", -0.10, -0.05, 0.48),
        ("creative writing storytelling narrative prose", 0.25, 0.12, 0.61),
        ("scientific explanation physics chemistry biology", 0.20, 0.10, 0.55),
        ("chat conversation question answering dialogue", -0.05, -0.02, 0.65),
    ]

    rows = []
    adapters = []
    for i, (desc, scale_a, scale_b, score) in enumerate(examples):
        tensors = make_gpt2_tensors(scale_a, scale_b)
        adapter_path = adapters_out / f"adapter_{i:02d}.pt"
        save_peft_adapter(adapter_path, tensors, {"score": score})
        rows.append({
            "description": desc,
            "path": str(adapter_path.relative_to(HERE)),
            "metrics": {"score": score},
        })
        adapters.append(AdapterSpec(
            desc,
            encoder.encode(desc),
            tensors,
            metrics={"score": score},
            source=str(adapter_path),
        ))

    # Write manifest
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[manifest] {MANIFEST} ({len(adapters)} adapters)")
    return adapters


def run_experiment() -> None:
    print("=" * 60)
    print("GPT-2 True-LoRA Experiment")
    print("=" * 60)

    set_seed(42)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Create training adapters ──────────────────────────────
    print("\n[1/5] Creating training adapters ...")
    encoder = HashingTextEncoder(dim=256)
    adapters = create_manifest(encoder)
    bank = AdapterBank(adapters)

    # ── Step 2: Create the TrueLoraGenerator ──────────────────────────
    print("[2/5] Initializing TrueLoraGenerator (GPT-2 specs) ...")
    model = TrueLoraGenerator(
        SPECS,
        bank,
        text_dim=256,
        hidden_dim=512,
        max_tensor_norm=4.0,
        ood_shrink_factor=0.25,
    )

    # ── Step 3: Train on adapter bank ─────────────────────────────────
    print("[3/5] Training on GPT-2 adapter bank (200 steps) ...")
    losses = train_on_adapter_bank(model, adapters, steps=200, lr=1e-3)
    print(f"      First loss: {losses[0]:.6f}")
    print(f"      Last loss:  {losses[-1]:.6f}")

    # ── Step 4: Generate new LoRA adapters from prompts ───────────────
    print("[4/5] Generating LoRA adapters from prompts ...")
    prompts = [
        "python code generation debugging unit tests",
        "japanese polite conversation business email",
        "creative fantasy storytelling world building",
        "scientific research paper methodology",
    ]

    for prompt in prompts:
        state_dict, report = model.generate(
            prompt,
            retrieval_k=len(adapters),
            retrieval_metric="score",
            metric_weight=0.5,
        )
        out_path = OUTPUTS / f"generated_{prompt.replace(' ', '_')[:40]}.pt"
        save_peft_adapter(out_path, state_dict, report)
        print(f"      [{prompt[:30]:30s}] uncertainty={report['uncertainty']:.3f}  "
              f"-> {out_path.name}")

    # ── Step 5: Evaluate with GPT-2 (if transformers available) ───────
    print("[5/5] Evaluating generated adapters on GPT-2 (causal LM) ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        gpt2 = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        gpt2.eval()

        # Print generated adapter tensor stats
        adapter_candidates = sorted(OUTPUTS.glob("generated_*.pt"))
        if not adapter_candidates:
            print(f"      (no generated adapters found)")
        else:
            print(f"\n      --- Generated LoRA tensor norms ---")
            for adapter_path in adapter_candidates:
                obj = torch.load(adapter_path, map_location="cpu")
                sd = obj.get("state_dict", obj)
                total_norm = sum(t.norm().item() for t in sd.values())
                prompt_label = adapter_path.stem.replace("generated_", "")[:35]
                print(f"      {prompt_label:35s}  total_norm={total_norm:.4f}")

        # Test generation with each adapter (using sampling for diversity)
        test_prompt = "def hello_world():"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        print(f"\n      --- GPT-2 generation comparison ---")
        print(f"      Prompt: {test_prompt!r}")

        # Baseline generation
        baseline_tokens = gpt2.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )
        baseline_text = tokenizer.decode(baseline_tokens[0], skip_special_tokens=True)
        print(f"      [baseline] {baseline_text[:80]!r}")

        # Apply each adapter
        if adapter_candidates:
            for adapter_path in adapter_candidates:
                obj = torch.load(adapter_path, map_location="cpu")
                sd = obj.get("state_dict", obj)
                specs = infer_lora_tensor_specs(sd)
                with temporary_lora(gpt2, sd, specs, strict=False):
                    adapted_tokens = gpt2.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    adapted_text = tokenizer.decode(adapted_tokens[0], skip_special_tokens=True)
                    label = adapter_path.stem.replace("generated_", "")[:30]
                    print(f"      [{label:30s}] {adapted_text[:80]!r}")

    except Exception as e:
        print(f"      GPT-2 evaluation skipped: {e}")

    # ── Step 6: Ablation report ───────────────────────────────────────
    print("\n[extra] Ablation report:")
    ablation = ablation_report(model, adapters, retrieval_k=len(adapters))
    print(f"        mean_blended_mse:   {ablation['mean_blended_mse']:.6f}")
    print(f"        mean_retrieval_mse: {ablation['mean_retrieval_mse']:.6f}")
    print(f"        mean_generated_mse: {ablation['mean_generated_mse']:.6f}")
    print(f"        blended_wins:       {ablation['blended_wins']:.0f}")
    print(f"        retrieval_wins:     {ablation['retrieval_wins']:.0f}")
    print(f"        generated_wins:     {ablation['generated_wins']:.0f}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"Outputs: {OUTPUTS}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
