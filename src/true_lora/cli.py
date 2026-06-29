from __future__ import annotations

import argparse
from pathlib import Path

import torch

from true_lora.adapter import (
    AdapterBank,
    AdapterSpec,
    LoraTensorSpec,
    infer_lora_tensor_specs,
    load_adapter_report,
    load_torch_state_dict,
    load_adapter_manifest,
    save_peft_directory,
    save_peft_adapter,
    validate_adapter_manifest,
)
from true_lora.benchmark import evaluate_classification, load_classification_jsonl
from true_lora.bank import adapter_bank_summary
from true_lora.consistency import load_prompt_groups, prompt_consistency_report
from true_lora.generator import TrueLoraGenerator, load_true_lora_checkpoint
from true_lora.hf_eval import evaluate_hf_causal_lm_generation, evaluate_hf_sequence_classification
from true_lora.peft_io import inspect_peft_directory
from true_lora.quality import QualityGate, gate_adapter
from true_lora.reliability import reliability_report_for_adapters
from true_lora.repro import set_seed
from true_lora.zeroshot import run_zero_shot_benchmark
from true_lora.reporting import (
    audit_reports,
    compare_reports,
    load_audit_profile,
    load_json_report,
    parse_metric_coverage_thresholds,
    write_json_report,
)
from true_lora.sensitivity import load_prompt_contrasts, prompt_sensitivity_report
from true_lora.text import HashingTextEncoder
from true_lora.toy_eval import accuracy_with_adapter, adapter_for_sign_task
from true_lora.train import ablation_report, leave_one_out_report, reconstruction_report, train_on_adapter_bank


def build_demo_bank() -> tuple[list[LoraTensorSpec], AdapterBank, list[AdapterSpec]]:
    specs = [LoraTensorSpec("layers.0.attn.q_proj", out_features=16, in_features=16, rank=4)]
    encoder = HashingTextEncoder()
    adapters: list[AdapterSpec] = []

    examples = [
        ("math reasoning algebra word problems", 0.35, 0.61),
        ("japanese translation polite business writing", -0.20, 0.48),
        ("python code generation debugging tests", 0.55, 0.72),
    ]
    for description, scale, score in examples:
        tensors = {
            "layers.0.attn.q_proj.lora_A.weight": torch.full((4, 16), scale),
            "layers.0.attn.q_proj.lora_B.weight": torch.full((16, 4), scale / 2),
        }
        adapters.append(AdapterSpec(description, encoder.encode(description), tensors, metrics={"score": score}))

    return specs, AdapterBank(adapters), adapters


def demo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    specs, bank, adapters = build_demo_bank()
    model = TrueLoraGenerator(specs, bank, max_tensor_norm=args.max_norm, ood_shrink_factor=args.ood_shrink_factor)
    train_on_adapter_bank(model, adapters, steps=args.steps)
    state_dict, report = model.generate(
        args.prompt,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    report["seed"] = float(args.seed) if args.seed is not None else float("nan")
    save_peft_adapter(args.out, state_dict, report)
    maybe_write_report(args.report_out, {"command": "demo", "adapter": args.out, "seed": args.seed, "generation": report})
    print(f"saved {args.out}")
    print(report)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    encoder = HashingTextEncoder(dim=args.text_dim)
    specs, bank, adapters = load_adapter_manifest(args.manifest, encoder)
    model = TrueLoraGenerator(
        specs,
        bank,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        max_tensor_norm=args.max_norm,
        ood_shrink_factor=args.ood_shrink_factor,
    )
    losses = train_on_adapter_bank(model, adapters, steps=args.steps, lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hyper_state_dict": model.hyper.state_dict(),
            "tensor_specs": [spec.__dict__ for spec in specs],
            "text_dim": args.text_dim,
            "hidden_dim": args.hidden_dim,
            "max_tensor_norm": args.max_norm,
            "seed": args.seed,
            "losses": losses,
        },
        args.out,
    )
    train_report = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "steps": len(losses),
        "text_dim": args.text_dim,
        "hidden_dim": args.hidden_dim,
        "max_tensor_norm": args.max_norm,
        "seed": args.seed,
    }
    maybe_write_report(args.report_out, {"command": "train", "checkpoint": args.out, **train_report})
    print(f"saved {args.out}")
    print(train_report)


def generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, _ = load_adapter_manifest(args.manifest, encoder)
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    state_dict, report = model.generate(
        args.prompt,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    report["seed"] = float(args.seed) if args.seed is not None else float("nan")
    save_peft_adapter(args.out, state_dict, report)
    maybe_write_report(
        args.report_out,
        {"command": "generate", "adapter": args.out, "prompt": args.prompt, "seed": args.seed, "generation": report},
    )
    print(f"saved {args.out}")
    print(report)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, adapters = load_adapter_manifest(args.manifest, encoder)
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    report = reconstruction_report(
        model,
        adapters,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    payload = {"command": "eval", "manifest": args.manifest, "seed": args.seed, "evaluation": report}
    if args.ablation:
        payload["ablation"] = ablation_report(
            model,
            adapters,
            retrieval_k=args.k,
            retrieval_metric=args.retrieval_metric,
            metric_weight=args.metric_weight,
            min_retrieval_score=args.min_retrieval_score,
        )
    maybe_write_report(args.report_out, payload)
    print(payload)


def toy_eval(args: argparse.Namespace) -> None:
    if args.adapter:
        state_dict = load_torch_state_dict(args.adapter)
    else:
        state_dict = adapter_for_sign_task()
    report = accuracy_with_adapter(state_dict)
    maybe_write_report(args.report_out, {"command": "toy-eval", "adapter": args.adapter, "evaluation": report})
    print(report)


def gate(args: argparse.Namespace) -> None:
    state_dict = load_torch_state_dict(args.adapter)
    if args.hf_generation_benchmark:
        if not args.model:
            raise SystemExit("--model is required with --hf-generation-benchmark")
        eval_report = evaluate_hf_causal_lm_generation(
            args.model,
            state_dict,
            args.hf_generation_benchmark,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            device=args.device,
            local_files_only=not args.allow_download,
        )
    elif args.hf_benchmark:
        if not args.model:
            raise SystemExit("--model is required with --hf-benchmark")
        eval_report = evaluate_hf_sequence_classification(
            args.model,
            state_dict,
            args.hf_benchmark,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            local_files_only=not args.allow_download,
        )
    elif args.benchmark:
        benchmark = load_classification_jsonl(args.benchmark, state_dict, module_name=args.module)
        eval_report = evaluate_classification(state_dict, benchmark)
    else:
        try:
            eval_report = accuracy_with_adapter(state_dict)
        except ValueError as exc:
            eval_report = {
                "baseline_accuracy": 0.0,
                "adapted_accuracy": 0.0,
                "restored_accuracy": 0.0,
                "eval_error": str(exc),
            }
    generation_report = load_adapter_report(args.adapter)
    consistency_payload = load_json_report(args.consistency_report) if args.consistency_report else None
    consistency_report = _nested_report(consistency_payload) if consistency_payload else None
    sensitivity_payload = load_json_report(args.sensitivity_report) if args.sensitivity_report else None
    sensitivity_report = _nested_report(sensitivity_payload) if sensitivity_payload else None
    reliability_payload = load_json_report(args.reliability_report) if args.reliability_report else None
    reliability_report = _nested_report(reliability_payload) if reliability_payload else None
    report = gate_adapter(
        state_dict,
        eval_report,
        generation_report=generation_report,
        consistency_report=consistency_report,
        sensitivity_report=sensitivity_report,
        reliability_report=reliability_report,
        gate=QualityGate(
            min_accuracy_delta=args.min_accuracy_delta,
            max_uncertainty=args.max_uncertainty,
            max_tensor_norm=args.max_tensor_norm,
            max_consistency_mse=args.max_consistency_mse,
            min_prompt_sensitivity_mse=args.min_prompt_sensitivity_mse,
            min_retrieval_score_delta=args.min_retrieval_score_delta,
            max_ece=args.max_ece,
            max_aurc=args.max_aurc,
            max_selective_risk=args.max_selective_risk,
            selective_risk_coverage=args.selective_risk_coverage,
        ),
    )
    combined = {**eval_report, **generation_report, **(consistency_report or {}), **(sensitivity_report or {}), **report}
    maybe_write_report(
        args.report_out,
        {
            "command": "gate",
            "adapter": args.adapter,
            "consistency_report": args.consistency_report,
            "sensitivity_report": args.sensitivity_report,
            "report": combined,
        },
    )
    print(combined)
    if not report["accepted"]:
        raise SystemExit(2)


def export_peft(args: argparse.Namespace) -> None:
    state_dict = load_torch_state_dict(args.adapter)
    report = load_adapter_report(args.adapter)
    specs = infer_lora_tensor_specs(state_dict)
    save_peft_directory(args.out, state_dict, specs, report, base_model_name_or_path=args.base_model)
    maybe_write_report(
        args.report_out,
        {"command": "export-peft", "adapter": args.adapter, "out": args.out, "specs": [spec.__dict__ for spec in specs]},
    )
    print(f"saved {args.out}")


def inspect_peft(args: argparse.Namespace) -> None:
    report = inspect_peft_directory(args.adapter_dir)
    maybe_write_report(args.report_out, {"command": "inspect-peft", "adapter_dir": args.adapter_dir, "report": report})
    print(report)


def merge_adapter(args: argparse.Namespace) -> None:
    from true_lora.merge import merge_adapter_into_hf_model

    report = merge_adapter_into_hf_model(
        adapter_path=args.adapter,
        model_name_or_path=args.model,
        output_dir=args.out,
        dtype=args.dtype,
        device=args.device,
        allow_download=args.allow_download,
    )
    maybe_write_report(args.report_out, {"command": "merge-adapter", **report})
    print(f"Merged adapter into {report['output_dir']}")
    print(report)


def merge_adapters(args: argparse.Namespace) -> None:
    from true_lora.merge import merge_adapters

    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    report = merge_adapters(
        adapter_paths=args.adapters,
        model_name_or_path=args.model,
        output_dir=args.out,
        weights=weights,
        dtype=args.dtype,
        device=args.device,
        allow_download=args.allow_download,
    )
    maybe_write_report(args.report_out, {"command": "merge-adapters", **report})
    print(f"Merged {len(args.adapters)} adapters into {report['output_dir']}")
    print(report)


def bench(args: argparse.Namespace) -> None:
    state_dict = load_torch_state_dict(args.adapter)
    benchmark = load_classification_jsonl(args.benchmark, state_dict, module_name=args.module)
    report = evaluate_classification(state_dict, benchmark)
    maybe_write_report(args.report_out, {"command": "bench", "adapter": args.adapter, "benchmark": args.benchmark, "report": report})
    print(report)


def hf_bench(args: argparse.Namespace) -> None:
    state_dict = load_torch_state_dict(args.adapter)
    report = evaluate_hf_sequence_classification(
        args.model,
        state_dict,
        args.benchmark,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    maybe_write_report(args.report_out, {"command": "hf-bench", "adapter": args.adapter, "benchmark": args.benchmark, "model": args.model, "report": report})
    print(report)


def hf_generate_bench(args: argparse.Namespace) -> None:
    state_dict = load_torch_state_dict(args.adapter)
    report = evaluate_hf_causal_lm_generation(
        args.model,
        state_dict,
        args.benchmark,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    maybe_write_report(args.report_out, {"command": "hf-generate-bench", "adapter": args.adapter, "benchmark": args.benchmark, "model": args.model, "report": report})
    print(report)


def maybe_write_report(path: Path | None, payload: dict) -> None:
    if path is not None:
        write_json_report(path, payload)


def checkpoint_text_dim(path: Path | None, default: int) -> int:
    if path is None:
        return default
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("text_dim", default))


def _nested_report(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    report = payload.get("report")
    return report if isinstance(report, dict) else payload


def compare_report_cmd(args: argparse.Namespace) -> None:
    rows = compare_reports(args.reports, metric=args.metric, descending=not args.ascending)
    payload = {"command": "compare-reports", "metric": args.metric, "rows": rows}
    maybe_write_report(args.report_out, payload)
    for row in rows:
        print(row)


def audit_report_cmd(args: argparse.Namespace) -> None:
    options = _audit_options(args)
    report = audit_reports(args.reports, **options)
    payload = {"command": "audit", "profile": args.profile, "report": report}
    maybe_write_report(args.report_out, payload)
    print(payload)
    if not report["accepted"]:
        raise SystemExit(2)


def _audit_options(args: argparse.Namespace) -> dict:
    options = load_audit_profile(args.profile) if args.profile else {}
    cli_values = {
        "min_accuracy_delta": args.min_accuracy_delta,
        "max_uncertainty": args.max_uncertainty,
        "max_consistency_mse": args.max_consistency_mse,
        "min_prompt_sensitivity_mse": args.min_prompt_sensitivity_mse,
        "min_retrieval_score_delta": args.min_retrieval_score_delta,
        "max_duplicate_pairs": args.max_duplicate_pairs,
        "max_duplicate_fingerprints": args.max_duplicate_fingerprints,
        "max_description_similarity": args.max_description_similarity,
        "min_adapter_count": args.min_adapter_count,
        "max_bank_tensor_norm": args.max_bank_tensor_norm,
    }
    for key, value in cli_values.items():
        if value is not None:
            options[key] = value
    coverage = dict(options.get("min_metric_coverage") or {})
    coverage.update(parse_metric_coverage_thresholds(args.min_metric_coverage))
    if coverage:
        options["min_metric_coverage"] = coverage
    options["require_ablation_not_worse"] = bool(options.get("require_ablation_not_worse")) or args.require_ablation_not_worse
    options["require_fingerprint_match"] = bool(options.get("require_fingerprint_match")) or args.require_fingerprint_match
    return options


def prompt_consistency(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, _ = load_adapter_manifest(args.manifest, encoder)
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    groups = load_prompt_groups(args.prompts)
    report = prompt_consistency_report(
        model,
        groups,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    payload = {"command": "prompt-consistency", "manifest": args.manifest, "prompts": args.prompts, "seed": args.seed, "report": report}
    maybe_write_report(args.report_out, payload)
    print(payload)


def prompt_sensitivity(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, _ = load_adapter_manifest(args.manifest, encoder)
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    contrasts = load_prompt_contrasts(args.contrasts)
    report = prompt_sensitivity_report(
        model,
        contrasts,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    payload = {"command": "prompt-sensitivity", "manifest": args.manifest, "contrasts": args.contrasts, "seed": args.seed, "report": report}
    maybe_write_report(args.report_out, payload)
    print(payload)


def reliability(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, adapters = load_adapter_manifest(args.manifest, encoder)
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    report = reliability_report_for_adapters(
        model,
        adapters,
        tolerance=args.tolerance,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
        calibrate=not args.no_calibrate,
    )
    payload = {"command": "reliability", "manifest": args.manifest, "seed": args.seed, "report": report}
    maybe_write_report(args.report_out, payload)
    print(payload)


def zero_shot(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    encoder = HashingTextEncoder(dim=args.text_dim)
    specs, _bank, adapters = load_adapter_manifest(args.manifest, encoder)
    # Bankless, conditioned hypernetwork: pure text-to-LoRA is what we measure the
    # zero-shot generalization of. Anchors from the seen split are set internally.
    model = TrueLoraGenerator(
        specs,
        adapter_bank=None,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        max_tensor_norm=args.max_norm,
        encoder=encoder,
        hyper_kind="conditioned",
    )
    report = run_zero_shot_benchmark(
        model,
        adapters,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed if args.seed is not None else 0,
        train_steps=args.train_steps,
        lr=args.lr,
        tolerance=args.tolerance,
        calibrate=not args.no_calibrate,
    )
    # Drop the bulky curves/records for the printed summary; keep the headline numbers.
    headline = {
        "generalization_gap": report["generalization_gap"],
        "calibration_linkage": report["calibration_linkage"],
        "honesty_gap": report["honesty_gap"],
        "heldout_aurc": report["heldout_aurc"],
        "honest": report["honest"],
        "train_mean_loss": report["train"]["mean_loss"],
        "heldout_mean_loss": report["heldout"]["mean_loss"],
    }
    payload = {"command": "zero-shot", "manifest": args.manifest, "seed": args.seed, "report": report}
    maybe_write_report(args.report_out, payload)
    print({"command": "zero-shot", "headline": headline, "split": report["split"]})


def pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    text_dim = checkpoint_text_dim(args.checkpoint, args.text_dim)
    encoder = HashingTextEncoder(dim=text_dim)
    specs, bank, adapters = load_adapter_manifest(args.manifest, encoder)
    train_report = None
    if args.checkpoint:
        model, _ = load_true_lora_checkpoint(
            args.checkpoint,
            bank,
            expected_specs=specs,
            ood_shrink_factor=args.ood_shrink_factor,
        )
    elif args.train_steps > 0:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )
        losses = train_on_adapter_bank(model, adapters, steps=args.train_steps, lr=args.lr)
        train_report = {"first_loss": losses[0], "last_loss": losses[-1], "steps": len(losses), "seed": args.seed}
    else:
        model = TrueLoraGenerator(
            specs,
            bank,
            text_dim=args.text_dim,
            hidden_dim=args.hidden_dim,
            max_tensor_norm=args.max_norm,
            ood_shrink_factor=args.ood_shrink_factor,
        )

    state_dict, generation_report = model.generate(
        args.prompt,
        retrieval_k=args.k,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    generation_report["seed"] = float(args.seed) if args.seed is not None else float("nan")
    save_peft_adapter(args.adapter_out, state_dict, generation_report)

    benchmark = load_classification_jsonl(args.benchmark, state_dict, module_name=args.module)
    eval_report = evaluate_classification(state_dict, benchmark)
    gate_report = gate_adapter(
        state_dict,
        eval_report,
        generation_report=generation_report,
        gate=QualityGate(
            min_accuracy_delta=args.min_accuracy_delta,
            max_uncertainty=args.max_uncertainty,
            max_tensor_norm=args.max_tensor_norm,
        ),
    )
    combined_gate = {**eval_report, **generation_report, **gate_report}

    export_report = None
    if args.export_dir and gate_report["accepted"]:
        export_specs = infer_lora_tensor_specs(state_dict)
        save_peft_directory(args.export_dir, state_dict, export_specs, generation_report, base_model_name_or_path=args.base_model)
        export_report = {"out": args.export_dir, "specs": [spec.__dict__ for spec in export_specs]}

    payload = {
        "command": "pipeline",
        "adapter": args.adapter_out,
        "prompt": args.prompt,
        "seed": args.seed,
        "train": train_report,
        "generation": generation_report,
        "evaluation": eval_report,
        "gate": combined_gate,
        "export": export_report,
    }
    maybe_write_report(args.report_out, payload)
    print(payload)
    if not gate_report["accepted"]:
        raise SystemExit(2)


def loo_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    encoder = HashingTextEncoder(dim=args.text_dim)
    specs, _, adapters = load_adapter_manifest(args.manifest, encoder)
    report = leave_one_out_report(
        specs,
        adapters,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        lr=args.lr,
        retrieval_k=args.k,
        max_tensor_norm=args.max_norm,
        ood_shrink_factor=args.ood_shrink_factor,
        retrieval_metric=args.retrieval_metric,
        metric_weight=args.metric_weight,
        min_retrieval_score=args.min_retrieval_score,
    )
    maybe_write_report(args.report_out, {"command": "loo-eval", "manifest": args.manifest, "seed": args.seed, "report": report})
    print(report)


def validate_manifest_cmd(args: argparse.Namespace) -> None:
    report = validate_adapter_manifest(
        args.manifest,
        required_metrics=args.required_metric,
        duplicate_similarity_threshold=args.duplicate_similarity_threshold,
        text_dim=args.text_dim,
    )
    maybe_write_report(args.report_out, {"command": "validate-manifest", "manifest": args.manifest, "report": report})
    print(report)
    if not report["ok"]:
        raise SystemExit(2)


def bank_summary_cmd(args: argparse.Namespace) -> None:
    encoder = HashingTextEncoder(dim=args.text_dim)
    _, _, adapters = load_adapter_manifest(args.manifest, encoder)
    report = adapter_bank_summary(adapters)
    payload = {"command": "bank-summary", "manifest": args.manifest, "report": report}
    maybe_write_report(args.report_out, payload)
    print(payload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="true-lora")
    sub = parser.add_subparsers(required=True)

    demo_parser = sub.add_parser("demo")
    demo_parser.add_argument("--prompt", default="write reliable python tests for debugging")
    demo_parser.add_argument("--out", type=Path, default=Path("true-lora-demo.pt"))
    demo_parser.add_argument("--steps", type=int, default=80)
    demo_parser.add_argument("--k", type=int, default=2)
    demo_parser.add_argument("--max-norm", type=float, default=4.0)
    demo_parser.add_argument("--retrieval-metric")
    demo_parser.add_argument("--metric-weight", type=float, default=0.0)
    demo_parser.add_argument("--min-retrieval-score", type=float)
    demo_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    demo_parser.add_argument("--seed", type=int)
    demo_parser.add_argument("--report-out", type=Path)
    demo_parser.set_defaults(func=demo)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=500)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--text-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--max-norm", type=float, default=4.0)
    train_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--report-out", type=Path)
    train_parser.set_defaults(func=train)

    gen_parser = sub.add_parser("generate")
    gen_parser.add_argument("--manifest", type=Path, required=True)
    gen_parser.add_argument("--prompt", required=True)
    gen_parser.add_argument("--out", type=Path, required=True)
    gen_parser.add_argument("--checkpoint", type=Path)
    gen_parser.add_argument("--k", type=int, default=4)
    gen_parser.add_argument("--text-dim", type=int, default=256)
    gen_parser.add_argument("--hidden-dim", type=int, default=512)
    gen_parser.add_argument("--max-norm", type=float, default=4.0)
    gen_parser.add_argument("--retrieval-metric")
    gen_parser.add_argument("--metric-weight", type=float, default=0.0)
    gen_parser.add_argument("--min-retrieval-score", type=float)
    gen_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    gen_parser.add_argument("--seed", type=int)
    gen_parser.add_argument("--report-out", type=Path)
    gen_parser.set_defaults(func=generate)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path)
    eval_parser.add_argument("--k", type=int, default=4)
    eval_parser.add_argument("--text-dim", type=int, default=256)
    eval_parser.add_argument("--hidden-dim", type=int, default=512)
    eval_parser.add_argument("--max-norm", type=float, default=4.0)
    eval_parser.add_argument("--retrieval-metric")
    eval_parser.add_argument("--metric-weight", type=float, default=0.0)
    eval_parser.add_argument("--min-retrieval-score", type=float)
    eval_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    eval_parser.add_argument("--seed", type=int)
    eval_parser.add_argument("--ablation", action="store_true")
    eval_parser.add_argument("--report-out", type=Path)
    eval_parser.set_defaults(func=evaluate)

    toy_parser = sub.add_parser("toy-eval")
    toy_parser.add_argument("--adapter", type=Path)
    toy_parser.add_argument("--report-out", type=Path)
    toy_parser.set_defaults(func=toy_eval)

    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--adapter", type=Path, required=True)
    gate_parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    gate_parser.add_argument("--max-uncertainty", type=float, default=0.8)
    gate_parser.add_argument("--max-tensor-norm", type=float, default=8.0)
    gate_parser.add_argument("--consistency-report", type=Path)
    gate_parser.add_argument("--max-consistency-mse", type=float)
    gate_parser.add_argument("--sensitivity-report", type=Path)
    gate_parser.add_argument("--min-prompt-sensitivity-mse", type=float)
    gate_parser.add_argument("--min-retrieval-score-delta", type=float)
    gate_parser.add_argument("--reliability-report", type=Path)
    gate_parser.add_argument("--max-ece", type=float)
    gate_parser.add_argument("--max-aurc", type=float)
    gate_parser.add_argument("--max-selective-risk", type=float)
    gate_parser.add_argument("--selective-risk-coverage", default="coverage_0.8")
    gate_parser.add_argument("--benchmark", type=Path)
    gate_parser.add_argument("--module", default="layer")
    gate_parser.add_argument("--hf-benchmark", type=Path)
    gate_parser.add_argument("--hf-generation-benchmark", type=Path)
    gate_parser.add_argument("--model", default="")
    gate_parser.add_argument("--batch-size", type=int, default=8)
    gate_parser.add_argument("--max-length", type=int, default=256)
    gate_parser.add_argument("--max-new-tokens", type=int, default=32)
    gate_parser.add_argument("--device", default="cpu")
    gate_parser.add_argument("--allow-download", action="store_true")
    gate_parser.add_argument("--report-out", type=Path)
    gate_parser.set_defaults(func=gate)

    export_parser = sub.add_parser("export-peft")
    export_parser.add_argument("--adapter", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.add_argument("--base-model", default="")
    export_parser.add_argument("--report-out", type=Path)
    export_parser.set_defaults(func=export_peft)

    inspect_parser = sub.add_parser("inspect-peft")
    inspect_parser.add_argument("--adapter-dir", type=Path, required=True)
    inspect_parser.add_argument("--report-out", type=Path)
    inspect_parser.set_defaults(func=inspect_peft)

    merge_parser = sub.add_parser("merge-adapter")
    merge_parser.add_argument("--adapter", type=Path, required=True)
    merge_parser.add_argument("--model", required=True)
    merge_parser.add_argument("--out", type=Path, required=True)
    merge_parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    merge_parser.add_argument("--device", default="cpu")
    merge_parser.add_argument("--allow-download", action="store_true")
    merge_parser.add_argument("--report-out", type=Path)
    merge_parser.set_defaults(func=merge_adapter)

    merge_multi_parser = sub.add_parser("merge-adapters")
    merge_multi_parser.add_argument("--adapters", nargs="+", type=Path, required=True)
    merge_multi_parser.add_argument("--model", required=True)
    merge_multi_parser.add_argument("--out", type=Path, required=True)
    merge_multi_parser.add_argument("--weights", default=None, help="Comma-separated weights (e.g., 0.5,0.5)")
    merge_multi_parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    merge_multi_parser.add_argument("--device", default="cpu")
    merge_multi_parser.add_argument("--allow-download", action="store_true")
    merge_multi_parser.add_argument("--report-out", type=Path)
    merge_multi_parser.set_defaults(func=merge_adapters)

    bench_parser = sub.add_parser("bench")
    bench_parser.add_argument("--adapter", type=Path, required=True)
    bench_parser.add_argument("--benchmark", type=Path, required=True)
    bench_parser.add_argument("--module", default="layer")
    bench_parser.add_argument("--report-out", type=Path)
    bench_parser.set_defaults(func=bench)

    hf_bench_parser = sub.add_parser("hf-bench")
    hf_bench_parser.add_argument("--adapter", type=Path, required=True)
    hf_bench_parser.add_argument("--benchmark", type=Path, required=True)
    hf_bench_parser.add_argument("--model", required=True)
    hf_bench_parser.add_argument("--batch-size", type=int, default=8)
    hf_bench_parser.add_argument("--max-length", type=int, default=256)
    hf_bench_parser.add_argument("--device", default="cpu")
    hf_bench_parser.add_argument("--allow-download", action="store_true")
    hf_bench_parser.add_argument("--report-out", type=Path)
    hf_bench_parser.set_defaults(func=hf_bench)

    hf_generate_parser = sub.add_parser("hf-generate-bench")
    hf_generate_parser.add_argument("--adapter", type=Path, required=True)
    hf_generate_parser.add_argument("--benchmark", type=Path, required=True)
    hf_generate_parser.add_argument("--model", required=True)
    hf_generate_parser.add_argument("--max-new-tokens", type=int, default=32)
    hf_generate_parser.add_argument("--max-length", type=int, default=512)
    hf_generate_parser.add_argument("--device", default="cpu")
    hf_generate_parser.add_argument("--allow-download", action="store_true")
    hf_generate_parser.add_argument("--report-out", type=Path)
    hf_generate_parser.set_defaults(func=hf_generate_bench)

    compare_parser = sub.add_parser("compare-reports")
    compare_parser.add_argument("reports", nargs="+", type=Path)
    compare_parser.add_argument("--metric", default="accuracy_delta")
    compare_parser.add_argument("--ascending", action="store_true")
    compare_parser.add_argument("--report-out", type=Path)
    compare_parser.set_defaults(func=compare_report_cmd)

    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("reports", nargs="+", type=Path)
    audit_parser.add_argument("--profile", type=Path)
    audit_parser.add_argument("--min-accuracy-delta", type=float)
    audit_parser.add_argument("--max-uncertainty", type=float)
    audit_parser.add_argument("--max-consistency-mse", type=float)
    audit_parser.add_argument("--min-prompt-sensitivity-mse", type=float)
    audit_parser.add_argument("--min-retrieval-score-delta", type=float)
    audit_parser.add_argument("--max-duplicate-pairs", type=int)
    audit_parser.add_argument("--max-duplicate-fingerprints", type=float)
    audit_parser.add_argument("--max-description-similarity", type=float)
    audit_parser.add_argument("--min-adapter-count", type=float)
    audit_parser.add_argument("--max-bank-tensor-norm", type=float)
    audit_parser.add_argument("--min-metric-coverage", action="append", default=[])
    audit_parser.add_argument("--require-ablation-not-worse", action="store_true")
    audit_parser.add_argument("--require-fingerprint-match", action="store_true")
    audit_parser.add_argument("--report-out", type=Path)
    audit_parser.set_defaults(func=audit_report_cmd)

    consistency_parser = sub.add_parser("prompt-consistency")
    consistency_parser.add_argument("--manifest", type=Path, required=True)
    consistency_parser.add_argument("--prompts", type=Path, required=True)
    consistency_parser.add_argument("--checkpoint", type=Path)
    consistency_parser.add_argument("--k", type=int, default=4)
    consistency_parser.add_argument("--text-dim", type=int, default=256)
    consistency_parser.add_argument("--hidden-dim", type=int, default=512)
    consistency_parser.add_argument("--max-norm", type=float, default=4.0)
    consistency_parser.add_argument("--retrieval-metric")
    consistency_parser.add_argument("--metric-weight", type=float, default=0.0)
    consistency_parser.add_argument("--min-retrieval-score", type=float)
    consistency_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    consistency_parser.add_argument("--seed", type=int)
    consistency_parser.add_argument("--report-out", type=Path)
    consistency_parser.set_defaults(func=prompt_consistency)

    sensitivity_parser = sub.add_parser("prompt-sensitivity")
    sensitivity_parser.add_argument("--manifest", type=Path, required=True)
    sensitivity_parser.add_argument("--contrasts", type=Path, required=True)
    sensitivity_parser.add_argument("--checkpoint", type=Path)
    sensitivity_parser.add_argument("--k", type=int, default=4)
    sensitivity_parser.add_argument("--text-dim", type=int, default=256)
    sensitivity_parser.add_argument("--hidden-dim", type=int, default=512)
    sensitivity_parser.add_argument("--max-norm", type=float, default=4.0)
    sensitivity_parser.add_argument("--retrieval-metric")
    sensitivity_parser.add_argument("--metric-weight", type=float, default=0.0)
    sensitivity_parser.add_argument("--min-retrieval-score", type=float)
    sensitivity_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    sensitivity_parser.add_argument("--seed", type=int)
    sensitivity_parser.add_argument("--report-out", type=Path)
    sensitivity_parser.set_defaults(func=prompt_sensitivity)

    reliability_parser = sub.add_parser("reliability")
    reliability_parser.add_argument("--manifest", type=Path, required=True)
    reliability_parser.add_argument("--checkpoint", type=Path)
    reliability_parser.add_argument("--tolerance", type=float, default=0.05)
    reliability_parser.add_argument("--k", type=int, default=4)
    reliability_parser.add_argument("--text-dim", type=int, default=256)
    reliability_parser.add_argument("--hidden-dim", type=int, default=512)
    reliability_parser.add_argument("--max-norm", type=float, default=4.0)
    reliability_parser.add_argument("--retrieval-metric")
    reliability_parser.add_argument("--metric-weight", type=float, default=0.0)
    reliability_parser.add_argument("--min-retrieval-score", type=float)
    reliability_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    reliability_parser.add_argument("--no-calibrate", action="store_true")
    reliability_parser.add_argument("--seed", type=int)
    reliability_parser.add_argument("--report-out", type=Path)
    reliability_parser.set_defaults(func=reliability)

    zero_shot_parser = sub.add_parser("zero-shot")
    zero_shot_parser.add_argument("--manifest", type=Path, required=True)
    zero_shot_parser.add_argument("--holdout-fraction", type=float, default=0.3)
    zero_shot_parser.add_argument("--train-steps", type=int, default=200)
    zero_shot_parser.add_argument("--lr", type=float, default=1e-2)
    zero_shot_parser.add_argument("--tolerance", type=float, default=0.05)
    zero_shot_parser.add_argument("--text-dim", type=int, default=256)
    zero_shot_parser.add_argument("--hidden-dim", type=int, default=512)
    zero_shot_parser.add_argument("--max-norm", type=float, default=8.0)
    zero_shot_parser.add_argument("--no-calibrate", action="store_true")
    zero_shot_parser.add_argument("--seed", type=int)
    zero_shot_parser.add_argument("--report-out", type=Path)
    zero_shot_parser.set_defaults(func=zero_shot)

    pipeline_parser = sub.add_parser("pipeline")
    pipeline_parser.add_argument("--manifest", type=Path, required=True)
    pipeline_parser.add_argument("--prompt", required=True)
    pipeline_parser.add_argument("--benchmark", type=Path, required=True)
    pipeline_parser.add_argument("--adapter-out", type=Path, required=True)
    pipeline_parser.add_argument("--checkpoint", type=Path)
    pipeline_parser.add_argument("--export-dir", type=Path)
    pipeline_parser.add_argument("--base-model", default="")
    pipeline_parser.add_argument("--module", default="layer")
    pipeline_parser.add_argument("--train-steps", type=int, default=0)
    pipeline_parser.add_argument("--lr", type=float, default=1e-3)
    pipeline_parser.add_argument("--k", type=int, default=4)
    pipeline_parser.add_argument("--text-dim", type=int, default=256)
    pipeline_parser.add_argument("--hidden-dim", type=int, default=512)
    pipeline_parser.add_argument("--max-norm", type=float, default=4.0)
    pipeline_parser.add_argument("--retrieval-metric")
    pipeline_parser.add_argument("--metric-weight", type=float, default=0.0)
    pipeline_parser.add_argument("--min-retrieval-score", type=float)
    pipeline_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    pipeline_parser.add_argument("--seed", type=int)
    pipeline_parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    pipeline_parser.add_argument("--max-uncertainty", type=float, default=0.8)
    pipeline_parser.add_argument("--max-tensor-norm", type=float, default=8.0)
    pipeline_parser.add_argument("--report-out", type=Path)
    pipeline_parser.set_defaults(func=pipeline)

    loo_parser = sub.add_parser("loo-eval")
    loo_parser.add_argument("--manifest", type=Path, required=True)
    loo_parser.add_argument("--steps", type=int, default=100)
    loo_parser.add_argument("--lr", type=float, default=1e-3)
    loo_parser.add_argument("--k", type=int, default=4)
    loo_parser.add_argument("--text-dim", type=int, default=256)
    loo_parser.add_argument("--hidden-dim", type=int, default=512)
    loo_parser.add_argument("--max-norm", type=float, default=4.0)
    loo_parser.add_argument("--retrieval-metric")
    loo_parser.add_argument("--metric-weight", type=float, default=0.0)
    loo_parser.add_argument("--min-retrieval-score", type=float)
    loo_parser.add_argument("--ood-shrink-factor", type=float, default=0.25)
    loo_parser.add_argument("--seed", type=int)
    loo_parser.add_argument("--report-out", type=Path)
    loo_parser.set_defaults(func=loo_eval)

    validate_parser = sub.add_parser("validate-manifest")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--required-metric", action="append", default=[])
    validate_parser.add_argument("--duplicate-similarity-threshold", type=float, default=0.98)
    validate_parser.add_argument("--text-dim", type=int, default=256)
    validate_parser.add_argument("--report-out", type=Path)
    validate_parser.set_defaults(func=validate_manifest_cmd)

    summary_parser = sub.add_parser("bank-summary")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--text-dim", type=int, default=256)
    summary_parser.add_argument("--report-out", type=Path)
    summary_parser.set_defaults(func=bank_summary_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
