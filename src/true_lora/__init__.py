from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec, adapter_fingerprint, validate_adapter_manifest
from true_lora.apply import lora_delta, merge_lora_into_linear, temporary_lora
from true_lora.bank import adapter_bank_summary
from true_lora.benchmark import evaluate_classification, load_classification_jsonl
from true_lora.consistency import adapter_pair_mse, load_prompt_groups, prompt_consistency_report
from true_lora.generator import (
    ConditionedHyperAdapter,
    HyperAdapter,
    TrueLoraGenerator,
    layer_index,
    load_true_lora_checkpoint,
    module_key,
)
from true_lora.text import HashingTextEncoder, SemanticTextEncoder
from true_lora.hf_eval import (
    evaluate_hf_causal_lm_generation,
    evaluate_hf_sequence_classification,
    load_generation_jsonl,
    load_text_classification_jsonl,
)
from true_lora.merge import merge_adapter_into_hf_model, merge_adapters
from true_lora.peft_io import inspect_peft_directory, load_peft_directory, load_peft_model
from true_lora.reliability import (
    HistogramBinningCalibrator,
    area_under_risk_coverage,
    collect_generation_records,
    expected_calibration_error,
    reliability_report,
    reliability_report_for_adapters,
    risk_coverage_points,
    selective_risk_at_coverage,
)
from true_lora.repro import set_seed
from true_lora.reporting import audit_reports, compare_reports, load_audit_profile, load_json_report, write_json_report
from true_lora.sensitivity import PromptContrast, load_prompt_contrasts, prompt_sensitivity_report
from true_lora.train import ablation_report

__all__ = [
    "AdapterBank",
    "AdapterSpec",
    "LoraTensorSpec",
    "adapter_fingerprint",
    "validate_adapter_manifest",
    "TrueLoraGenerator",
    "HyperAdapter",
    "ConditionedHyperAdapter",
    "HashingTextEncoder",
    "SemanticTextEncoder",
    "module_key",
    "layer_index",
    "load_true_lora_checkpoint",
    "lora_delta",
    "merge_lora_into_linear",
    "temporary_lora",
    "adapter_bank_summary",
    "evaluate_classification",
    "load_classification_jsonl",
    "adapter_pair_mse",
    "load_prompt_groups",
    "prompt_consistency_report",
    "evaluate_hf_sequence_classification",
    "evaluate_hf_causal_lm_generation",
    "load_generation_jsonl",
    "load_text_classification_jsonl",
    "merge_adapter_into_hf_model",
    "merge_adapters",
    "inspect_peft_directory",
    "load_peft_directory",
    "load_peft_model",
    "compare_reports",
    "audit_reports",
    "load_audit_profile",
    "load_json_report",
    "write_json_report",
    "set_seed",
    "PromptContrast",
    "load_prompt_contrasts",
    "prompt_sensitivity_report",
    "ablation_report",
    "expected_calibration_error",
    "risk_coverage_points",
    "area_under_risk_coverage",
    "selective_risk_at_coverage",
    "HistogramBinningCalibrator",
    "reliability_report",
    "collect_generation_records",
    "reliability_report_for_adapters",
]
