from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from true_lora.adapter import infer_lora_tensor_specs
from true_lora.apply import temporary_lora


def has_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def load_text_classification_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not isinstance(row.get("label"), int):
                raise ValueError(f"{path}:{line_number} requires text:str and label:int")
            rows.append({"text": row["text"], "label": int(row["label"])})
    if not rows:
        raise ValueError(f"{path} did not contain text classification examples")
    return rows


def load_generation_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
                raise ValueError(f"{path}:{line_number} requires prompt:str and answer:str")
            rows.append({"prompt": row["prompt"], "answer": row["answer"]})
    if not rows:
        raise ValueError(f"{path} did not contain generation examples")
    return rows


def evaluate_hf_sequence_classification(
    model_name_or_path: str,
    state_dict: dict[str, torch.Tensor],
    dataset_path: Path,
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cpu",
    local_files_only: bool = True,
) -> dict[str, float]:
    if not has_transformers():
        raise RuntimeError("transformers is required for Hugging Face evaluation")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = load_text_classification_jsonl(dataset_path)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    specs = infer_lora_tensor_specs(state_dict)
    baseline = _hf_accuracy(model, tokenizer, rows, labels, batch_size, max_length, device)
    with temporary_lora(model, state_dict, specs, strict=False):
        adapted = _hf_accuracy(model, tokenizer, rows, labels, batch_size, max_length, device)
    restored = _hf_accuracy(model, tokenizer, rows, labels, batch_size, max_length, device)
    return {
        "baseline_accuracy": baseline,
        "adapted_accuracy": adapted,
        "restored_accuracy": restored,
        "examples": float(len(rows)),
        "applied_specs": float(len(specs)),
    }


def evaluate_hf_causal_lm_generation(
    model_name_or_path: str,
    state_dict: dict[str, torch.Tensor],
    dataset_path: Path,
    max_new_tokens: int = 32,
    max_length: int = 512,
    device: str = "cpu",
    local_files_only: bool = True,
) -> dict[str, float]:
    if not has_transformers():
        raise RuntimeError("transformers is required for Hugging Face evaluation")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_generation_jsonl(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    specs = infer_lora_tensor_specs(state_dict)
    baseline = _generation_exact_match(model, tokenizer, rows, max_new_tokens, max_length, device)
    with temporary_lora(model, state_dict, specs, strict=False):
        adapted = _generation_exact_match(model, tokenizer, rows, max_new_tokens, max_length, device)
    restored = _generation_exact_match(model, tokenizer, rows, max_new_tokens, max_length, device)
    return {
        "baseline_exact_match": baseline,
        "adapted_exact_match": adapted,
        "restored_exact_match": restored,
        "baseline_accuracy": baseline,
        "adapted_accuracy": adapted,
        "restored_accuracy": restored,
        "examples": float(len(rows)),
        "applied_specs": float(len(specs)),
    }


def _hf_accuracy(model, tokenizer, rows, labels, batch_size: int, max_length: int, device: str) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu()
            pred = logits.argmax(dim=-1)
            gold = labels[start : start + len(batch)]
            correct += int((pred == gold).sum())
            total += len(batch)
    return correct / total


def _generation_exact_match(model, tokenizer, rows, max_new_tokens: int, max_length: int, device: str) -> float:
    correct = 0
    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(
                row["prompt"],
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[0, prompt_tokens:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if _normalize_text(text) == _normalize_text(row["answer"]):
                correct += 1
    return correct / len(rows)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())
