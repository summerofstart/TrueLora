from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec
from true_lora.adapter import adapter_fingerprint, load_adapter_manifest, save_peft_adapter, save_peft_directory, validate_adapter_manifest
from true_lora.apply import lora_delta, temporary_lora
from true_lora.bank import adapter_bank_summary
from true_lora.benchmark import evaluate_classification, load_classification_jsonl
from true_lora.consistency import load_prompt_groups, prompt_consistency_report
from true_lora.generator import TrueLoraGenerator, load_true_lora_checkpoint
from true_lora.hf_eval import load_generation_jsonl, load_text_classification_jsonl
from true_lora.peft_io import inspect_peft_directory
from true_lora.quality import QualityGate, gate_adapter
from true_lora.reporting import audit_reports, compare_reports, load_audit_profile, write_json_report
from true_lora.repro import set_seed
from true_lora.sensitivity import load_prompt_contrasts, prompt_sensitivity_report
from true_lora.text import HashingTextEncoder
from true_lora.toy_eval import ToyClassifier, accuracy_with_adapter, adapter_for_sign_task
from true_lora.train import ablation_report, leave_one_out_report

from pathlib import Path
import tempfile
import torch


def main() -> None:
    encoder = HashingTextEncoder(dim=32)
    adapters = [
        AdapterSpec(
            "math reasoning",
            encoder.encode("math reasoning"),
            {
                "layer.lora_A.weight": torch.ones(2, 4),
                "layer.lora_B.weight": torch.ones(4, 2),
            },
            metrics={"score": 0.1},
        ),
        AdapterSpec(
            "translation writing",
            encoder.encode("translation writing"),
            {
                "layer.lora_A.weight": -torch.ones(2, 4),
                "layer.lora_B.weight": -torch.ones(4, 2),
            },
            metrics={"score": 0.9},
        ),
    ]
    bank = AdapterBank(adapters)
    found, weights = bank.retrieve(encoder.encode("math reasoning algebra"), k=2)
    assert found[0].description == "math reasoning"
    assert weights[0] > weights[1]
    metric_found, _ = bank.retrieve(encoder.encode("math reasoning algebra"), k=2, metric="score", metric_weight=4.0)
    assert metric_found[0].description == "translation writing"

    model = TrueLoraGenerator(
        [LoraTensorSpec("layer", out_features=4, in_features=4, rank=2)],
        bank,
        text_dim=32,
        hidden_dim=16,
        max_tensor_norm=0.5,
    )
    state_dict, report = model.generate("math reasoning", retrieval_k=2, retrieval_metric="score", metric_weight=0.5)
    assert set(state_dict) == {"layer.lora_A.weight", "layer.lora_B.weight"}
    assert adapter_fingerprint(state_dict) == adapter_fingerprint(dict(reversed(list(state_dict.items()))))
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    assert 0.0 <= report["uncertainty"] <= 1.0
    assert report["metric_weight"] == 0.5
    assert report["retrieved_adapters"][0]["description"] == "math reasoning"
    assert report["retrieved_adapters"][0]["metrics"]["score"] == 0.1
    assert report["retrieved_adapters"][1]["description"] == "translation writing"
    assert report["retrieved_adapters"][0]["weight"] > 0.0
    assert len(report["retrieved_adapters"][0]["fingerprint"]) == 64
    components, component_report = model.generate_components("math reasoning", retrieval_k=2)
    assert set(components) == {"blended", "retrieval", "generated"}
    assert set(components["blended"]) == set(state_dict)
    assert component_report["generated_weight"] >= 0.0
    shrunk_state, shrunk_report = model.generate("math reasoning", retrieval_k=2, min_retrieval_score=2.0)
    assert shrunk_report["abstained"] == 1.0
    assert shrunk_report["shrink_factor"] == 0.25
    assert sum(t.norm() for t in shrunk_state.values()) < sum(t.norm() for t in state_dict.values())
    set_seed(123)
    seeded_a = TrueLoraGenerator(
        [LoraTensorSpec("layer", out_features=4, in_features=4, rank=2)],
        bank,
        text_dim=32,
        hidden_dim=16,
        max_tensor_norm=0.5,
    )
    seeded_state_a, _ = seeded_a.generate("math reasoning", retrieval_k=2)
    set_seed(123)
    seeded_b = TrueLoraGenerator(
        [LoraTensorSpec("layer", out_features=4, in_features=4, rank=2)],
        bank,
        text_dim=32,
        hidden_dim=16,
        max_tensor_norm=0.5,
    )
    seeded_state_b, _ = seeded_b.generate("math reasoning", retrieval_k=2)
    assert all(torch.equal(seeded_state_a[name], seeded_state_b[name]) for name in seeded_state_a)

    delta = lora_delta(torch.ones(2, 4), torch.ones(4, 2), alpha=2.0)
    assert delta.shape == (4, 4)

    toy_adapter = adapter_for_sign_task()
    toy_report = accuracy_with_adapter(toy_adapter)
    assert toy_report["adapted_accuracy"] > toy_report["baseline_accuracy"]
    assert toy_report["restored_accuracy"] == toy_report["baseline_accuracy"]
    gate_report = gate_adapter(
        toy_adapter,
        toy_report,
        generation_report={"uncertainty": 0.1},
        gate=QualityGate(min_accuracy_delta=0.1),
    )
    assert gate_report["accepted"] is True
    unstable_gate = gate_adapter(
        toy_adapter,
        toy_report,
        generation_report={"uncertainty": 0.1},
        consistency_report={"mean_pairwise_mse": 0.5},
        gate=QualityGate(min_accuracy_delta=0.1, max_consistency_mse=0.1),
    )
    assert unstable_gate["accepted"] is False
    assert unstable_gate["failures"] == "consistency_mse"
    insensitive_gate = gate_adapter(
        toy_adapter,
        toy_report,
        generation_report={"uncertainty": 0.1},
        sensitivity_report={"mean_control_mse": 0.01, "mean_retrieval_score_delta": -0.2},
        gate=QualityGate(min_accuracy_delta=0.1, min_prompt_sensitivity_mse=0.1, min_retrieval_score_delta=0.0),
    )
    assert insensitive_gate["accepted"] is False
    assert insensitive_gate["failures"] == "prompt_sensitivity_mse,retrieval_score_delta"

    toy_model = ToyClassifier()
    original = toy_model.layer.weight.detach().clone()
    with temporary_lora(toy_model, toy_adapter, [LoraTensorSpec("layer", 2, 4, 2, alpha=2.0)]) as applied:
        assert applied == ["layer"]
        assert not torch.equal(toy_model.layer.weight, original)
    assert torch.equal(toy_model.layer.weight, original)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "first.pt"
        second = root / "second.pt"
        save_peft_adapter(first, adapters[0].tensors)
        save_peft_adapter(second, adapters[1].tensors)
        manifest = root / "adapters.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    '{"description":"math reasoning","path":"first.pt"}',
                    '{"description":"translation writing","path":"second.pt"}',
                ]
            ),
            encoding="utf-8",
        )
        specs, manifest_bank, manifest_adapters = load_adapter_manifest(manifest, encoder)
        assert len(specs) == 1
        assert len(manifest_bank.adapters) == 2
        assert manifest_adapters[0].source == str(first.resolve())
        assert len(manifest_adapters[0].fingerprint or "") == 64
        summary = adapter_bank_summary(manifest_adapters)
        assert summary["adapter_count"] == 2.0
        assert summary["unique_fingerprints"] == 2.0
        assert summary["duplicate_fingerprints"] == 0.0
        assert summary["description_similarity"]["count"] == 1.0
        manifest_report = validate_adapter_manifest(manifest, required_metrics=["score"])
        assert manifest_report["ok"] is True
        assert len(manifest_report["warnings"]) == 2
        assert len(manifest_report["rows"][0]["fingerprint"]) == 64
        duplicate_manifest = root / "duplicate-adapters.jsonl"
        duplicate_manifest.write_text(
            "\n".join(
                [
                    '{"description":"math reasoning","path":"first.pt"}',
                    '{"description":"math reasoning","path":"second.pt"}',
                ]
            ),
            encoding="utf-8",
        )
        duplicate_report = validate_adapter_manifest(duplicate_manifest)
        assert duplicate_report["ok"] is True
        assert len(duplicate_report["duplicate_pairs"]) == 1
        assert duplicate_report["duplicate_pairs"][0]["left_line"] == 1
        loo = leave_one_out_report(specs, manifest_adapters, text_dim=32, hidden_dim=16, steps=1)
        assert loo["examples"] == 2.0
        ablated = ablation_report(model, adapters, retrieval_k=2)
        assert ablated["examples"] == 2.0
        assert "mean_blended_mse" in ablated
        assert "mean_retrieval_mse" in ablated
        assert "mean_generated_mse" in ablated
        prompt_file = root / "prompts.jsonl"
        prompt_file.write_text(
            "\n".join(
                [
                    '{"group":"code","prompt":"python tests debugging"}',
                    '{"group":"code","prompt":"debugging unit tests in python"}',
                ]
            ),
            encoding="utf-8",
        )
        consistency = prompt_consistency_report(model, load_prompt_groups(prompt_file), retrieval_k=2)
        assert consistency["examples"] == 2.0
        assert consistency["group_count"] == 1.0
        assert consistency["pairs"] == 1.0
        contrasts_file = root / "contrasts.jsonl"
        contrasts_file.write_text(
            '{"group":"code","aligned":"python tests debugging","control":"polite japanese translation"}\n',
            encoding="utf-8",
        )
        sensitivity = prompt_sensitivity_report(model, load_prompt_contrasts(contrasts_file), retrieval_k=2)
        assert sensitivity["examples"] == 1.0
        assert "mean_control_mse" in sensitivity
        assert "mean_retrieval_score_delta" in sensitivity
        checkpoint = root / "checkpoint.pt"
        torch.save(
            {
                "hyper_state_dict": model.hyper.state_dict(),
                "tensor_specs": [spec.__dict__ for spec in [LoraTensorSpec("layer", 4, 4, 2)]],
                "text_dim": 32,
                "hidden_dim": 16,
                "max_tensor_norm": 0.5,
            },
            checkpoint,
        )
        loaded_model, loaded_checkpoint = load_true_lora_checkpoint(
            checkpoint,
            manifest_bank,
            expected_specs=[LoraTensorSpec("layer", 4, 4, 2)],
        )
        assert loaded_checkpoint["text_dim"] == 32
        assert loaded_model.max_tensor_norm == 0.5
        try:
            load_true_lora_checkpoint(checkpoint, manifest_bank, expected_specs=[LoraTensorSpec("other", 4, 4, 2)])
            raise AssertionError("expected checkpoint spec mismatch")
        except ValueError:
            pass
        export_dir = root / "peft"
        save_peft_directory(export_dir, adapters[0].tensors, specs, {"uncertainty": 0.1}, "toy-base")
        assert (export_dir / "adapter_model.bin").exists()
        assert "true_lora_tensor_specs" in (export_dir / "adapter_config.json").read_text(encoding="utf-8")
        peft_report = inspect_peft_directory(export_dir)
        assert peft_report["base_model_name_or_path"] == "toy-base"
        assert peft_report["spec_count"] == 1
        saved_obj = torch.load(first, map_location="cpu")
        assert len(saved_obj["true_lora_report"]["adapter_fingerprint"]) == 64
        bench = root / "bench.jsonl"
        bench.write_text(
            "\n".join(
                [
                    '{"features":[1,1,0,0],"label":1}',
                    '{"features":[-1,-1,0,0],"label":0}',
                    '{"features":[2,-3,0,0],"label":0}',
                    '{"features":[3,1,0,0],"label":1}',
                ]
            ),
            encoding="utf-8",
        )
        loaded_bench = load_classification_jsonl(bench, toy_adapter)
        bench_report = evaluate_classification(toy_adapter, loaded_bench)
        assert bench_report["adapted_accuracy"] == 1.0
        text_bench = root / "text-bench.jsonl"
        text_bench.write_text(
            "\n".join(
                [
                    '{"text":"correct answer","label":1}',
                    '{"text":"wrong answer","label":0}',
                ]
            ),
            encoding="utf-8",
        )
        assert len(load_text_classification_jsonl(text_bench)) == 2
        generation_bench = root / "generation-bench.jsonl"
        generation_bench.write_text(
            "\n".join(
                [
                    '{"prompt":"2+2=","answer":"4"}',
                    '{"prompt":"capital of France:","answer":"Paris"}',
                ]
            ),
            encoding="utf-8",
        )
        assert len(load_generation_jsonl(generation_bench)) == 2
        report_path = root / "report.json"
        write_json_report(report_path, {"path": report_path, "value": torch.tensor(1.5)})
        assert '"value": 1.5' in report_path.read_text(encoding="utf-8")
        first_report = root / "first-report.json"
        second_report = root / "second-report.json"
        write_json_report(first_report, {"command": "gate", "report": {"accuracy_delta": 0.1}})
        write_json_report(second_report, {"command": "gate", "report": {"accuracy_delta": 0.4}})
        compared = compare_reports([first_report, second_report])
        assert compared[0]["path"] == str(second_report)
        consistency_report = root / "consistency-report.json"
        sensitivity_report = root / "sensitivity-report.json"
        ablation_report_path = root / "ablation-report.json"
        manifest_audit_report = root / "manifest-audit-report.json"
        generation_audit_report = root / "generation-audit-report.json"
        bank_summary_report = root / "bank-summary-report.json"
        write_json_report(consistency_report, {"command": "prompt-consistency", "report": {"mean_pairwise_mse": 0.01}})
        write_json_report(
            sensitivity_report,
            {"command": "prompt-sensitivity", "report": {"mean_control_mse": 0.4, "mean_retrieval_score_delta": 0.2}},
        )
        write_json_report(
            ablation_report_path,
            {"command": "eval", "ablation": {"mean_blended_mse": 0.1, "mean_retrieval_mse": 0.2}},
        )
        write_json_report(
            manifest_audit_report,
            {"command": "validate-manifest", "report": {"rows": [{"path": "adapter-a.pt", "fingerprint": "abc"}]}},
        )
        write_json_report(
            generation_audit_report,
            {"command": "generate", "generation": {"retrieved_adapters": [{"source": "adapter-a.pt", "fingerprint": "abc"}]}},
        )
        write_json_report(
            bank_summary_report,
            {
                "command": "bank-summary",
                "report": {
                    "duplicate_fingerprints": 0.0,
                    "adapter_count": 4.0,
                    "metric_coverage": {"score": 1.0},
                    "description_similarity": {"max": 0.2},
                    "tensor_norms": {"max": 2.0},
                },
            },
        )
        audited = audit_reports(
            [
                second_report,
                consistency_report,
                sensitivity_report,
                ablation_report_path,
                manifest_audit_report,
                generation_audit_report,
                bank_summary_report,
            ],
            min_accuracy_delta=0.2,
            max_consistency_mse=0.02,
            min_prompt_sensitivity_mse=0.1,
            min_retrieval_score_delta=0.0,
            max_duplicate_fingerprints=0.0,
            max_description_similarity=0.5,
            min_adapter_count=2.0,
            max_bank_tensor_norm=4.0,
            min_metric_coverage={"score": 1.0},
            require_ablation_not_worse=True,
            require_fingerprint_match=True,
        )
        assert audited["accepted"] is True
        audit_profile = root / "audit-profile.json"
        write_json_report(
            audit_profile,
            {
                "min_accuracy_delta": 0.2,
                "max_consistency_mse": 0.02,
                "min_prompt_sensitivity_mse": 0.1,
                "min_retrieval_score_delta": 0.0,
                "max_duplicate_fingerprints": 0.0,
                "max_description_similarity": 0.5,
                "min_adapter_count": 2.0,
                "max_bank_tensor_norm": 4.0,
                "min_metric_coverage": {"score": 1.0},
                "require_ablation_not_worse": True,
                "require_fingerprint_match": True,
            },
        )
        profiled = audit_reports(
            [
                second_report,
                consistency_report,
                sensitivity_report,
                ablation_report_path,
                manifest_audit_report,
                generation_audit_report,
                bank_summary_report,
            ],
            **load_audit_profile(audit_profile),
        )
        assert profiled["accepted"] is True
        rejected = audit_reports(
            [second_report, ablation_report_path],
            min_accuracy_delta=0.2,
            require_ablation_not_worse=True,
            max_duplicate_pairs=0,
        )
        assert rejected["accepted"] is True
        write_json_report(ablation_report_path, {"command": "eval", "ablation": {"mean_blended_mse": 0.3, "mean_retrieval_mse": 0.2}})
        rejected = audit_reports([second_report, ablation_report_path], require_ablation_not_worse=True)
        assert rejected["accepted"] is False
        assert rejected["failures"] == ["ablation_blended_worse_than_retrieval"]
        write_json_report(
            generation_audit_report,
            {"command": "generate", "generation": {"retrieved_adapters": [{"source": "adapter-a.pt", "fingerprint": "def"}]}},
        )
        rejected = audit_reports([manifest_audit_report, generation_audit_report], require_fingerprint_match=True)
        assert rejected["accepted"] is False
        assert rejected["failures"] == ["fingerprint_mismatch"]
        rejected = audit_reports([bank_summary_report], min_metric_coverage={"score": 1.1})
        assert rejected["accepted"] is False
        assert rejected["failures"] == ["metric_coverage:score"]
        rejected = audit_reports([bank_summary_report], min_adapter_count=8.0)
        assert rejected["accepted"] is False
        assert rejected["failures"] == ["adapter_count"]
    print("smoke tests passed")


if __name__ == "__main__":
    main()
