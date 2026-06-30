import torch

from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec
from true_lora.generator import (
    ConditionedHyperAdapter,
    HyperAdapter,
    TrueLoraGenerator,
    layer_index,
    module_key,
)
from true_lora.reliability import (
    HistogramBinningCalibrator,
    area_under_risk_coverage,
    expected_calibration_error,
    reliability_report,
    reliability_report_for_adapters,
    risk_coverage_points,
    selective_risk_at_coverage,
)
from true_lora.text import HashingTextEncoder, SemanticTextEncoder
from true_lora.train import train_on_adapter_bank
from true_lora.zeroshot import (
    pearson_correlation,
    run_zero_shot_benchmark,
    split_adapters_by_description,
    zero_shot_benchmark,
)


def make_bank():
    encoder = HashingTextEncoder(dim=32)
    tensors_a = {
        "layer.lora_A.weight": torch.ones(2, 4),
        "layer.lora_B.weight": torch.ones(4, 2),
    }
    tensors_b = {
        "layer.lora_A.weight": -torch.ones(2, 4),
        "layer.lora_B.weight": -torch.ones(4, 2),
    }
    adapters = [
        AdapterSpec("math reasoning", encoder.encode("math reasoning"), tensors_a),
        AdapterSpec("translation writing", encoder.encode("translation writing"), tensors_b),
    ]
    return encoder, AdapterBank(adapters)


def test_retrieval_prefers_matching_description():
    encoder, bank = make_bank()
    adapters, weights = bank.retrieve(encoder.encode("math reasoning algebra"), k=2)
    assert adapters[0].description == "math reasoning"
    assert weights[0] > weights[1]


def test_generator_returns_clipped_lora_tensors():
    _, bank = make_bank()
    specs = [LoraTensorSpec("layer", out_features=4, in_features=4, rank=2)]
    model = TrueLoraGenerator(specs, bank, text_dim=32, hidden_dim=16, max_tensor_norm=0.5)
    state_dict, report = model.generate("math reasoning", retrieval_k=2)

    assert set(state_dict) == {"layer.lora_A.weight", "layer.lora_B.weight"}
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    assert 0.0 <= report["uncertainty"] <= 1.0


def make_gqa_specs(layers, ranks=4):
    # q_proj keeps full width, v_proj is narrower (grouped-query attention).
    specs = []
    for li in layers:
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.q_proj", 16, 16, ranks))
        specs.append(LoraTensorSpec(f"model.layers.{li}.self_attn.v_proj", 4, 16, ranks))
    return specs


def make_conditioned_bank(specs, encoder):
    descriptions = [("python code generation", 0.2), ("japanese translation", -0.2)]
    adapters = []
    for desc, scale in descriptions:
        tensors = {}
        for spec in specs:
            tensors[f"{spec.name}.lora_A.weight"] = torch.full(spec.a_shape, scale)
            tensors[f"{spec.name}.lora_B.weight"] = torch.full(spec.b_shape, scale / 2)
        adapters.append(AdapterSpec(desc, encoder.encode(desc), tensors, metrics={"score": 0.5}))
    return AdapterBank(adapters), adapters


def test_module_key_and_layer_index_parsing():
    assert module_key("model.layers.18.self_attn.q_proj") == "model.layers.{}.self_attn.q_proj"
    assert module_key("model.layers.0.self_attn.q_proj") == module_key("model.layers.18.self_attn.q_proj")
    assert layer_index("model.layers.18.self_attn.q_proj") == 18
    assert layer_index("layer") == 0


def test_conditioned_generator_handles_mixed_shapes():
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6, 12])
    bank, _ = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=16, max_tensor_norm=0.5,
        encoder=encoder, hyper_kind="conditioned",
    )
    state_dict, report = model.generate("python code generation", retrieval_k=2)

    # v_proj (narrow) and q_proj (full) deltas keep their distinct shapes.
    assert state_dict["model.layers.0.self_attn.v_proj.lora_B.weight"].shape == (4, 4)
    assert state_dict["model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (16, 4)
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    assert 0.0 <= report["uncertainty"] <= 1.0


def test_conditioned_hypernetwork_scales_with_module_types_not_layers():
    # Many layers, two module types. The conditioned net shares heads across layers,
    # so it stays far smaller than the flat net whose output grows with total size.
    specs = make_gqa_specs(list(range(24)))
    conditioned = ConditionedHyperAdapter(32, 64, specs)
    flat = HyperAdapter(32, 64, specs)
    cond_params = sum(p.numel() for p in conditioned.parameters())
    flat_params = sum(p.numel() for p in flat.parameters())
    assert cond_params < flat_params


def test_conditioned_hypernetwork_trains():
    torch.manual_seed(0)
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    bank, adapters = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    losses = train_on_adapter_bank(model, adapters, steps=60, lr=1e-2)
    assert losses[-1] < losses[0]


class _TinyBase(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_lora_sft_model_application_matches_lora_delta():
    from true_lora.apply import lora_delta
    from true_lora.sft import LoraSFTModel

    torch.manual_seed(0)
    dim, rank = 8, 4
    base = _TinyBase(dim)
    spec = LoraTensorSpec("proj", out_features=dim, in_features=dim, rank=rank, alpha=rank)
    a = torch.randn(rank, dim) * 0.1
    b = torch.randn(dim, rank) * 0.1
    x = torch.randn(3, dim)

    with LoraSFTModel(base, [spec]) as sft:
        sft.set_adapter({"proj.lora_A.weight": a, "proj.lora_B.weight": b})
        hooked = base(x)

    delta_w = lora_delta(a, b, alpha=spec.alpha)  # (out, in)
    expected = base(x) + x @ delta_w.T
    assert torch.allclose(hooked, expected, atol=1e-5)


def test_sft_train_hypernetwork_reduces_downstream_loss():
    from true_lora.sft import sft_train_hypernetwork

    torch.manual_seed(0)
    dim, rank = 8, 4
    base = _TinyBase(dim)
    spec = LoraTensorSpec("proj", out_features=dim, in_features=dim, rank=rank, alpha=rank)

    # A perfect rank-r LoRA exists: target = base_weight + B0 @ A0 (scale = alpha/rank = 1).
    a0 = torch.randn(rank, dim) * 0.3
    b0 = torch.randn(dim, rank) * 0.3
    target_w = base.proj.weight.detach() + b0 @ a0
    x = torch.randn(32, dim)
    y = x @ target_w.T

    def mse_loss_fn(model, payload):
        xb, yb = payload
        return torch.nn.functional.mse_loss(model(xb), yb)

    encoder = HashingTextEncoder(dim=16)
    generator = TrueLoraGenerator(
        [spec], adapter_bank=None, text_dim=16, hidden_dim=32,
        encoder=encoder, hyper_kind="conditioned",
    )
    losses = sft_train_hypernetwork(
        generator, base, [("correct the projection", (x, y))], mse_loss_fn,
        steps=250, lr=1e-2,
    )

    # End-to-end SFT drives the downstream loss down...
    assert losses[-1] < losses[0] * 0.5
    # ...and the base model stays frozen while the hypernetwork learns.
    assert base.proj.weight.requires_grad is False
    assert any(p.grad is not None for p in generator.hyper.parameters())


def test_bankless_generation_uses_hypernetwork_only():
    import math

    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    _, adapters = make_conditioned_bank(specs, encoder)
    # No adapter_bank: pure text-to-LoRA via the hypernetwork.
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=32, hidden_dim=32, max_tensor_norm=0.5,
        encoder=encoder, hyper_kind="conditioned",
    )
    # Training does not need a bank -- it learns straight from (description -> tensors).
    losses = train_on_adapter_bank(model, adapters, steps=40, lr=1e-2)
    assert losses[-1] < losses[0]

    state_dict, report = model.generate("python code generation", retrieval_k=8)
    assert state_dict["model.layers.0.self_attn.v_proj.lora_B.weight"].shape == (4, 4)
    assert all(tensor.norm() <= 0.5001 for tensor in state_dict.values())
    # No retrieval happened.
    assert report["retrieved_adapters"] == []
    assert math.isnan(report["max_retrieval_score"])
    assert report["generated_weight"] == 1.0
    assert 0.0 <= report["uncertainty"] <= 1.0


def test_ensemble_off_matches_single_forward():
    # Default (ensemble=1) must be byte-for-byte the old single-forward behavior.
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    base, base_rep = model.generate("python code generation")
    same, same_rep = model.generate("python code generation", ensemble=1)
    assert set(base) == set(same)
    for key in base:
        assert torch.equal(base[key], same[key])
    assert base_rep["uncertainty"] == same_rep["uncertainty"]
    assert base_rep["epistemic"] == 0.0
    assert base_rep["ensemble_size"] == 1.0


def test_ensemble_zero_noise_is_identity():
    # With no perturbation every member is the same forward, so the averaged adapter
    # equals the single forward and disagreement is zero.
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    base, _ = model.generate("python code generation")
    pooled, rep = model.generate("python code generation", ensemble=8, ensemble_noise=0.0)
    for key in base:
        assert torch.allclose(base[key], pooled[key], atol=1e-6)
    assert rep["epistemic"] == 0.0


def test_ensemble_produces_epistemic_signal_and_is_deterministic():
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6, 12])
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=32, hidden_dim=32, max_tensor_norm=0.5,
        encoder=encoder, hyper_kind="conditioned",
    )
    single, single_rep = model.generate("python code generation")
    a, rep_a = model.generate("python code generation", ensemble=8, ensemble_noise=0.1)
    b, rep_b = model.generate("python code generation", ensemble=8, ensemble_noise=0.1)

    # A real epistemic signal appears and raises the reported uncertainty.
    assert rep_a["epistemic"] > 0.0
    assert rep_a["ensemble_size"] == 8.0
    assert rep_a["uncertainty"] >= single_rep["uncertainty"] - 1e-9
    # Shapes are preserved and the norm clip still holds on the averaged adapter.
    assert a["model.layers.0.self_attn.v_proj.lora_B.weight"].shape == (4, 4)
    assert all(tensor.norm() <= 0.5001 for tensor in a.values())
    # Deterministic in the seed: same prompt + settings -> identical adapter.
    for key in a:
        assert torch.equal(a[key], b[key])
    assert rep_a["epistemic"] == rep_b["epistemic"]


def test_ensemble_improves_calibration_linkage_on_unseen_tasks():
    # The headline claim: ensemble disagreement gives Text-to-LoRA a confidence that
    # actually tracks the generalization gap, where the single-forward variance head
    # does not. Same trained model, scored two ways.
    torch.manual_seed(0)
    encoder = HashingTextEncoder(dim=64)
    specs = make_gqa_specs([0, 6, 12])
    descriptions = [f"task alpha {i}" for i in range(6)] + [f"domain beta {i}" for i in range(6)]
    # Distinct random adapters (real LoRAs differ structurally, not just by a scalar).
    gen = torch.Generator().manual_seed(0)
    adapters = []
    for idx, desc in enumerate(descriptions):
        scale = 0.1 + 0.05 * idx
        tensors = {}
        for spec in specs:
            tensors[f"{spec.name}.lora_A.weight"] = torch.randn(spec.a_shape, generator=gen) * scale
            tensors[f"{spec.name}.lora_B.weight"] = torch.randn(spec.b_shape, generator=gen) * scale
        adapters.append(AdapterSpec(desc, encoder.encode(desc), tensors, metrics={"score": 0.5}))
    train, heldout = split_adapters_by_description(adapters, holdout_fraction=0.34, seed=0)
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=64, hidden_dim=64, max_tensor_norm=8.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    train_on_adapter_bank(model, train, steps=300, lr=1e-2)

    single = zero_shot_benchmark(model, train, heldout, tolerance=0.05, ensemble=1)
    ens = zero_shot_benchmark(
        model, train, heldout, tolerance=0.05, ensemble=9, ensemble_noise=0.05,
    )

    # Ensemble confidence ranks unseen tasks by quality; the variance head does not.
    assert ens["calibration_linkage"] > single["calibration_linkage"]
    assert ens["calibration_linkage"] > 0.3
    # Selective generation at 50% coverage answers the better half -> lower risk.
    assert (
        ens["selective_generalization"]["coverage_0.5"]
        <= ens["selective_generalization"]["coverage_1.0"] + 1e-9
    )
    # Honest: the produced adapter quality is essentially unchanged (no free lunch on
    # MSE -- the win is in the uncertainty, not the point estimate).
    assert abs(ens["heldout"]["mean_loss"] - single["heldout"]["mean_loss"]) < 0.02


def test_semantic_encoder_interface():
    encoder = SemanticTextEncoder(fallback_dim=48)
    vector = encoder.encode("python code generation")
    assert vector.shape == (encoder.dim,)
    # Vectors are L2-normalized for the cosine-similarity retrieval contract.
    assert abs(float(vector.norm()) - 1.0) < 1e-4
    assert encoder.backend in {"sentence-transformers", "hashing-fallback"}


def test_ece_zero_for_perfectly_calibrated():
    # Confidence exactly equals long-run accuracy in each group -> ECE 0.
    confidences = [0.1] * 10 + [0.9] * 10
    corrects = [1.0] + [0.0] * 9 + [1.0] * 9 + [0.0]
    report = expected_calibration_error(confidences, corrects, n_bins=10)
    assert report["ece"] < 1e-9


def test_ece_detects_overconfidence():
    # Always says 0.99 confident but only half are right -> large gap.
    confidences = [0.99] * 20
    corrects = [1.0] * 10 + [0.0] * 10
    report = expected_calibration_error(confidences, corrects, n_bins=10)
    assert report["ece"] > 0.4


def test_risk_coverage_rewards_good_confidence_ranking():
    # Confidence perfectly anti-correlates with loss: most confident are lossless.
    confidences = [0.9, 0.8, 0.7, 0.6]
    losses = [0.0, 0.0, 1.0, 1.0]
    points = risk_coverage_points(confidences, losses)
    assert points[0]["risk"] == 0.0  # most confident is correct
    assert points[-1]["coverage"] == 1.0
    # Selective generation at 50% coverage takes only the lossless half.
    assert selective_risk_at_coverage(confidences, losses, 0.5) == 0.0
    # Good ranking -> AURC below the full-coverage (mean) risk of 0.5.
    assert area_under_risk_coverage(confidences, losses) < 0.5


def test_calibrator_reduces_ece():
    torch.manual_seed(0)
    # Confidence is informative but miscalibrated (squashed toward 0.5).
    raw_conf = []
    corrects = []
    for _ in range(400):
        p = float(torch.rand(1))
        y = 1.0 if float(torch.rand(1)) < p else 0.0
        raw_conf.append(0.5 + 0.3 * (p - 0.5))  # compressed -> miscalibrated
        corrects.append(y)
    before = expected_calibration_error(raw_conf, corrects)["ece"]
    calibrator = HistogramBinningCalibrator(n_bins=10).fit(raw_conf, corrects)
    after = expected_calibration_error(calibrator.transform(raw_conf), corrects)["ece"]
    assert after <= before + 1e-9


def test_reliability_report_for_adapters_bridges_generator():
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    bank, adapters = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    report = reliability_report_for_adapters(
        model, adapters, tolerance=0.05, retrieval_k=2, min_retrieval_score=0.9
    )
    assert "ece" in report and "aurc" in report
    assert "calibrated_ece" in report
    assert 0.0 <= report["selective_risk"]["coverage_0.8"]
    assert "abstention" in report
    assert len(report["records"]) == len(adapters)


def make_task_adapters(specs, encoder, descriptions):
    # Each task gets distinct target tensors keyed off a per-task scale, so the
    # hypernetwork must actually learn a description -> weights mapping.
    adapters = []
    for idx, desc in enumerate(descriptions):
        scale = 0.1 + 0.1 * idx
        tensors = {}
        for spec in specs:
            tensors[f"{spec.name}.lora_A.weight"] = torch.full(spec.a_shape, scale)
            tensors[f"{spec.name}.lora_B.weight"] = torch.full(spec.b_shape, -scale / 2)
        adapters.append(AdapterSpec(desc, encoder.encode(desc), tensors, metrics={"score": 0.5}))
    return adapters


def test_pearson_correlation_handles_perfect_and_degenerate():
    import math

    assert abs(pearson_correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert abs(pearson_correlation([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-9
    assert math.isnan(pearson_correlation([1, 1, 1], [1, 2, 3]))  # zero variance
    assert math.isnan(pearson_correlation([5.0], [5.0]))          # <2 points


def test_split_holds_out_distinct_descriptions():
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0])
    descriptions = ["alpha task", "beta task", "gamma task", "delta task"]
    adapters = make_task_adapters(specs, encoder, descriptions)
    train, heldout = split_adapters_by_description(adapters, holdout_fraction=0.25, seed=1)

    train_desc = {a.description for a in train}
    heldout_desc = {a.description for a in heldout}
    assert train_desc.isdisjoint(heldout_desc)  # no leakage
    assert train_desc | heldout_desc == set(descriptions)
    assert len(heldout) == 1 and len(train) == 3


def test_zero_shot_benchmark_reports_gap_and_calibration_linkage():
    import math

    torch.manual_seed(0)
    encoder = HashingTextEncoder(dim=64)
    specs = make_gqa_specs([0, 6])
    descriptions = [
        "python code generation", "japanese translation", "math reasoning",
        "creative storytelling", "sql query writing", "image captioning",
    ]
    adapters = make_task_adapters(specs, encoder, descriptions)
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=64, hidden_dim=64, max_tensor_norm=8.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    report = run_zero_shot_benchmark(
        model, adapters, holdout_fraction=0.34, seed=0,
        train_steps=120, lr=1e-2, tolerance=0.05,
    )

    # The held-out descriptions never appeared in training.
    assert set(report["split"]["train_descriptions"]).isdisjoint(
        report["split"]["heldout_descriptions"]
    )
    # Core generalization measurement: seen tasks fit better than unseen ones.
    assert report["heldout"]["mean_loss"] >= report["train"]["mean_loss"] - 1e-6
    assert math.isfinite(report["generalization_gap"])
    # Calibration linkage is a correlation in [-1, 1] (or nan if degenerate).
    linkage = report["calibration_linkage"]
    assert math.isnan(linkage) or -1.0 <= linkage <= 1.0
    # Reliability suite is computed on the held-out split.
    assert "ece" in report["reliability"] and "aurc" in report["reliability"]
    assert "coverage_0.5" in report["selective_generalization"]
    # Records are tagged by split and cover every task.
    assert {r["split"] for r in report["records"]} == {"train", "heldout"}
    assert len(report["records"]) == len(descriptions)
    assert isinstance(report["honest"], bool)


def test_distribution_anchors_lower_confidence_on_ood():
    encoder = HashingTextEncoder(dim=64)
    specs = make_gqa_specs([0])
    model = TrueLoraGenerator(
        specs, adapter_bank=None, text_dim=64, hidden_dim=32,
        encoder=encoder, hyper_kind="conditioned",
    )
    seen = ["python code generation", "japanese translation"]
    model.set_distribution_anchors(seen)

    _, seen_report = model.generate(seen[0])
    _, ood_report = model.generate("quantum chromodynamics lattice gauge theory zzz")

    # A seen prompt sits exactly on an anchor -> ~zero novelty, full-strength confidence.
    assert seen_report["novelty"] < 1e-5
    # An unrelated prompt is far from every anchor -> higher novelty -> lower confidence.
    assert ood_report["novelty"] > seen_report["novelty"]
    assert (1.0 - ood_report["uncertainty"]) <= (1.0 - seen_report["uncertainty"])
    assert 0.0 <= ood_report["max_anchor_similarity"] <= 1.0 + 1e-6

    # Clearing the anchors restores the original no-novelty behavior.
    model.set_distribution_anchors(None)
    _, cleared = model.generate("quantum chromodynamics lattice gauge theory zzz")
    assert cleared["novelty"] == 0.0
    import math
    assert math.isnan(cleared["max_anchor_similarity"])


def test_anchors_do_not_change_blended_adapter_only_confidence():
    # Regression: the novelty signal must inform the reported confidence ONLY, never
    # the produced adapter. Folding it into the retrieval blend weight previously
    # shifted retrieval-vs-generation and could lower accuracy on slightly-novel prompts.
    encoder = HashingTextEncoder(dim=32)
    specs = make_gqa_specs([0, 6])
    bank, adapters = make_conditioned_bank(specs, encoder)
    model = TrueLoraGenerator(
        specs, bank, text_dim=32, hidden_dim=32, max_tensor_norm=4.0,
        encoder=encoder, hyper_kind="conditioned",
    )
    prompt = "an unrelated out-of-distribution prompt zzz"

    before, rep_before = model.generate(prompt, retrieval_k=2)
    model.set_distribution_anchors(adapters)
    after, rep_after = model.generate(prompt, retrieval_k=2)

    # The prompt is novel, so the reported uncertainty rises...
    assert rep_after["novelty"] > 0.0
    assert rep_after["uncertainty"] >= rep_before["uncertainty"] - 1e-9
    # ...but the blend weight and the produced adapter are byte-for-byte unchanged.
    assert abs(rep_before["generated_weight"] - rep_after["generated_weight"]) < 1e-9
    assert rep_after["blend_uncertainty"] == rep_before["blend_uncertainty"]
    assert set(before) == set(after)
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6)
