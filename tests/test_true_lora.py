import torch

from true_lora.adapter import AdapterBank, AdapterSpec, LoraTensorSpec
from true_lora.generator import TrueLoraGenerator
from true_lora.text import HashingTextEncoder


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
