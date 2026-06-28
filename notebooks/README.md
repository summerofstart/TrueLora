# 🍄 True-LoRA Tutorial: matutake にコーディング能力を付与

## 概要

このNotebookは、[True-LoRA](https://github.com/MARVserver/TrueLora)を使って、[summerMC/matutake](https://huggingface.co/summerMC/matutake) (2B params, Qwen2-based) にPythonコーディング能力を付与する方法を説明します。

## 対象

- **モデル:** summerMC/matutake
- **ハードウェア:** Colab Free Tier (T4 GPU, ~15GB RAM)
- **所要時間:** 約15-20分

## フロー

1. **環境セットアップ** — True-LoRAのインストール
2. **モデルの準備** — matutakeのダウンロードとアーキテクチャ解析
3. **コーディングアダプタバンクの構築** — コーディングタスク用のアダプタを作成
4. **True-LoRAトレーニング** — 検索型LoRA生成器を学習
5. **LoRAアダプタの生成** — 「Python code generation」用のLoRAを生成
6. **マージ** — アダプタをベースモデルに統合
7. **テスト** — コーディング能力の検証
8. **HuggingFace Hubにアップロード** (オプション)

## 主なパラメータ

| パラメータ | 値 | 説明 |
| --- | --- | --- |
| LoRA Rank | 8 | 低ランク近似のランク |
| LoRA Alpha | 16.0 | スケーリング係数 |
| Target Layers | 7層 (0, 4, 8, 12, 16, 20, 24) | 対象レイヤー |
| LoRA Targets | 7モジュール/層 | q, k, v, o, gate, up, down |
| Total LoRA Specs | 49 | 全LoRA仕様数 |
| Adapter Bank Size | 29 | アダプタバンクサイズ |
| Training Steps | 200 | トレーニングステップ数 |

## 使用方法

### Colabで開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MARVserver/TrueLora/blob/main/notebooks/matutake_coding_lora_tutorial.ipynb)

### ローカルで実行

```bash
pip install git+https://github.com/MARVserver/TrueLora.git
pip install transformers accelerate
jupyter notebook notebooks/matutake_coding_lora_tutorial.ipynb
```

## 出力ファイル

| ファイル | 説明 |
| --- | --- |
| `truelora_work/adapters/coding_adapters.pt` | コーディングアダプタバンク |
| `truelora_work/generator_checkpoint.pt` | トレーニング済み生成器 |
| `truelora_work/output/coding_lora.pt` | 生成されたLoRAアダプタ |
| `truelora_work/output/matutake_coding_merged/` | マージ済みモデル |
| `truelora_work/output/generation_report.json` | 生成レポート |

## 使用例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "summerMC/matutake-coding-lora",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("summerMC/matutake-coding-lora")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## カスタマイズ

### 別のターゲットタスク

```python
# 日本語→英語翻訳
translation_prompt = "japanese to english translation"

# 要約
summarization_prompt = "document summarization"

# 数学
math_prompt = "mathematical problem solving"
```

### パラメータ調整

- **`LORA_RANK`:** 4-16（メモリと表現力のトレードオフ）
- **`TARGET_LAYERS`:** より多くのレイヤーを対象にすると表現力が上がるが、メモリが増加
- **`train_steps`:** 100-500（Colab Free Tierでは200程度が現実的）
- **`retrieval_k`:** 4-12（検索するアダプタ数）

## 注意事項

- Colab Free Tierではメモリ制限があるため、大量のトレーニングステップは現実的ではありません
- モデルの品質を向上させるには、より高品質なアダプタバンクが必要です
- 実際のコーディング能力は、プロンプトの工夫によっても左右されます

## ライセンス

- **True-LoRA:** MIT License
- **matutake:** ベースモデルおよびデータセットのライセンスに準拠
