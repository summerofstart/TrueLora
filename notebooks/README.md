# 🎛️ True-LoRA Playground — 好きなプロンプトで LoRA を作るデモ

## 概要

この Notebook は [True-LoRA](https://github.com/MARVserver/TrueLora) の **bankless（検索DBなし・純 Text-to-LoRA）** な使い方を体験するプレイグラウンドです。少数の `(説明文 → LoRA)` ペアで条件付きハイパーネットを数秒で学習し、**任意のプロンプト**から LoRA アダプタをその場で生成します。生成したアダプタは（オプションで）[summerMC/matutake](https://huggingface.co/summerMC/matutake) (2B params, Qwen2-based) にマージして実際に動かせます。

## 対象

- **モデル:** summerMC/matutake（マージは任意。生成・学習だけなら不要）
- **ハードウェア:** Colab Free Tier (T4 GPU) — 生成・学習・高度なデモは CPU でも数秒
- **所要時間:** 生成のみ数分／matutake マージ込みで約15-20分

## フロー

1. **セットアップ** — True-LoRA のインストール
2. **学習データを用意** — `(説明文, LoRA値)` のペアを作成（Adapter Bank なし）
3. **ハイパーネットを学習** — 条件付き生成器を一度だけ学習（数秒）
4. **好きなプロンプトで LoRA を生成** — フォームに入力して即生成・保存
5. **(高度・オプション)** end-to-end SFT のミニデモ（自己完結・CPU）
6. **(高度・オプション)** ゼロショット汎化ベンチ＋較正連動（自己完結・CPU）
7. **(オプション)** 生成した LoRA を matutake にマージして生成を試す

## 主なパラメータ（Step 2-3 の既定値）

| パラメータ | 値 | 説明 |
| --- | --- | --- |
| LoRA Rank | 4 | 低ランク近似のランク |
| LoRA Alpha | 8.0 | スケーリング係数 |
| Target Layers | 4層 (0, 6, 12, 18) | 対象レイヤー |
| LoRA Targets | q_proj, v_proj | 対象モジュール（GQA なので出力次元が異なる） |
| Total LoRA Specs | 8 | 全 LoRA 仕様数 |
| 学習ペア数 | 15 | `(説明文 → LoRA)` の学習データ（Adapter Bank は不使用） |
| Hidden Dim | 256 | ハイパーネットの隠れ層 |
| Training Steps | 200 | 学習ステップ数 |
| Hyper Kind | `conditioned` | 共有 trunk ＋ (タスク, 層, モジュール) 条件付け |

> ℹ️ これは **bankless** な構成です。`adapter_bank=None` なので検索（retrieval）は行わず、ハイパーネットだけがプロンプトから LoRA を生成します。`retrieval_k` は API 互換のため受け付けますが、この構成では検索結果は空になります。

## 使用方法

### Colab で開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MARVserver/TrueLora/blob/main/notebooks/matutake_coding_lora_tutorial.ipynb)

### ローカルで実行

```bash
pip install git+https://github.com/MARVserver/TrueLora.git
pip install transformers accelerate     # matutake マージを試す場合
jupyter notebook notebooks/matutake_coding_lora_tutorial.ipynb
```

## 出力ファイル（`./output/` 以下）

| ファイル | 説明 |
| --- | --- |
| `output/lora_<prompt>.pt` | 生成された LoRA アダプタ（PEFT 互換、レポート埋め込み） |
| `output/report_<prompt>.json` | 生成レポート（確信度・不確実性など） |
| `output/matutake_merged/` | LoRA をマージ済みの matutake（Step 7 を実行した場合） |

## オフライン時の注意

- **意味エンコーダ:** `SemanticTextEncoder` は多言語 SBERT を使いますが、ライブラリ／重みが取得できない環境では決定論的な **hashing フォールバック**に自動で切り替わります。
- hashing フォールバックのトークナイザは英数字ベースのため、**日本語のみのプロンプト**はうまく埋め込めません（Step 4 の日本語例は、SBERT が読み込めるオンライン環境で意図通りに動きます）。

## 仕組み（要点）

- **条件付きハイパーネット:** 共有 trunk を `(タスク, 層, モジュール)` で条件付けし、モジュール種別ごとのヘッドが LoRA を1ブロックずつデコードします。パラメータ数はモデルの深さではなく**モジュール種別数**に比例するため、深いモデルでもコンパクトです。
- **確信度と信頼性:** 生成と同時に不確実性を報告します。ゼロショット汎化ベンチ（Step 6）では、確信度が未知タスクの品質を当てられるか（calibration linkage）まで測ります。
- **PEFT 互換:** 生成物は標準の HuggingFace PEFT 形式で保存され、そのままマージできます。

## ライセンス

- **True-LoRA:** MIT License
- **matutake:** ベースモデルおよびデータセットのライセンスに準拠
