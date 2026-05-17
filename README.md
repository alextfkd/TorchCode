---
title: TorchCode
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🔥 TorchCode

**Crack the PyTorch interview.**

Practice implementing operators and architectures from scratch — the exact skills top ML teams test for.

*Like LeetCode, but for tensors. Self-hosted. Jupyter-based. Instant feedback.*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/duoan/TorchCode?style=social)](https://github.com/duoan/TorchCode)
[![GitHub Container Registry](https://img.shields.io/badge/ghcr.io-TorchCode-blue?style=flat-square&logo=github)](https://ghcr.io/duoan/torchcode)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-TorchCode-blue?style=flat-square)](https://huggingface.co/spaces/duoan/TorchCode)
![Problems](https://img.shields.io/badge/problems-56-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen?style=flat-square)

[![Star History Chart](https://api.star-history.com/svg?repos=duoan/TorchCode&type=Date)](https://star-history.com/#duoan/TorchCode&Date)

</div>

---

## 🎓 このフォークについて — W2-W5 練習トラック

![Practice](https://img.shields.io/badge/practice-29問-orange?style=flat-square) ![Lang](https://img.shields.io/badge/lang-日本語-blue?style=flat-square) ![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square) ![Base](https://img.shields.io/badge/base-duoan%2FTorchCode-lightgrey?style=flat-square)

**[duoan/TorchCode](https://github.com/duoan/TorchCode) のフォークをベースにした自作練習問題集。** PyTorch で CNN 学習の主要トピックを W2-W5 の 4 週分に整理した **29 問を全問日本語化**。本家の 40 問に加え、典型的な CNN 学習レシピ（pooling / augmentation / 評価指標 / 現代 optimizer 系）に直結する 16 問を **spec-driven 生成インフラ** と一緒に追加。

### 週次マッピング

| Week | テーマ | 問題数 | フォルダ | 1問目を Colab で開く |
|:----:|--------|:------:|----------|:-------------------:|
| **W2** | MLP / 基本分類 / 基礎 optimization | 8 | [`practice/W2/`](practice/W2/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W2/40_linear_regression.ipynb) |
| **W3** | 正則化 / 正規化 / advanced optimization | 9 | [`practice/W3/`](practice/W3/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W3/20_weight_init.ipynb) |
| **W4** | CNN 基礎 + 基本 transforms | 7 | [`practice/W4/`](practice/W4/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/22_conv2d.ipynb) |
| **W5** | CIFAR-10 advanced レシピ | 5 | [`practice/W5/`](practice/W5/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W5/51_label_smoothing_ce.ipynb) |

各週フォルダの `README.md` に **学習順** で問題リスト、各 `.ipynb` は実装 → `check("...")` で自動採点（5 テスト/問、計 145 テスト）。1 問目以外を開きたい時は各週の README から任意の `.ipynb` の Colab badge をクリック。

### 使い方

**Colab で（推奨・セットアップ不要）：**
`practice/W{n}/` 配下の `.ipynb` 右上の Colab badge をクリック → Run All → ✏️ セルに実装を書く → 最後の `check("...")` セルで採点。

**ローカル（Docker / JupyterLab）：**
```bash
make run                                # Docker 起動
# ブラウザで http://localhost:8888 → practice/W2/ に移動
```

### メンテ・拡張手順

変更タイプごとに対応する再生成スクリプトを走らせる：

| やりたいこと | 編集対象 | 再生成コマンド |
|------|----------|---------------|
| 新規問題を追加 | `problem_specs/{id}.py` を新規作成 | `python scripts/build.py --verify` |
| spec 問題（#41-56）の説明/テスト/解答を修正 | `problem_specs/{id}.py` | `python scripts/build.py --verify` |
| upstream 問題（#01-40）の intro 修正 | `templates/{file}.ipynb` の cell 0 | `python scripts/sync_solutions.py` |
| 週マッピング変更 | `scripts/week_mapping.py` | `python scripts/build_weeks.py` |
| 週フォルダ完全リセット（in-progress 破棄）| （上記） | `python scripts/build_weeks.py --reset` |
| 全 56 解答の健全性チェック | （なし） | `python scripts/verify_all_solutions.py` |

### ソースと生成物の対応

このリポジトリは spec-driven (16 問) と upstream-hand-written (40 問) のハイブリッド。編集禁止のファイルを直接いじると次の再生成で消える。

| ソース（編集 OK） | 生成物（編集禁止、再生成される） |
|------|------|
| `problem_specs/*.py` (16 問) | `torch_judge/tasks/{id}.py` + `templates/{4,5}*.ipynb` + `solutions/{4,5}*_solution.ipynb` |
| `templates/0*-40_*.ipynb` (40 既存) cell 0 | 対応する `solutions/*_solution.ipynb` の cell 0 (intro 部分のみ、code は upstream のまま) |
| `scripts/week_mapping.py` | `practice/W{n}/`, `practice/W{n}/README.md`, `practice/README.md` |

大きな変更後は `verify_all_solutions.py` で 56 解答が全 pass することを確認するのが安全。

詳細は下記 [Architecture](#%EF%B8%8F-architecture) / [Adding Your Own Problems](#-adding-your-own-problems) も参照。

### License

本家 [duoan/TorchCode](https://github.com/duoan/TorchCode) は **MIT License** で公開されている。本フォークもそれを継承し MIT License で公開する。フォーク独自の追加・改変部分も MIT で利用可能。詳細は [`LICENSE`](LICENSE) を参照。

---

以下は **本家 TorchCode の README**（英語、56 問全体の解説）。フォーク独自の追加問題は #41 以降。

---

## 🎯 Why TorchCode?

Top companies (Meta, Google DeepMind, OpenAI, etc.) expect ML engineers to implement core operations **from memory on a whiteboard**. Reading papers isn't enough — you need to write `softmax`, `LayerNorm`, `MultiHeadAttention`, and full Transformer blocks code.

TorchCode gives you a **structured practice environment** with:

| | Feature | |
|---|---|---|
| 🧩 | **40 curated problems** | The most frequently asked PyTorch interview topics |
| ⚖️ | **Automated judge** | Correctness checks, gradient verification, and timing |
| 🎨 | **Instant feedback** | Colored pass/fail per test case, just like competitive programming |
| 💡 | **Hints when stuck** | Nudges without full spoilers |
| 📖 | **Reference solutions** | Study optimal implementations after your attempt |
| 📊 | **Progress tracking** | What you've solved, best times, and attempt counts |
| 🔄 | **One-click reset** | Toolbar button to reset any notebook back to its blank template — practice the same problem as many times as you want |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) | **Open in Colab** | Every notebook has an "Open in Colab" badge + toolbar button — run problems in Google Colab with zero setup |

No cloud. No signup. No GPU needed. Just `make run` — or try it instantly on Hugging Face.

---

## 🚀 Quick Start

### Option 0 — Try it online (zero install)

**[Launch on Hugging Face Spaces](https://huggingface.co/spaces/duoan/TorchCode)** — opens a full JupyterLab environment in your browser. Nothing to install.

Or open any problem directly in Google Colab — every notebook has an [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/01_relu.ipynb) badge.

### Option 0b — Use the judge in Colab (pip)

In Google Colab, install the judge from this fork's git URL (so you get the full task set, including the additions in this fork that aren't on the upstream PyPI package):

```bash
!pip install -q --force-reinstall --no-deps git+https://github.com/alextfkd/TorchCode.git
```

(The notebook templates already have this install cell at the top — just **Run All** in Colab.)

Then in a notebook cell:

```python
from torch_judge import check, status, hint, reset_progress
status()           # list all problems and your progress
check("relu")      # run tests for the "relu" task
hint("relu")       # show a hint
```

### Option 1 — Pull the pre-built image (fastest)

```bash
docker run -p 8888:8888 -e PORT=8888 ghcr.io/duoan/torchcode:latest
```

If the registry image is unavailable for your platform, use Option 2 instead. This is the common path on Apple Silicon / `arm64`.

### Option 2 — Build locally

```bash
make run
```

`make run` will try the prebuilt image first and automatically fall back to a local build when needed.

Open **<http://localhost:8888>** — that's it. Works with both Docker and Podman (auto-detected).

---

## 📋 Problem Set

> **Frequency**: 🔥 = very likely in interviews, ⭐ = commonly asked, 💡 = emerging / differentiator

### 🧱 Fundamentals — "Implement X from scratch"

The bread and butter of ML coding interviews. You'll be asked to write these without `torch.nn`.

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 1 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/01_relu.ipynb" target="_blank">ReLU</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/01_relu.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `relu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Activation functions, element-wise ops |
| 2 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/02_softmax.ipynb" target="_blank">Softmax</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/02_softmax.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_softmax(x, dim)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Numerical stability, exp/log tricks |
| 16 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/16_cross_entropy.ipynb" target="_blank">Cross-Entropy Loss</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/16_cross_entropy.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cross_entropy_loss(logits, targets)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Log-softmax, logsumexp trick |
| 17 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/17_dropout.ipynb" target="_blank">Dropout</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/17_dropout.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyDropout` (nn.Module) | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Train/eval mode, inverted scaling |
| 18 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/18_embedding.ipynb" target="_blank">Embedding</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/18_embedding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyEmbedding` (nn.Module) | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Lookup table, `weight[indices]` |
| 19 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/19_gelu.ipynb" target="_blank">GELU</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/19_gelu.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_gelu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Gaussian error linear unit, `torch.erf` |
| 20 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/20_weight_init.ipynb" target="_blank">Kaiming Init</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/20_weight_init.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `kaiming_init(weight)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | `std = sqrt(2/fan_in)`, variance scaling |
| 21 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/21_gradient_clipping.ipynb" target="_blank">Gradient Clipping</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/21_gradient_clipping.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `clip_grad_norm(params, max_norm)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Norm-based clipping, direction preservation |
| 31 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb" target="_blank">Gradient Accumulation</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `accumulated_step(model, opt, ...)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 💡 | Micro-batching, loss scaling |
| 40 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/40_linear_regression.ipynb" target="_blank">Linear Regression</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/40_linear_regression.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `LinearRegression` (3 methods) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Normal equation, GD from scratch, nn.Linear |
| 3 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/03_linear.ipynb" target="_blank">Linear Layer</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/03_linear.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SimpleLinear` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | `y = xW^T + b`, Kaiming init, `nn.Parameter` |
| 4 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/04_layernorm.ipynb" target="_blank">LayerNorm</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/04_layernorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_layer_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Normalization, running stats, affine transform |
| 7 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/07_batchnorm.ipynb" target="_blank">BatchNorm</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/07_batchnorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_batch_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Batch vs layer statistics, train/eval behavior |
| 8 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/08_rmsnorm.ipynb" target="_blank">RMSNorm</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/08_rmsnorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `rms_norm(x, weight)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | LLaMA-style norm, simpler than LayerNorm |
| 15 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/15_mlp.ipynb" target="_blank">SwiGLU MLP</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/15_mlp.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SwiGLUMLP` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Gated FFN, `SiLU(gate) * up`, LLaMA/Mistral-style |
| 22 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/22_conv2d.ipynb" target="_blank">Conv2d</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/22_conv2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_conv2d(x, weight, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Convolution, unfold, stride/padding |
| 41 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/41_maxpool2d.ipynb" target="_blank">2D Max Pooling</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/41_maxpool2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_max_pool2d(x, k, stride, padding)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Unfold + amax, pad with `-inf` for negative inputs |
| 49 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/49_avg_pool2d.ipynb" target="_blank">2D Average Pooling</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/49_avg_pool2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_avg_pool2d(x, k, stride, padding)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Unfold + mean, `count_include_pad=True` default |
| 50 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/50_global_avg_pool.ipynb" target="_blank">Global Average Pooling</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/50_global_avg_pool.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `global_avg_pool(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Mean over (H, W), ResNet/MobileNet head replacing FC |
| 51 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/51_label_smoothing_ce.ipynb" target="_blank">Label Smoothing CE</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/51_label_smoothing_ce.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `label_smoothing_ce(logits, targets, ε)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Smoothed target dist, modern training recipe |
| 52 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/52_top_k_accuracy.ipynb" target="_blank">Top-k Accuracy</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/52_top_k_accuracy.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `top_k_accuracy(logits, targets, k)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | `topk` indices + `any`, ImageNet eval standard |
| 53 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/53_nll_loss.ipynb" target="_blank">NLL Loss</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/53_nll_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_nll_loss(log_probs, targets)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Advanced indexing, `CE = log_softmax + NLL` |

### 🧠 Attention Mechanisms — The heart of modern ML interviews

If you're interviewing for any role touching LLMs or Transformers, expect at least one of these.

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 23 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/23_cross_attention.ipynb" target="_blank">Cross-Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/23_cross_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MultiHeadCrossAttention` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Encoder-decoder, Q from decoder, K/V from encoder |
| 5 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/05_attention.ipynb" target="_blank">Scaled Dot-Product Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/05_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `scaled_dot_product_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | `softmax(QK^T/√d_k)V`, the foundation of everything |
| 6 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/06_multihead_attention.ipynb" target="_blank">Multi-Head Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/06_multihead_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MultiHeadAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Parallel heads, split/concat, projection matrices |
| 9 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/09_causal_attention.ipynb" target="_blank">Causal Self-Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/09_causal_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `causal_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Autoregressive masking with `-inf`, GPT-style |
| 10 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/10_gqa.ipynb" target="_blank">Grouped Query Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/10_gqa.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `GroupQueryAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | GQA (LLaMA 2), KV sharing across heads |
| 11 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/11_sliding_window.ipynb" target="_blank">Sliding Window Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/11_sliding_window.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sliding_window_attention(Q, K, V, w)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Mistral-style local attention, O(n·w) complexity |
| 12 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/12_linear_attention.ipynb" target="_blank">Linear Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/12_linear_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `linear_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Kernel trick, `φ(Q)(φ(K)^TV)`, O(n·d²) |
| 14 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/14_kv_cache.ipynb" target="_blank">KV Cache Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/14_kv_cache.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `KVCacheAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Incremental decoding, cache K/V, prefill vs decode |
| 24 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/24_rope.ipynb" target="_blank">RoPE</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/24_rope.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `apply_rope(q, k)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Rotary position embedding, relative position via rotation |
| 25 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/25_flash_attention.ipynb" target="_blank">Flash Attention</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/25_flash_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `flash_attention(Q, K, V, block_size)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Tiled attention, online softmax, memory-efficient |

### 🏗️ Architecture & Adaptation — Put it all together

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 26 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/26_lora.ipynb" target="_blank">LoRA</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/26_lora.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `LoRALinear` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Low-rank adaptation, frozen base + `BA` update |
| 27 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/27_vit_patch.ipynb" target="_blank">ViT Patch Embedding</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/27_vit_patch.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `PatchEmbedding` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 💡 | Image → patches → linear projection |
| 13 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/13_gpt2_block.ipynb" target="_blank">GPT-2 Block</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/13_gpt2_block.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `GPT2Block` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Pre-norm, causal MHA + MLP (4x, GELU), residual connections |
| 28 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/28_moe.ipynb" target="_blank">Mixture of Experts</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/28_moe.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MixtureOfExperts` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Mixtral-style, top-k routing, expert MLPs |

### 🎨 Data Augmentation — "Boost CIFAR-10 accuracy without changing the model"

The data side of the recipe. Together with normalization + cosine LR, these turn a baseline CNN into a competitive one.

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 42 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/42_normalize.ipynb" target="_blank">Per-Channel Normalize</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/42_normalize.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_normalize(x, mean, std)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Channel-wise `(x − μ) / σ`, broadcast to (C, 1, 1) |
| 43 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/43_random_hflip.ipynb" target="_blank">Random Horizontal Flip</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/43_random_hflip.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `random_horizontal_flip(x, p)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Per-sample Bernoulli mask + `torch.flip` |
| 44 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/44_random_crop.ipynb" target="_blank">Random Crop with Padding</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/44_random_crop.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `random_crop(x, size, padding)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | `F.pad` + per-sample random offset slice |
| 45 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/45_cutout.ipynb" target="_blank">Cutout / RandomErasing</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/45_cutout.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cutout(x, size)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Random rectangle zero-mask, DeVries 2017 |
| 46 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/46_mixup.ipynb" target="_blank">Mixup</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/46_mixup.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `mixup(x, y, α)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | `Beta(α, α)`, 4-tuple `(x_mix, y_a, y_b, lam)` interface |
| 47 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/47_cutmix.ipynb" target="_blank">CutMix</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/47_cutmix.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cutmix(x, y, α)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Area-based λ **recomputed** after boundary clipping |
| 48 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/48_tta_hflip.ipynb" target="_blank">TTA (Horizontal Flip)</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/48_tta_hflip.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `tta_hflip(model, x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 💡 | Probability-space averaging, free 0.3–1% bump |

### ⚙️ Training & Optimization

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 29 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/29_adam.ipynb" target="_blank">Adam Optimizer</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/29_adam.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyAdam` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Momentum + RMSProp, bias correction |
| 30 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/30_cosine_lr.ipynb" target="_blank">Cosine LR Scheduler</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/30_cosine_lr.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cosine_lr_schedule(step, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Linear warmup + cosine annealing |
| 54 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/54_sgd_momentum.ipynb" target="_blank">SGD with Momentum</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/54_sgd_momentum.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MySGDMomentum` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | `v = μ·v + g` (PyTorch convention — no `(1−μ)` factor) |
| 55 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/55_weight_decay.ipynb" target="_blank">Weight Decay (L2)</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/55_weight_decay.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `apply_weight_decay(params, wd)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | `g += wd·p`, compare with decoupled WD (#56) |
| 56 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/56_adamw.ipynb" target="_blank">AdamW</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/56_adamw.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyAdamW` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Decoupled WD: `p *= (1 − lr·λ)`, Transformer default |

### 🎯 Inference & Decoding

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 32 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/32_topk_sampling.ipynb" target="_blank">Top-k / Top-p Sampling</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/32_topk_sampling.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sample_top_k_top_p(logits, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Nucleus sampling, temperature scaling |
| 33 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/33_beam_search.ipynb" target="_blank">Beam Search</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/33_beam_search.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `beam_search(log_prob_fn, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Hypothesis expansion, pruning, eos handling |
| 34 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/34_speculative_decoding.ipynb" target="_blank">Speculative Decoding</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/34_speculative_decoding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `speculative_decode(target, draft, ...)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Accept/reject, draft model acceleration |

### 🔬 Advanced — Differentiators

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 35 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/35_bpe.ipynb" target="_blank">BPE Tokenizer</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/35_bpe.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SimpleBPE` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Byte-pair encoding, merge rules, subword splits |
| 36 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/36_int8_quantization.ipynb" target="_blank">INT8 Quantization</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/36_int8_quantization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `Int8Linear` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Per-channel quantize, scale/zero-point, buffer vs param |
| 37 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/37_dpo_loss.ipynb" target="_blank">DPO Loss</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/37_dpo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `dpo_loss(chosen, rejected, ...)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Direct preference optimization, alignment training |
| 38 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/38_grpo_loss.ipynb" target="_blank">GRPO Loss</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/38_grpo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `grpo_loss(logps, rewards, group_ids, eps)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Group relative policy optimization, RLAIF, within-group normalized advantages |
| 39 | <a href="https://github.com/alextfkd/TorchCode/blob/master/templates/39_ppo_loss.ipynb" target="_blank">PPO Loss</a> <a href="https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/templates/39_ppo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `ppo_loss(new_logps, old_logps, advantages, clip_ratio)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | PPO clipped surrogate loss, policy gradient, trust region |

---

## ⚙️ How It Works

Each problem has **two** notebooks:

| File | Purpose |
|------|---------|
| `01_relu.ipynb` | ✏️ Blank template — write your code here |
| `01_relu_solution.ipynb` | 📖 Reference solution — check when stuck |

### Workflow

```text
1. Open a blank notebook           →  Read the problem description
2. Implement your solution         →  Use only basic PyTorch ops
3. Debug freely                    →  print(x.shape), check gradients, etc.
4. Run the judge cell              →  check("relu")
5. See instant colored feedback    →  ✅ pass / ❌ fail per test case
6. Stuck? Get a nudge              →  hint("relu")
7. Review the reference solution   →  01_relu_solution.ipynb
8. Click 🔄 Reset in the toolbar  →  Blank slate — practice again!
```

### In-Notebook API

```python
from torch_judge import check, hint, status

check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint without full spoiler
status()                    # Progress dashboard — solved / attempted / todo
```

---

## 📅 Suggested Study Plan

> **Total: ~12–16 hours spread across 3–4 weeks. Perfect for interview prep on a deadline.**

| Week | Focus | Problems | Time |
|:----:|-------|----------|:----:|
| **1** | 🧱 Foundations | ReLU → Softmax → CE Loss → Dropout → Embedding → GELU → Linear → LayerNorm → BatchNorm → RMSNorm → SwiGLU MLP → Conv2d | 2–3 hrs |
| **2** | 🧠 Attention Deep Dive | SDPA → MHA → Cross-Attn → Causal → GQA → KV Cache → Sliding Window → RoPE → Linear Attn → Flash Attn | 3–4 hrs |
| **3** | 🏗️ Architecture + Training | GPT-2 Block → LoRA → MoE → ViT Patch → Adam → Cosine LR → Grad Clip → Grad Accumulation → Kaiming Init | 3–4 hrs |
| **4** | 🎯 Inference + Advanced | Top-k/p Sampling → Beam Search → Speculative Decoding → BPE → INT8 Quant → DPO Loss → GRPO Loss → PPO Loss + speed run | 3–4 hrs |

---

## 🏛️ Architecture

```text
┌──────────────────────────────────────────┐
│           Docker / Podman Container      │
│                                          │
│  JupyterLab (:8888)                      │
│    ├── templates/  (reset on each run)   │
│    ├── solutions/  (reference impl)      │
│    ├── torch_judge/ (auto-grading)       │
│    ├── torchcode-labext (JLab plugin)    │
│    │     🔄 Reset — restore template     │
│    │     🔗 Colab — open in Colab        │
│    └── PyTorch (CPU), NumPy              │
│                                          │
│  Judge checks:                           │
│    ✓ Output correctness (allclose)       │
│    ✓ Gradient flow (autograd)            │
│    ✓ Shape consistency                   │
│    ✓ Edge cases & numerical stability    │
└──────────────────────────────────────────┘
```

Single container. Single port. No database. No frontend framework. No GPU.

## 🛠️ Commands

```bash
make run    # Build & start (http://localhost:8888)
make stop   # Stop the container
make clean  # Stop + remove volumes + reset all progress
```

## 🧩 Adding Your Own Problems

TorchCode uses auto-discovery — just drop a new file in `torch_judge/tasks/`:

```python
TASK = {
    "id": "my_task",
    "title": "My Custom Problem",
    "difficulty": "medium",
    "function_name": "my_function",
    "hint": "Think about broadcasting...",
    "tests": [ ... ],
}
```

No registration needed. The judge picks it up automatically.

---

## 📦 Publishing `torch-judge` to PyPI (maintainers)

The judge is published as a separate package so Colab/users can `pip install torch-judge` without cloning the repo.

### Automatic (GitHub Action)

Pushing to `master` after changing the package version triggers [`.github/workflows/pypi-publish.yml`](.github/workflows/pypi-publish.yml), which builds and uploads to PyPI. No git tag is required.

1. **Bump version** in `torch_judge/_version.py` (e.g. `__version__ = "0.1.1"`).
2. **Configure PyPI Trusted Publisher** (one-time):
   - PyPI → Your project **torch-judge** → **Publishing** → **Add a new pending publisher**
   - Owner: `duoan`, Repository: `TorchCode`, Workflow: `pypi-publish.yml`, Environment: (leave empty)
   - Run the workflow once (push a version bump to `master` or **Actions → Publish torch-judge to PyPI → Run workflow**); PyPI will then link the publisher.
3. **Release**: commit the version bump and `git push origin master`.

Alternatively, use an API token: add repository secret `PYPI_API_TOKEN` (value = `pypi-...` from PyPI) and set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` from that secret in the workflow if you prefer not to use Trusted Publishing.

### Manual

```bash
pip install build twine
python -m build
twine upload dist/*
```

Version is in `torch_judge/_version.py`; bump it before each release.

---

## ❓ FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>
<br>
No. Everything runs on CPU. The problems test correctness and understanding, not throughput.
</details>

<details>
<summary><b>Can I keep my solutions between runs?</b></summary>
<br>
Blank templates reset on every <code>make run</code> so you practice from scratch. Save your work under a different filename if you want to keep it. You can also click the <b>🔄 Reset</b> button in the notebook toolbar at any time to restore the blank template without restarting.
</details>

<details>
<summary><b>Can I use Google Colab instead?</b></summary>
<br>
Yes! Every notebook has an <b>Open in Colab</b> badge at the top. Click it to open the problem directly in Google Colab — no Docker or local setup needed. You can also use the <b>Colab</b> toolbar button inside JupyterLab.
</details>

<details>
<summary><b>How are solutions graded?</b></summary>
<br>
The judge runs your function against multiple test cases using <code>torch.allclose</code> for numerical correctness, verifies gradients flow properly via autograd, and checks edge cases specific to each operation.
</details>

<details>
<summary><b>Who is this for?</b></summary>
<br>
Anyone preparing for ML/AI engineering interviews at top tech companies, or anyone who wants to deeply understand how PyTorch operations work under the hood.
</details>

---

## 🤝 Contributors

Thanks to everyone who has contributed to TorchCode.

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/duoan">
                    <img src="https://avatars.githubusercontent.com/u/2378740?v=4" width="100;" alt="duoan"/>
                    <br />
                    <sub><b>duoan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Ando233">
                    <img src="https://avatars.githubusercontent.com/u/74404658?v=4" width="100;" alt="Ando233"/>
                    <br />
                    <sub><b>Ando233</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ThierryHJ">
                    <img src="https://avatars.githubusercontent.com/u/51846529?v=4" width="100;" alt="ThierryHJ"/>
                    <br />
                    <sub><b>ThierryHJ</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->

Auto-generated from the [GitHub contributors graph](https://github.com/duoan/TorchCode/graphs/contributors) with avatars and GitHub usernames.

---

<div align="center">

**Built for engineers who want to deeply understand what they build.**

If this helped your interview prep, consider giving it a ⭐

---

### ☕ Buy Me a Coffee

<a href="https://buymeacoffee.com/duoan" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

<img src="./bmc_qr.png" alt="BMC QR Code" width="150" height="150">

*Scan to support*

</div>
