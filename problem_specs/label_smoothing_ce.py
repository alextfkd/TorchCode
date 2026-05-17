"""B3 — Label Smoothing CE (Szegedy 2016, W5 advanced レシピの中核)."""

PROBLEM = {
    "id": "label_smoothing_ce",
    "number": 51,
    "title": "Label Smoothing Cross-Entropy",
    "difficulty": "Easy",
    "fn_name": "label_smoothing_ce",

    "intro_md": (
        "**Label Smoothing Cross-Entropy** (Szegedy et al. 2016) を実装する。one-hot target を "
        "soften することで model が無限に自信を持つのを防ぐ。modern training recipe の定番。\n\n"
        "### Smoothed target distribution\n"
        "$$q'[c] = \\begin{cases} 1 - \\varepsilon + \\varepsilon/K & c = y \\\\ \\varepsilon/K & c \\neq y \\end{cases}$$\n\n"
        "この target に対する標準 cross-entropy：\n"
        "$$L = -\\sum_c q'[c] \\log p[c], \\quad p = \\text{softmax}(\\text{logits})$$\n\n"
        "### なぜ効く\n"
        "- over-confident prediction を防ぐ（logit gap が bound される）\n"
        "- 正則化として効く — ImageNet で ~0.3-0.5% 上がる定番\n"
        "- calibration error が下がる\n"
    ),

    "signature": (
        "def label_smoothing_ce(logits, targets, smoothing=0.1):\n"
        "    # logits: (B, K) raw scores\n"
        "    # targets: (B,) class indices (long)\n"
        "    # smoothing: ε in [0, 1)\n"
        "    # Returns: scalar loss (mean over batch)"
    ),

    "rules": [
        "`nn.CrossEntropyLoss(label_smoothing=...)` や `F.cross_entropy(label_smoothing=...)` は **使わない**",
        "数値安定性のため `F.log_softmax` を使う（softmax → log ではない）",
        "`smoothing=0` の時、標準 cross-entropy と完全一致すること",
        "batch の **mean** を return（scalar）",
        "Target dist: 正解クラスは `(1-ε) + ε/K`、それ以外は `ε/K`",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def label_smoothing_ce(logits, targets, smoothing=0.1):\n"
        "    pass  # log_softmax、smoothed target dist 構築、-sum(q * log_p) → mean"
    ),

    "solution_body": (
        "def label_smoothing_ce(logits, targets, smoothing=0.1):\n"
        "    K = logits.size(-1)\n"
        "    log_probs = F.log_softmax(logits, dim=-1)\n"
        "    one_hot = F.one_hot(targets, K).to(log_probs.dtype)\n"
        "    # 正解クラスは (1-ε) + ε/K、それ以外は ε/K\n"
        "    target_dist = one_hot * (1 - smoothing) + smoothing / K\n"
        "    loss = -(target_dist * log_probs).sum(dim=-1).mean()\n"
        "    return loss"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "logits = torch.randn(4, 10)\n"
        "targets = torch.tensor([3, 1, 7, 0])\n"
        "print('CE (ε=0):  ', label_smoothing_ce(logits, targets, smoothing=0.0).item())\n"
        "print('CE (ε=0.1):', label_smoothing_ce(logits, targets, smoothing=0.1).item())"
    ),

    "hint": (
        "`F.log_softmax(logits, dim=-1)`。target dist は "
        "`one_hot * (1 - smoothing) + smoothing / K` で構築 — 正解クラスは `(1-ε) + ε/K`、"
        "それ以外は `ε/K`。Loss = `-(target_dist * log_probs).sum(-1).mean()`。"
    ),

    "tests": [
        {
            "name": "Returns scalar",
            "code": (
                "\nimport torch\n"
                "logits = torch.randn(4, 10)\n"
                "targets = torch.randint(0, 10, (4,))\n"
                "loss = {fn}(logits, targets, smoothing=0.1)\n"
                "assert loss.dim() == 0, f'Expected scalar, got shape {loss.shape}'\n"
            ),
        },
        {
            "name": "smoothing=0 equals standard CrossEntropy",
            "code": (
                "\nimport torch\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "logits = torch.randn(8, 10)\n"
                "targets = torch.randint(0, 10, (8,))\n"
                "out = {fn}(logits, targets, smoothing=0.0)\n"
                "ref = F.cross_entropy(logits, targets)\n"
                "assert torch.allclose(out, ref, atol=1e-5), f'ε=0: {out.item():.6f} vs CE {ref.item():.6f}'\n"
            ),
        },
        {
            "name": "Matches nn.CrossEntropyLoss(label_smoothing=0.1)",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "torch.manual_seed(0)\n"
                "logits = torch.randn(16, 5)\n"
                "targets = torch.randint(0, 5, (16,))\n"
                "out = {fn}(logits, targets, smoothing=0.1)\n"
                "ref = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)\n"
                "assert torch.allclose(out, ref, atol=1e-5), f'Got {out.item():.6f}, expected {ref.item():.6f}'\n"
            ),
        },
        {
            "name": "Gradient flows back to logits",
            "code": (
                "\nimport torch\n"
                "logits = torch.randn(4, 10, requires_grad=True)\n"
                "targets = torch.randint(0, 10, (4,))\n"
                "loss = {fn}(logits, targets, smoothing=0.1)\n"
                "loss.backward()\n"
                "assert logits.grad is not None and logits.grad.shape == logits.shape\n"
            ),
        },
        {
            "name": "Smoothing increases loss on confident-correct predictions",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "# Strongly correct predictions: smoothing should INCREASE the loss\n"
                "logits = torch.zeros(4, 5)\n"
                "logits[torch.arange(4), torch.tensor([0, 1, 2, 3])] = 10.0\n"
                "targets = torch.tensor([0, 1, 2, 3])\n"
                "loss_0 = {fn}(logits, targets, smoothing=0.0)\n"
                "loss_1 = {fn}(logits, targets, smoothing=0.2)\n"
                "assert loss_1 > loss_0, f'Smoothing should increase loss on confident-correct: {loss_0.item()} vs {loss_1.item()}'\n"
            ),
        },
    ],
}
