"""A7 — TTA (Test-Time Augmentation, horizontal-flip averaging)."""

PROBLEM = {
    "id": "tta_hflip",
    "number": 48,
    "title": "TTA (Horizontal Flip Averaging)",
    "difficulty": "Easy",
    "fn_name": "tta_hflip",

    "memo": (
        "推論時の augmentation。test 画像と flip 版で 2 回 forward、softmax 確率を平均。"
    ),
    "details": (
        "0.3–1% タダで accuracy が上がる、Kaggle の鉄板。\n\nProbability-space (softmax 後) で平均するのが標準 — logit-space (softmax 前) より自然な ensemble に。5-crop (center + 4 corner) との組み合わせも可能、推論 cost は 2x-10x になるが多くの場合元が取れる。"
    ),
    "intro_md": (
        "**Test-Time Augmentation** with horizontal flip を実装する。model を `x` と "
        "`x.flip(-1)` の両方で走らせ、softmax 確率を **平均**（logit じゃない）。"
        "image classification で 0.3〜1.0% タダで上がる。\n\n"
        "### 標準作法 — probability-space averaging\n"
        "softmax の **後**で平均する（logit を平均してから softmax するんじゃない）。"
        "各 TTA 分岐を独立 classifier と見て predictions を ensemble する流儀。"
        "Kaggle / 論文の慣例。\n"
    ),

    "signature": (
        "def tta_hflip(model, x):\n"
        "    # model: nn.Module taking (B, C, H, W) → (B, num_classes) logits\n"
        "    # x: (B, C, H, W) input\n"
        "    # Returns: (B, num_classes) averaged probabilities"
    ),

    "rules": [
        "softmax を **先に**適用してから平均（probability-space ensemble、logit-space ではない）",
        "`torch.no_grad()` で wrap — 推論なので autograd tape 不要",
        "*確率* (各行 sum=1) を return、logit ではない",
        "model state を変更しない（caller が `model.eval()` 済みと仮定）",
    ],

    "imports": "import torch\nimport torch.nn as nn",

    "template_body": (
        "def tta_hflip(model, x):\n"
        "    pass  # softmax(model(x)) + softmax(model(x.flip(-1))) / 2、all under no_grad"
    ),

    "solution_body": (
        "def tta_hflip(model, x):\n"
        "    with torch.no_grad():\n"
        "        p1 = torch.softmax(model(x), dim=-1)\n"
        "        p2 = torch.softmax(model(x.flip(-1)), dim=-1)\n"
        "    return (p1 + p2) / 2"
    ),

    "demo_code": (
        "import torch\n"
        "import torch.nn as nn\n"
        "torch.manual_seed(0)\n"
        "\n"
        "model = nn.Sequential(\n"
        "    nn.Conv2d(3, 8, 3, padding=1),\n"
        "    nn.AdaptiveAvgPool2d(1),\n"
        "    nn.Flatten(),\n"
        "    nn.Linear(8, 10),\n"
        ")\n"
        "model.eval()\n"
        "x = torch.randn(2, 3, 16, 16)\n"
        "p = tta_hflip(model, x)\n"
        "print('Output shape:', p.shape, '| row sums:', p.sum(dim=-1).tolist())"
    ),

    "hint": (
        "2 つの forward pass: `model(x)` と `model(x.flip(-1))`。それぞれを `softmax(dim=-1)` で "
        "確率に変換、平均。推論なので `torch.no_grad()` で wrap。"
    ),

    "tests": [
        {
            "name": "Output shape (B, num_classes)",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "torch.manual_seed(0)\n"
                "model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\n"
                "model.eval()\n"
                "x = torch.randn(4, 3, 16, 16)\n"
                "out = {fn}(model, x)\n"
                "assert out.shape == (4, 10), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Output is a probability distribution (rows sum to 1, non-negative)",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "torch.manual_seed(0)\n"
                "model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\n"
                "model.eval()\n"
                "x = torch.randn(3, 3, 8, 8)\n"
                "out = {fn}(model, x)\n"
                "sums = out.sum(dim=-1)\n"
                "assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f'Row sums: {sums.tolist()}'\n"
                "assert (out >= 0).all(), 'Probabilities must be non-negative'\n"
            ),
        },
        {
            "name": "Equals averaged softmax of model(x) and model(flip(x))",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\n"
                "model.eval()\n"
                "x = torch.randn(2, 3, 8, 8)\n"
                "out = {fn}(model, x)\n"
                "with torch.no_grad():\n"
                "    expected = (F.softmax(model(x), dim=-1) + F.softmax(model(x.flip(-1)), dim=-1)) / 2\n"
                "assert torch.allclose(out, expected, atol=1e-6), f'Max diff: {(out-expected).abs().max():.6f}'\n"
            ),
        },
        {
            "name": "Invariant to flipping the input (tta(x) == tta(flip(x)))",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "torch.manual_seed(0)\n"
                "model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\n"
                "model.eval()\n"
                "x = torch.randn(2, 3, 8, 8)\n"
                "a = {fn}(model, x)\n"
                "b = {fn}(model, x.flip(-1))\n"
                "assert torch.allclose(a, b, atol=1e-6), 'TTA must be invariant to flipping the input'\n"
            ),
        },
        {
            "name": "Differs from single-forward for asymmetric input",
            "code": (
                "\nimport torch\n"
                "import torch.nn as nn\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\n"
                "model.eval()\n"
                "x = torch.randn(2, 3, 8, 8)\n"
                "out_tta = {fn}(model, x)\n"
                "with torch.no_grad():\n"
                "    out_single = F.softmax(model(x), dim=-1)\n"
                "assert not torch.allclose(out_tta, out_single, atol=1e-4), 'TTA result should differ from single forward'\n"
            ),
        },
    ],
}
