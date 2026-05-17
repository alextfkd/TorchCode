"""B2 — Global Average Pooling (ResNet/MobileNet head の FC 代替)."""

PROBLEM = {
    "id": "global_avg_pool",
    "number": 50,
    "title": "Global Average Pooling",
    "difficulty": "Easy",
    "fn_name": "global_avg_pool",

    "memo": (
        "現代 CNN classifier の標準 head。`(B, C, H, W) → (B, C)` で空間を畳む、FC layer の代替。"
    ),
    "details": (
        "ResNet, MobileNet, EfficientNet など全部これ。NIN (2013) が起源。\n\n利点: param が大幅減 (`H×W×C` の FC weight が消える)、平行移動 invariance が built-in、入力サイズに依存しない。"
    ),
    "intro_md": (
        "**Global Average Pooling** を実装する。feature map の空間次元を畳んで channel あたり "
        "1 値にする。ResNet / MobileNet / EfficientNet など modern CNN classifier の標準 "
        "\"head\" で、大きな FC layer を置き換える。\n\n"
        "$$\\text{out}[b, c] = \\frac{1}{H \\cdot W} \\sum_{h, w} x[b, c, h, w]$$\n\n"
        "### なぜ GAP\n"
        "- 各 channel に global concept を持たせる\n"
        "- FC head より大幅にパラメータ削減\n"
        "- 平行移動 invariance が built-in\n"
    ),

    "signature": (
        "def global_avg_pool(x):\n"
        "    # x: (B, C, H, W)\n"
        "    # Returns: (B, C) — flattened, NOT (B, C, 1, 1)"
    ),

    "rules": [
        "`F.adaptive_avg_pool2d`, `nn.AdaptiveAvgPool2d`, `nn.AvgPool2d` は **使わない**",
        "Output shape は `(B, C)` — flatten 済みで、直接 `nn.Linear` に流せる形（`(B, C, 1, 1)` じゃない）",
        "任意の `H`, `W` で動く（正方形である必要なし）",
        "勾配が空間次元全体に均等に流れる",
    ],

    "imports": "import torch",

    "template_body": (
        "def global_avg_pool(x):\n"
        "    pass  # (H, W) dim で mean — 1行"
    ),

    "solution_body": (
        "def global_avg_pool(x):\n"
        "    return x.mean(dim=(-2, -1))"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(2, 16, 8, 8)\n"
        "out = global_avg_pool(x)\n"
        "print('In:', x.shape, '→ Out:', out.shape)"
    ),

    "hint": (
        "1行: `x.mean(dim=(-2, -1))`。output が `(B, C)` であって `(B, C, 1, 1)` じゃないこと — "
        "`nn.Linear` に直接流すため。"
    ),

    "tests": [
        {
            "name": "Output shape (B, C), not (B, C, 1, 1)",
            "code": (
                "\nimport torch\n"
                "x = torch.randn(4, 16, 7, 7)\n"
                "out = {fn}(x)\n"
                "assert out.shape == (4, 16), f'Shape: {out.shape} — must be (B, C) not (B, C, 1, 1)'\n"
            ),
        },
        {
            "name": "Matches mean over (H, W)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(2, 8, 6, 6)\n"
                "out = {fn}(x)\n"
                "ref = x.mean(dim=(-2, -1))\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max()}'\n"
            ),
        },
        {
            "name": "Matches F.adaptive_avg_pool2d(x, 1).flatten(1)",
            "code": (
                "\nimport torch\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(3, 12, 5, 5)\n"
                "out = {fn}(x)\n"
                "ref = F.adaptive_avg_pool2d(x, 1).flatten(1)\n"
                "assert torch.allclose(out, ref, atol=1e-6)\n"
            ),
        },
        {
            "name": "Works for non-square (H ≠ W)",
            "code": (
                "\nimport torch\n"
                "x = torch.randn(2, 4, 6, 10)\n"
                "out = {fn}(x)\n"
                "assert out.shape == (2, 4), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Gradient distributes as 1/(H*W) per input",
            "code": (
                "\nimport torch\n"
                "x = torch.randn(1, 3, 4, 4, requires_grad=True)\n"
                "out = {fn}(x)\n"
                "out.sum().backward()\n"
                "expected = torch.full_like(x, 1.0 / (4 * 4))\n"
                "assert torch.allclose(x.grad, expected, atol=1e-5), 'Gradient should be 1/(H*W) per input'\n"
            ),
        },
    ],
}
