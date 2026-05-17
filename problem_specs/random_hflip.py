"""A2 — Random Horizontal Flip (DL基礎 W4-5 transform pipeline 定番)."""

PROBLEM = {
    "id": "random_hflip",
    "number": 43,
    "title": "Random Horizontal Flip",
    "difficulty": "Easy",
    "fn_name": "random_horizontal_flip",

    "intro_md": (
        "**Random Horizontal Flip** を実装する。自然画像 (CIFAR-10, ImageNet) で定番の "
        "augmentation。batch 入力では各 sample が独立に確率 `p` で flip される。\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32)         → 確率 p=0.5 で flip\n"
        "x: (B=8, 3, 32, 32)    → 8 sample が独立に flip\n"
        "```"
    ),

    "signature": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # p: probability of flipping each sample\n"
        "    # Returns: same shape as x"
    ),

    "rules": [
        "`torchvision.transforms.RandomHorizontalFlip` や `T.functional.hflip` は **使わない**",
        "`(C, H, W)` と `(B, C, H, W)` 両方をサポート",
        "batch 入力では各 sample が **独立** に flip",
        "`p=0` → 恒等、`p=1` → 全 sample が flip",
        "`dtype` と `device` を保つ",
    ],

    "imports": "import torch",

    "template_body": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    pass  # (B, C, H, W) の場合: per-sample Bernoulli mask → 選ばれた sample を flip(-1)"
    ),

    "solution_body": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    if x.dim() == 3:\n"
        "        if torch.rand(1).item() < p:\n"
        "            return x.flip(-1)\n"
        "        return x\n"
        "    # (B, C, H, W) — per-sample 独立な mask\n"
        "    B = x.shape[0]\n"
        "    mask = torch.rand(B, device=x.device) < p\n"
        "    out = x.clone()\n"
        "    out[mask] = x[mask].flip(-1)\n"
        "    return out"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.arange(2*3*4*4).float().view(2, 3, 4, 4)\n"
        "out = random_horizontal_flip(x, p=0.5)\n"
        "print('Sample 0 flipped?', not torch.equal(out[0], x[0]))\n"
        "print('Sample 1 flipped?', not torch.equal(out[1], x[1]))"
    ),

    "hint": (
        "batch 入力では長さ B の Bernoulli mask (`torch.rand(B) < p`) をサンプルし、"
        "選ばれた sample だけ最後の dim で flip。`x.clone()` を作って "
        "`out[mask] = x[mask].flip(-1)` の fancy indexing で書き戻す。"
    ),

    "tests": [
        {
            "name": "Shape preserved (both (C,H,W) and (B,C,H,W))",
            "code": (
                "\nimport torch\n"
                "for shape in [(3, 8, 8), (4, 3, 8, 8)]:\n"
                "    x = torch.rand(*shape)\n"
                "    out = {fn}(x, p=0.5)\n"
                "    assert out.shape == x.shape, f'Shape {out.shape} != {shape}'\n"
            ),
        },
        {
            "name": "p=0 → identity",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.rand(4, 3, 8, 8)\n"
                "out = {fn}(x, p=0.0)\n"
                "assert torch.allclose(out, x), 'p=0 must return identity'\n"
            ),
        },
        {
            "name": "p=1 → all flipped",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.rand(4, 3, 8, 8)\n"
                "out = {fn}(x, p=1.0)\n"
                "assert torch.allclose(out, x.flip(-1)), 'p=1 must flip all samples'\n"
            ),
        },
        {
            "name": "Per-sample independence in batch",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(42)\n"
                "B = 100\n"
                "x = torch.randn(B, 3, 4, 4)\n"
                "out = {fn}(x, p=0.5)\n"
                "n_flipped = sum(torch.allclose(out[i], x[i].flip(-1)) for i in range(B))\n"
                "n_kept = sum(torch.allclose(out[i], x[i]) for i in range(B))\n"
                "assert n_flipped + n_kept == B, f'Some samples neither flipped nor kept: {B - n_flipped - n_kept}'\n"
                "assert 25 < n_flipped < 75, f'Expected ~50 flipped at p=0.5, got {n_flipped}'\n"
            ),
        },
        {
            "name": "dtype and device preserved",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 4, 4, dtype=torch.float64)\n"
                "out = {fn}(x, p=0.5)\n"
                "assert out.dtype == x.dtype, f'dtype: {out.dtype} != {x.dtype}'\n"
                "assert out.device == x.device, 'device mismatch'\n"
            ),
        },
    ],
}
