"""A4 — Cutout / RandomErasing (DeVries & Taylor 2017、W5 advanced 系)."""

PROBLEM = {
    "id": "cutout",
    "number": 45,
    "title": "Cutout / RandomErasing",
    "difficulty": "Medium",
    "fn_name": "cutout",

    "intro_md": (
        "**Cutout** を実装する。入力画像のランダムな `size × size` 矩形領域を zero mask する "
        "CNN 正則化 (DeVries & Taylor 2017)。\n\n"
        "batch 入力では各 sample が独立な mask 位置を持つ。Mask は全 channel に対して同じ "
        "空間位置に適用される。\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32), size=8 → 全 3 channel で同じ 8×8 領域を 0 に\n"
        "```"
    ),

    "signature": (
        "def cutout(x, size):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # size: int — side length of the zero-mask square\n"
        "    # Returns: same shape, one size×size region per sample zeroed"
    ),

    "rules": [
        "`torchvision.transforms.RandomErasing` は **使わない**",
        "Mask は `size × size` の正方形、画像内に **完全に収まる** ように配置",
        "全 channel の同じ空間位置に適用",
        "batch 入力では各 sample が **独立** な mask 位置",
        "input を in-place で改変しない（新規 tensor を return）",
    ],

    "imports": "import torch",

    "template_body": (
        "def cutout(x, size):\n"
        "    pass  # input を clone、sample ごとに (i, j) sampling、out[..., i:i+size, j:j+size] = 0"
    ),

    "solution_body": (
        "def cutout(x, size):\n"
        "    out = x.clone()\n"
        "    if x.dim() == 3:\n"
        "        _, H, W = x.shape\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        out[:, i:i+size, j:j+size] = 0\n"
        "        return out\n"
        "    B, _, H, W = x.shape\n"
        "    for b in range(B):\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        out[b, :, i:i+size, j:j+size] = 0\n"
        "    return out"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.ones(1, 3, 16, 16)\n"
        "out = cutout(x, size=4)\n"
        "print('Zeros in output:', (out == 0).sum().item(), '(expected 3 * 4 * 4 = 48)')"
    ),

    "hint": (
        "先に clone して in-place 改変を回避。`(i, j)` を `[0, H-size+1)` でサンプル。"
        "mask は全 channel の同じ空間 slice に適用するので、sample あたりの zero 数 = `C * size * size`。"
    ),

    "tests": [
        {
            "name": "Shape preserved",
            "code": (
                "\nimport torch\n"
                "for shape in [(3, 16, 16), (4, 3, 16, 16)]:\n"
                "    x = torch.rand(*shape) + 1.0\n"
                "    out = {fn}(x, size=4)\n"
                "    assert out.shape == x.shape, f'Shape {out.shape} != {shape}'\n"
            ),
        },
        {
            "name": "Exactly C*size*size zeros per sample",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.rand(4, 3, 16, 16) + 1.0  # all positive — no input zeros\n"
                "out = {fn}(x, size=5)\n"
                "expected = 3 * 5 * 5\n"
                "for b in range(4):\n"
                "    n_zero = (out[b] == 0).sum().item()\n"
                "    assert n_zero == expected, f'Sample {b}: {n_zero} zeros, expected {expected}'\n"
            ),
        },
        {
            "name": "Per-sample independent mask positions",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.rand(8, 1, 32, 32) + 1.0\n"
                "out = {fn}(x, size=4)\n"
                "positions = set()\n"
                "for b in range(8):\n"
                "    zero_mask = (out[b, 0] == 0)\n"
                "    rows = zero_mask.any(dim=1).nonzero().flatten()\n"
                "    cols = zero_mask.any(dim=0).nonzero().flatten()\n"
                "    positions.add((rows.min().item(), cols.min().item()))\n"
                "assert len(positions) > 1, f'All {len(positions)} masks at same position — not independent'\n"
            ),
        },
        {
            "name": "Input not modified in-place",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 8, 8) + 1.0\n"
                "x_orig = x.clone()\n"
                "_ = {fn}(x, size=3)\n"
                "assert torch.allclose(x, x_orig), 'Input was modified in-place'\n"
            ),
        },
        {
            "name": "Determinism with seed",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 16, 16) + 1.0\n"
                "torch.manual_seed(0)\n"
                "a = {fn}(x, size=4)\n"
                "torch.manual_seed(0)\n"
                "b = {fn}(x, size=4)\n"
                "assert torch.allclose(a, b), 'Same seed must give same mask'\n"
            ),
        },
    ],
}
