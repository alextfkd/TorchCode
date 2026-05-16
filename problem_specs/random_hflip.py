"""A2 — Random Horizontal Flip (DL基礎 W4-5 transform pipeline 定番)."""

PROBLEM = {
    "id": "random_hflip",
    "number": 43,
    "title": "Random Horizontal Flip",
    "difficulty": "Easy",
    "fn_name": "random_horizontal_flip",

    "intro_md": (
        "Implement **random horizontal flip** — the canonical augmentation for natural images "
        "(CIFAR-10, ImageNet). Each sample in a batch flips independently with probability `p`.\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32)         → maybe flipped (50% chance with p=0.5)\n"
        "x: (B=8, 3, 32, 32)    → each of 8 samples flipped independently\n"
        "```"
    ),

    "signature": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # p: probability of flipping each sample\n"
        "    # Returns: same shape as x"
    ),

    "rules": [
        "Do **NOT** use `torchvision.transforms.RandomHorizontalFlip` or `T.functional.hflip`",
        "Must support both `(C, H, W)` and `(B, C, H, W)` inputs",
        "For batched input, each sample must flip **independently**",
        "`p=0` → identity, `p=1` → all samples flipped",
        "Must preserve `dtype` and `device`",
    ],

    "imports": "import torch",

    "template_body": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    pass  # for (B, C, H, W): per-sample Bernoulli mask, then flip(-1) on selected"
    ),

    "solution_body": (
        "def random_horizontal_flip(x, p=0.5):\n"
        "    if x.dim() == 3:\n"
        "        if torch.rand(1).item() < p:\n"
        "            return x.flip(-1)\n"
        "        return x\n"
        "    # (B, C, H, W) — per-sample independent mask\n"
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
        "For batched input, sample a Bernoulli mask of length B (`torch.rand(B) < p`), then "
        "flip x along the last dim only for the selected samples. Use `x.clone()` for the output and "
        "write back with fancy indexing: `out[mask] = x[mask].flip(-1)`."
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
