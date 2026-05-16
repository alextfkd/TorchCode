"""A3 — Random Crop with Padding (CIFAR-10 pad=4, crop=32 の定石)."""

PROBLEM = {
    "id": "random_crop",
    "number": 44,
    "title": "Random Crop with Padding",
    "difficulty": "Easy",
    "fn_name": "random_crop",

    "intro_md": (
        "Implement **random crop with zero-padding** — the standard CIFAR-10 trick: pad the image "
        "by 4 pixels, then random-crop back to 32×32.\n\n"
        "Adds translation invariance to training data with essentially zero compute cost.\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32), padding=4 → padded to (3, 40, 40) → cropped to (3, 32, 32) at random offset\n"
        "```"
    ),

    "signature": (
        "def random_crop(x, size, padding=0):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # size: int — output crop size (square)\n"
        "    # padding: int — zero-pad on all sides before cropping\n"
        "    # Returns: (C, size, size) or (B, C, size, size)"
    ),

    "rules": [
        "Do **NOT** use `torchvision.transforms.RandomCrop`",
        "Pad with zeros first (`F.pad`), then random crop of `size × size`",
        "Support both `(C, H, W)` and `(B, C, H, W)`",
        "For batched input, each sample cropped at an **independently-sampled** offset",
        "`size` is a single int (square output)",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def random_crop(x, size, padding=0):\n"
        "    pass  # F.pad first, then sample (i, j) and slice x[..., i:i+size, j:j+size]"
    ),

    "solution_body": (
        "def random_crop(x, size, padding=0):\n"
        "    if padding > 0:\n"
        "        x = F.pad(x, [padding] * 4)\n"
        "    if x.dim() == 3:\n"
        "        _, H, W = x.shape\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        return x[:, i:i+size, j:j+size]\n"
        "    # (B, C, H, W) — per-sample offsets\n"
        "    B = x.shape[0]\n"
        "    H, W = x.shape[-2], x.shape[-1]\n"
        "    out = torch.empty(B, x.shape[1], size, size, dtype=x.dtype, device=x.device)\n"
        "    for b in range(B):\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        out[b] = x[b, :, i:i+size, j:j+size]\n"
        "    return out"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.rand(4, 3, 32, 32)\n"
        "out = random_crop(x, size=32, padding=4)\n"
        "print('In:', x.shape, '→ Out:', out.shape)"
    ),

    "hint": (
        "After F.pad, sample (i, j) uniformly in `[0, H_padded - size + 1)` and slice "
        "`x[..., i:i+size, j:j+size]`. For batched input, sample (i, j) **per sample** so each gets "
        "a different crop."
    ),

    "tests": [
        {
            "name": "Output shape (B, C, size, size)",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(4, 3, 32, 32)\n"
                "out = {fn}(x, size=32, padding=4)\n"
                "assert out.shape == (4, 3, 32, 32), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Unbatched (C, H, W) also works",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(3, 16, 16)\n"
                "out = {fn}(x, size=12)\n"
                "assert out.shape == (3, 12, 12), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Identity when size == H and padding == 0",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 8, 8)\n"
                "out = {fn}(x, size=8, padding=0)\n"
                "assert torch.allclose(out, x), 'No-op crop must equal input'\n"
            ),
        },
        {
            "name": "Determinism with seed",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(1, 3, 16, 16)\n"
                "torch.manual_seed(0)\n"
                "a = {fn}(x, size=12, padding=0)\n"
                "torch.manual_seed(0)\n"
                "b = {fn}(x, size=12, padding=0)\n"
                "assert torch.allclose(a, b), 'Same seed must give same crop'\n"
            ),
        },
        {
            "name": "Randomness — different seeds give different crops",
            "code": (
                "\nimport torch\n"
                "x = torch.arange(1*1*20*20).float().view(1, 1, 20, 20)\n"
                "torch.manual_seed(0)\n"
                "a = {fn}(x, size=5, padding=0)\n"
                "torch.manual_seed(1)\n"
                "b = {fn}(x, size=5, padding=0)\n"
                "assert not torch.allclose(a, b), 'Different seeds should give different crops'\n"
            ),
        },
    ],
}
