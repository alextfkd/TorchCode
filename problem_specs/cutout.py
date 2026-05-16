"""A4 — Cutout / RandomErasing (DeVries & Taylor 2017、W5 advanced 系)."""

PROBLEM = {
    "id": "cutout",
    "number": 45,
    "title": "Cutout / RandomErasing",
    "difficulty": "Medium",
    "fn_name": "cutout",

    "intro_md": (
        "Implement **Cutout** — mask a random `size × size` region of the input image with zeros. "
        "A surprisingly effective CNN regularizer (DeVries & Taylor 2017).\n\n"
        "For batched input, each sample gets an independently-sampled mask position. The mask "
        "covers all channels at the chosen spatial position.\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32), size=8 → one random 8×8 patch zeroed across all 3 channels\n"
        "```"
    ),

    "signature": (
        "def cutout(x, size):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # size: int — side length of the zero-mask square\n"
        "    # Returns: same shape, one size×size region per sample zeroed"
    ),

    "rules": [
        "Do **NOT** use `torchvision.transforms.RandomErasing`",
        "Mask is a `size × size` square at a random position, **fully inside** the image",
        "Mask is applied across ALL channels at the chosen spatial position",
        "For batched input, each sample gets an **independent** mask position",
        "Must NOT modify the input in-place — return a new tensor",
    ],

    "imports": "import torch",

    "template_body": (
        "def cutout(x, size):\n"
        "    pass  # clone input, sample (i, j) per sample, set out[..., i:i+size, j:j+size] = 0"
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
        "Clone the input first to avoid in-place modification. Sample (i, j) in `[0, H-size+1)`. "
        "The mask covers all channels at the chosen spatial slice, so zeros per sample = `C * size * size`."
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
