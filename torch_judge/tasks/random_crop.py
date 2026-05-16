"""Random Crop with Padding task — AUTO-GENERATED from problem_specs/random_crop.py. Do not edit directly."""

TASK = {
    "title": "Random Crop with Padding",
    "difficulty": "Easy",
    "function_name": "random_crop",
    "hint": "After F.pad, sample (i, j) uniformly in `[0, H_padded - size + 1)` and slice `x[..., i:i+size, j:j+size]`. For batched input, sample (i, j) **per sample** so each gets a different crop.",
    "tests": [
        {
            "name": "Output shape (B, C, size, size)",
            "code": "\nimport torch\nx = torch.rand(4, 3, 32, 32)\nout = {fn}(x, size=32, padding=4)\nassert out.shape == (4, 3, 32, 32), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Unbatched (C, H, W) also works",
            "code": "\nimport torch\nx = torch.rand(3, 16, 16)\nout = {fn}(x, size=12)\nassert out.shape == (3, 12, 12), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Identity when size == H and padding == 0",
            "code": "\nimport torch\nx = torch.rand(2, 3, 8, 8)\nout = {fn}(x, size=8, padding=0)\nassert torch.allclose(out, x), 'No-op crop must equal input'\n"
        },
        {
            "name": "Determinism with seed",
            "code": "\nimport torch\nx = torch.rand(1, 3, 16, 16)\ntorch.manual_seed(0)\na = {fn}(x, size=12, padding=0)\ntorch.manual_seed(0)\nb = {fn}(x, size=12, padding=0)\nassert torch.allclose(a, b), 'Same seed must give same crop'\n"
        },
        {
            "name": "Randomness — different seeds give different crops",
            "code": "\nimport torch\nx = torch.arange(1*1*20*20).float().view(1, 1, 20, 20)\ntorch.manual_seed(0)\na = {fn}(x, size=5, padding=0)\ntorch.manual_seed(1)\nb = {fn}(x, size=5, padding=0)\nassert not torch.allclose(a, b), 'Different seeds should give different crops'\n"
        }
    ]
}
