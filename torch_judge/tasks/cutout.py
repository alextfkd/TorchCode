"""Cutout / RandomErasing task — AUTO-GENERATED from problem_specs/cutout.py. Do not edit directly."""

TASK = {
    "title": "Cutout / RandomErasing",
    "difficulty": "Medium",
    "function_name": "cutout",
    "hint": "Clone the input first to avoid in-place modification. Sample (i, j) in `[0, H-size+1)`. The mask covers all channels at the chosen spatial slice, so zeros per sample = `C * size * size`.",
    "tests": [
        {
            "name": "Shape preserved",
            "code": "\nimport torch\nfor shape in [(3, 16, 16), (4, 3, 16, 16)]:\n    x = torch.rand(*shape) + 1.0\n    out = {fn}(x, size=4)\n    assert out.shape == x.shape, f'Shape {out.shape} != {shape}'\n"
        },
        {
            "name": "Exactly C*size*size zeros per sample",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.rand(4, 3, 16, 16) + 1.0  # all positive — no input zeros\nout = {fn}(x, size=5)\nexpected = 3 * 5 * 5\nfor b in range(4):\n    n_zero = (out[b] == 0).sum().item()\n    assert n_zero == expected, f'Sample {b}: {n_zero} zeros, expected {expected}'\n"
        },
        {
            "name": "Per-sample independent mask positions",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.rand(8, 1, 32, 32) + 1.0\nout = {fn}(x, size=4)\npositions = set()\nfor b in range(8):\n    zero_mask = (out[b, 0] == 0)\n    rows = zero_mask.any(dim=1).nonzero().flatten()\n    cols = zero_mask.any(dim=0).nonzero().flatten()\n    positions.add((rows.min().item(), cols.min().item()))\nassert len(positions) > 1, f'All {len(positions)} masks at same position — not independent'\n"
        },
        {
            "name": "Input not modified in-place",
            "code": "\nimport torch\nx = torch.rand(2, 3, 8, 8) + 1.0\nx_orig = x.clone()\n_ = {fn}(x, size=3)\nassert torch.allclose(x, x_orig), 'Input was modified in-place'\n"
        },
        {
            "name": "Determinism with seed",
            "code": "\nimport torch\nx = torch.rand(2, 3, 16, 16) + 1.0\ntorch.manual_seed(0)\na = {fn}(x, size=4)\ntorch.manual_seed(0)\nb = {fn}(x, size=4)\nassert torch.allclose(a, b), 'Same seed must give same mask'\n"
        }
    ]
}
