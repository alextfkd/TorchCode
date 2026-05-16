"""CutMix task — AUTO-GENERATED from problem_specs/cutmix.py. Do not edit directly."""

TASK = {
    "title": "CutMix",
    "difficulty": "Medium",
    "function_name": "cutmix",
    "hint": "`cut_h = int(H * sqrt(1-λ))`. Sample center uniformly, then clip with `y1=max(0, cy-cut_h//2)` etc. Paste `x[perm, :, y1:y2, x1:x2]` into the same slice of `x_mixed`. **Recompute λ** from the realized `(y2-y1)*(x2-x1) / (H*W)` area before returning — this matters when the rectangle is clipped at boundaries.",
    "tests": [
        {
            "name": "Output is a 4-tuple with correct shapes",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(4, 3, 16, 16)\ny = torch.arange(4)\nout = {fn}(x, y, alpha=1.0)\nassert len(out) == 4\nx_mix, y_a, y_b, lam = out\nassert x_mix.shape == x.shape\nassert y_a.shape == y.shape and y_b.shape == y.shape\n"
        },
        {
            "name": "y_a is original, y_b is a permutation of y",
            "code": "\nimport torch\ntorch.manual_seed(0)\ny = torch.tensor([5, 7, 2, 9])\nx = torch.randn(4, 3, 8, 8)\n_, y_a, y_b, _ = {fn}(x, y, alpha=1.0)\nassert torch.equal(y_a, y), 'y_a must equal y'\nassert sorted(y_b.tolist()) == sorted(y.tolist()), 'y_b not a permutation'\n"
        },
        {
            "name": "lam is float in [0, 1]",
            "code": "\nimport torch\nfor seed in range(8):\n    torch.manual_seed(seed)\n    x = torch.randn(4, 3, 16, 16)\n    y = torch.arange(4)\n    _, _, _, lam = {fn}(x, y, alpha=1.0)\n    assert isinstance(lam, float), f'lam should be float, got {type(lam).__name__}'\n    assert 0.0 <= lam <= 1.0, f'lam {lam} out of [0, 1]'\n"
        },
        {
            "name": "lam matches actual cut area (recomputed after clipping)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nB, C, H, W = 2, 1, 32, 32\nx = torch.zeros(B, C, H, W)\nx[1] = 1.0\ny = torch.tensor([0, 1])\nx_mix, y_a, y_b, lam = {fn}(x, y, alpha=1.0)\nif y_b[0].item() == 1:\n    # sample 0 received pasted region from sample 1 (ones)\n    ones = (x_mix[0] == 1).sum().item()\n    expected_cut = (1.0 - lam) * H * W\n    assert abs(ones - expected_cut) < 1.0, f'Cut area {ones} != (1-lam)*H*W = {expected_cut:.1f}'\nelse:\n    assert (x_mix[0] == 0).all(), 'When perm[0]=0, sample 0 should be unchanged'\n"
        },
        {
            "name": "Pixels are either from x or x[perm] — no blending",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(4, 3, 16, 16)\ny = torch.arange(4)\nx_orig = x.clone()\nx_mix, _, y_b, _ = {fn}(x, y, alpha=1.0)\nperm = y_b\nis_from_x = (x_mix == x_orig)\nis_from_perm = (x_mix == x_orig[perm])\nassert (is_from_x | is_from_perm).all(), 'Every pixel must come from x or x[perm], not be blended'\n"
        }
    ]
}
