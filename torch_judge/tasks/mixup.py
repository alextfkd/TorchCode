"""Mixup task — AUTO-GENERATED from problem_specs/mixup.py. Do not edit directly."""

TASK = {
    "title": "Mixup",
    "difficulty": "Medium",
    "function_name": "mixup",
    "hint": "`torch.distributions.Beta(alpha, alpha).sample().item()` で λ を Python float で取得。`torch.randperm(B)` で shuffle。`(x_mixed, y, y[perm], lam)` を return — loss は外で `λ*CE(pred, y_a) + (1-λ)*CE(pred, y_b)` の形で組む。",
    "tests": [
        {
            "name": "Output is a 4-tuple with correct shapes",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(4, 3, 8, 8)\ny = torch.arange(4)\nout = {fn}(x, y, alpha=1.0)\nassert len(out) == 4, f'Expected 4-tuple, got {len(out)}-tuple'\nx_mix, y_a, y_b, lam = out\nassert x_mix.shape == x.shape, f'x_mix shape {x_mix.shape}'\nassert y_a.shape == y.shape and y_b.shape == y.shape, 'label shapes wrong'\n"
        },
        {
            "name": "y_a equals original y",
            "code": "\nimport torch\ntorch.manual_seed(0)\ny = torch.tensor([5, 7, 2, 9])\nx = torch.randn(4, 3)\n_, y_a, _, _ = {fn}(x, y, alpha=1.0)\nassert torch.equal(y_a, y), f'y_a {y_a.tolist()} != y {y.tolist()}'\n"
        },
        {
            "name": "y_b is a permutation of y",
            "code": "\nimport torch\ntorch.manual_seed(0)\ny = torch.tensor([5, 7, 2, 9])\nx = torch.randn(4, 3)\n_, _, y_b, _ = {fn}(x, y, alpha=1.0)\nassert sorted(y_b.tolist()) == sorted(y.tolist()), f'y_b not a permutation: {y_b.tolist()}'\n"
        },
        {
            "name": "lam is a Python float in [0, 1]",
            "code": "\nimport torch\nfor seed in range(5):\n    torch.manual_seed(seed)\n    x = torch.randn(4, 3)\n    y = torch.arange(4)\n    _, _, _, lam = {fn}(x, y, alpha=1.0)\n    assert isinstance(lam, float), f'lam should be float, got {type(lam).__name__}'\n    assert 0.0 <= lam <= 1.0, f'lam {lam} out of [0,1]'\n"
        },
        {
            "name": "x_mix == lam*x + (1-lam)*x[perm] (perm inferred from y_b)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nB = 8\nx = torch.randn(B, 3, 4, 4)\ny = torch.arange(B)  # so y_b == perm directly\nx_mix, y_a, y_b, lam = {fn}(x, y, alpha=1.0)\nperm = y_b\nexpected = lam * x + (1 - lam) * x[perm]\nassert torch.allclose(x_mix, expected, atol=1e-5), f'Max diff: {(x_mix-expected).abs().max():.6f}'\n"
        }
    ]
}
