"""Random Horizontal Flip task — AUTO-GENERATED from problem_specs/random_hflip.py. Do not edit directly."""

TASK = {
    "title": "Random Horizontal Flip",
    "difficulty": "Easy",
    "function_name": "random_horizontal_flip",
    "hint": "batch 入力では長さ B の Bernoulli mask (`torch.rand(B) < p`) をサンプルし、選ばれた sample だけ最後の dim で flip。`x.clone()` を作って `out[mask] = x[mask].flip(-1)` の fancy indexing で書き戻す。",
    "tests": [
        {
            "name": "Shape preserved (both (C,H,W) and (B,C,H,W))",
            "code": "\nimport torch\nfor shape in [(3, 8, 8), (4, 3, 8, 8)]:\n    x = torch.rand(*shape)\n    out = {fn}(x, p=0.5)\n    assert out.shape == x.shape, f'Shape {out.shape} != {shape}'\n"
        },
        {
            "name": "p=0 → identity",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.rand(4, 3, 8, 8)\nout = {fn}(x, p=0.0)\nassert torch.allclose(out, x), 'p=0 must return identity'\n"
        },
        {
            "name": "p=1 → all flipped",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.rand(4, 3, 8, 8)\nout = {fn}(x, p=1.0)\nassert torch.allclose(out, x.flip(-1)), 'p=1 must flip all samples'\n"
        },
        {
            "name": "Per-sample independence in batch",
            "code": "\nimport torch\ntorch.manual_seed(42)\nB = 100\nx = torch.randn(B, 3, 4, 4)\nout = {fn}(x, p=0.5)\nn_flipped = sum(torch.allclose(out[i], x[i].flip(-1)) for i in range(B))\nn_kept = sum(torch.allclose(out[i], x[i]) for i in range(B))\nassert n_flipped + n_kept == B, f'Some samples neither flipped nor kept: {B - n_flipped - n_kept}'\nassert 25 < n_flipped < 75, f'Expected ~50 flipped at p=0.5, got {n_flipped}'\n"
        },
        {
            "name": "dtype and device preserved",
            "code": "\nimport torch\nx = torch.rand(2, 3, 4, 4, dtype=torch.float64)\nout = {fn}(x, p=0.5)\nassert out.dtype == x.dtype, f'dtype: {out.dtype} != {x.dtype}'\nassert out.device == x.device, 'device mismatch'\n"
        }
    ]
}
