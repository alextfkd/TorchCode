"""Global Average Pooling task — AUTO-GENERATED from problem_specs/global_avg_pool.py. Do not edit directly."""

TASK = {
    "title": "Global Average Pooling",
    "difficulty": "Easy",
    "function_name": "global_avg_pool",
    "hint": "1行: `x.mean(dim=(-2, -1))`。output が `(B, C)` であって `(B, C, 1, 1)` じゃないこと — `nn.Linear` に直接流すため。",
    "tests": [
        {
            "name": "Output shape (B, C), not (B, C, 1, 1)",
            "code": "\nimport torch\nx = torch.randn(4, 16, 7, 7)\nout = {fn}(x)\nassert out.shape == (4, 16), f'Shape: {out.shape} — must be (B, C) not (B, C, 1, 1)'\n"
        },
        {
            "name": "Matches mean over (H, W)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(2, 8, 6, 6)\nout = {fn}(x)\nref = x.mean(dim=(-2, -1))\nassert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max()}'\n"
        },
        {
            "name": "Matches F.adaptive_avg_pool2d(x, 1).flatten(1)",
            "code": "\nimport torch\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nx = torch.randn(3, 12, 5, 5)\nout = {fn}(x)\nref = F.adaptive_avg_pool2d(x, 1).flatten(1)\nassert torch.allclose(out, ref, atol=1e-6)\n"
        },
        {
            "name": "Works for non-square (H ≠ W)",
            "code": "\nimport torch\nx = torch.randn(2, 4, 6, 10)\nout = {fn}(x)\nassert out.shape == (2, 4), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Gradient distributes as 1/(H*W) per input",
            "code": "\nimport torch\nx = torch.randn(1, 3, 4, 4, requires_grad=True)\nout = {fn}(x)\nout.sum().backward()\nexpected = torch.full_like(x, 1.0 / (4 * 4))\nassert torch.allclose(x.grad, expected, atol=1e-5), 'Gradient should be 1/(H*W) per input'\n"
        }
    ]
}
