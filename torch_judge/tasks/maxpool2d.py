"""2D Max Pooling task — AUTO-GENERATED from problem_specs/maxpool2d.py. Do not edit directly."""

TASK = {
    "title": "2D Max Pooling",
    "difficulty": "Medium",
    "function_name": "my_max_pool2d",
    "hint": "`x.unfold(2, k, s).unfold(3, k, s)` で patch を取って shape `(B, C, H_out, W_out, kH, kW)`、最後の 2 dim で max。padding は `float('-inf')` で （NOT 0 — negative input が pad に負けるため）。",
    "tests": [
        {
            "name": "Output shape (default stride)",
            "code": "\nimport torch\nx = torch.randn(2, 3, 8, 8)\nout = {fn}(x, kernel_size=2)\nassert out.shape == (2, 3, 4, 4), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Matches F.max_pool2d",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(2, 4, 8, 8)\nout = {fn}(x, kernel_size=2)\nref = torch.nn.functional.max_pool2d(x, kernel_size=2)\nassert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Custom stride",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 2, 7, 7)\nout = {fn}(x, kernel_size=3, stride=2)\nref = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)\nassert out.shape == ref.shape, f'Shape: {out.shape} vs {ref.shape}'\nassert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Padding with -inf (negative inputs)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = -torch.rand(1, 1, 4, 4) - 0.1\nout = {fn}(x, kernel_size=3, stride=1, padding=1)\nref = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)\nassert out.shape == ref.shape, f'Shape: {out.shape} vs {ref.shape}'\nassert torch.allclose(out, ref, atol=1e-6), f'Padding must use -inf (not 0). Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Gradient routes to argmax only",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 1, 4, 4, requires_grad=True)\nout = {fn}(x, kernel_size=2)\nout.sum().backward()\nassert x.grad is not None, 'Missing gradient'\nnonzero = (x.grad != 0).sum().item()\nassert nonzero == 4, f'Expected exactly 4 non-zero grads (one per 2x2 window), got {nonzero}'\n"
        }
    ]
}
