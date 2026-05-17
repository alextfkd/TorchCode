"""2D Average Pooling task — AUTO-GENERATED from problem_specs/avg_pool2d.py. Do not edit directly."""

TASK = {
    "title": "2D Average Pooling",
    "difficulty": "Easy",
    "function_name": "my_avg_pool2d",
    "hint": "MaxPool と同じ `unfold(2, k, s).unfold(3, k, s)` を使うが、`.amax(...)` の代わりに `.mean(dim=(-1, -2))`。zero pad で `F.pad`（default）、zero-pad 値は mean に含まれる — PyTorch の `count_include_pad=True` default 挙動。",
    "tests": [
        {
            "name": "Output shape (default stride)",
            "code": "\nimport torch\nx = torch.randn(2, 3, 8, 8)\nout = {fn}(x, kernel_size=2)\nassert out.shape == (2, 3, 4, 4), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Matches F.avg_pool2d",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(2, 4, 8, 8)\nout = {fn}(x, kernel_size=2)\nref = torch.nn.functional.avg_pool2d(x, kernel_size=2)\nassert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Custom stride",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 2, 7, 7)\nout = {fn}(x, kernel_size=3, stride=2)\nref = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2)\nassert out.shape == ref.shape\nassert torch.allclose(out, ref, atol=1e-6)\n"
        },
        {
            "name": "Padding includes zeros in the mean (default count_include_pad=True)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(1, 1, 4, 4)\nout = {fn}(x, kernel_size=3, stride=1, padding=1)\nref = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)\nassert out.shape == ref.shape\nassert torch.allclose(out, ref, atol=1e-6), 'count_include_pad=True default — zeros count in mean'\n"
        },
        {
            "name": "Gradient distributes uniformly (1/k² per input)",
            "code": "\nimport torch\nx = torch.randn(1, 1, 4, 4, requires_grad=True)\nout = {fn}(x, kernel_size=2)\nout.sum().backward()\nexpected = torch.full_like(x, 0.25)\nassert torch.allclose(x.grad, expected, atol=1e-5), 'Gradient should be 1/(k*k) per input'\n"
        }
    ]
}
