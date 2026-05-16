"""B1 — 2D Average Pooling (MaxPool の妹、Conv 後のスムーズなダウンサンプル)."""

PROBLEM = {
    "id": "avg_pool2d",
    "number": 49,
    "title": "2D Average Pooling",
    "difficulty": "Easy",
    "fn_name": "my_avg_pool2d",

    "intro_md": (
        "Implement **2D average pooling** from scratch — the smoother alternative to max pooling, "
        "and the building block of Global Average Pooling.\n\n"
        "$$\\text{out}[b, c, i, j] = \\frac{1}{kH \\cdot kW} \\sum_{(p,q) \\in \\text{window}(i,j)} x[b, c, p, q]$$\n\n"
        "### Difference from MaxPool\n"
        "- Average is **smooth** (differentiable everywhere); max is non-smooth\n"
        "- Padding uses **zeros** (NOT `-inf`), and zero-pad values are **included** in the mean (PyTorch default)\n"
    ),

    "signature": (
        "def my_avg_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    # x: (B, C, H, W)\n"
        "    # Returns: (B, C, H_out, W_out)"
    ),

    "rules": [
        "Do **NOT** use `F.avg_pool2d`, `nn.AvgPool2d`, or `F.adaptive_avg_pool2d`",
        "`unfold`, `reshape`, `.mean()` are allowed",
        "`stride` defaults to `kernel_size` (non-overlapping windows)",
        "For `padding > 0`, pad with **zeros** (PyTorch's `count_include_pad=True` default)",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def my_avg_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    pass  # F.pad with zeros, unfold(2,k,s).unfold(3,k,s), then .mean over the window dims"
    ),

    "solution_body": (
        "def my_avg_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    if stride is None:\n"
        "        stride = kernel_size\n"
        "    if padding > 0:\n"
        "        x = F.pad(x, [padding] * 4)  # zero pad\n"
        "    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)\n"
        "    return patches.mean(dim=(-1, -2))"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.arange(1*1*4*4).float().view(1, 1, 4, 4)\n"
        "print('Input:\\n', x.squeeze())\n"
        "print('AvgPool(k=2):\\n', my_avg_pool2d(x, kernel_size=2).squeeze())"
    ),

    "hint": (
        "Same `unfold(2, k, s).unfold(3, k, s)` trick as MaxPool, but use `.mean(dim=(-1, -2))` "
        "instead of `.amax(...)`. Pad with zeros (default `F.pad`), and the zero-pad values get "
        "included in the mean — that matches PyTorch's `count_include_pad=True` default."
    ),

    "tests": [
        {
            "name": "Output shape (default stride)",
            "code": (
                "\nimport torch\n"
                "x = torch.randn(2, 3, 8, 8)\n"
                "out = {fn}(x, kernel_size=2)\n"
                "assert out.shape == (2, 3, 4, 4), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Matches F.avg_pool2d",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(2, 4, 8, 8)\n"
                "out = {fn}(x, kernel_size=2)\n"
                "ref = torch.nn.functional.avg_pool2d(x, kernel_size=2)\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
            ),
        },
        {
            "name": "Custom stride",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(1, 2, 7, 7)\n"
                "out = {fn}(x, kernel_size=3, stride=2)\n"
                "ref = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2)\n"
                "assert out.shape == ref.shape\n"
                "assert torch.allclose(out, ref, atol=1e-6)\n"
            ),
        },
        {
            "name": "Padding includes zeros in the mean (default count_include_pad=True)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(1, 1, 4, 4)\n"
                "out = {fn}(x, kernel_size=3, stride=1, padding=1)\n"
                "ref = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)\n"
                "assert out.shape == ref.shape\n"
                "assert torch.allclose(out, ref, atol=1e-6), 'count_include_pad=True default — zeros count in mean'\n"
            ),
        },
        {
            "name": "Gradient distributes uniformly (1/k² per input)",
            "code": (
                "\nimport torch\n"
                "x = torch.randn(1, 1, 4, 4, requires_grad=True)\n"
                "out = {fn}(x, kernel_size=2)\n"
                "out.sum().backward()\n"
                "expected = torch.full_like(x, 0.25)\n"
                "assert torch.allclose(x.grad, expected, atol=1e-5), 'Gradient should be 1/(k*k) per input'\n"
            ),
        },
    ],
}
