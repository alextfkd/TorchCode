"""B1 — 2D Average Pooling (MaxPool の妹、Conv 後のスムーズなダウンサンプル)."""

PROBLEM = {
    "id": "avg_pool2d",
    "number": 49,
    "title": "2D Average Pooling",
    "difficulty": "Easy",
    "fn_name": "my_avg_pool2d",

    "intro_md": (
        "**2D Average Pooling** を実装する。max pooling の smooth 版で、Global Average Pooling "
        "の構成要素でもある。\n\n"
        "$$\\text{out}[b, c, i, j] = \\frac{1}{kH \\cdot kW} \\sum_{(p,q) \\in \\text{window}(i,j)} x[b, c, p, q]$$\n\n"
        "### MaxPool との違い\n"
        "- 平均は **smooth**（どこでも微分可能）、max は non-smooth\n"
        "- Padding は **zero** で行う（`-inf` ではない）、zero-pad 値は **平均に含まれる**（PyTorch default）\n"
    ),

    "signature": (
        "def my_avg_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    # x: (B, C, H, W)\n"
        "    # Returns: (B, C, H_out, W_out)"
    ),

    "rules": [
        "`F.avg_pool2d`, `nn.AvgPool2d`, `F.adaptive_avg_pool2d` は **使わない**",
        "`unfold`, `reshape`, `.mean()` は許可",
        "`stride` の default は `kernel_size` (non-overlapping windows)",
        "`padding > 0` の場合、**zero** で pad（PyTorch の `count_include_pad=True` default）",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def my_avg_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    pass  # F.pad で zero pad、unfold(2,k,s).unfold(3,k,s)、window dim で .mean"
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
        "MaxPool と同じ `unfold(2, k, s).unfold(3, k, s)` を使うが、`.amax(...)` の代わりに "
        "`.mean(dim=(-1, -2))`。zero pad で `F.pad`（default）、zero-pad 値は mean に含まれる — "
        "PyTorch の `count_include_pad=True` default 挙動。"
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
