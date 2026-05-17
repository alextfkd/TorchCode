"""A1 — Per-Channel Normalize (DL基礎 W4-5 transform pipeline の最初)."""

PROBLEM = {
    "id": "normalize",
    "number": 42,
    "title": "Per-Channel Normalize",
    "difficulty": "Easy",
    "fn_name": "my_normalize",

    "intro_md": (
        "**channel ごとの正規化** を実装する。CNN 学習パイプラインで一番最初に走る "
        "transform（CIFAR-10, ImageNet など）。\n\n"
        "$$\\text{out}[c, h, w] = \\frac{x[c, h, w] - \\mu[c]}{\\sigma[c]}$$\n\n"
        "### Example\n"
        "```\n"
        "x       : (3, 32, 32) float image in [0, 1]\n"
        "mean    : (0.4914, 0.4822, 0.4465)   # CIFAR-10 の channel 平均\n"
        "std     : (0.2470, 0.2435, 0.2616)   # CIFAR-10 の channel 標準偏差\n"
        "→ (3, 32, 32)、各 channel が z-score 化された結果\n"
        "```"
    ),

    "signature": (
        "def my_normalize(x, mean, std):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # mean, std: length-C tuple/list/tensor\n"
        "    # Returns: same shape as x"
    ),

    "rules": [
        "`torchvision.transforms.Normalize` や `T.functional.normalize` は **使わない**",
        "`(C, H, W)` と `(B, C, H, W)` 両方の入力をサポート",
        "`mean`, `std` は tuple / list / tensor のいずれも受け付ける",
        "`x` の `dtype` と `device` を保つ",
        "勾配が `x` に流れること",
    ],

    "imports": "import torch",

    "template_body": (
        "def my_normalize(x, mean, std):\n"
        "    pass  # mean/std を (C, 1, 1) に broadcast してから (x - μ) / σ"
    ),

    "solution_body": (
        "def my_normalize(x, mean, std):\n"
        "    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)\n"
        "    std = torch.as_tensor(std, dtype=x.dtype, device=x.device)\n"
        "    # (C, 1, 1) に reshape — (C, H, W) と (B, C, H, W) どちらでも broadcast が効く\n"
        "    shape = [-1, 1, 1]\n"
        "    return (x - mean.view(shape)) / std.view(shape)"
    ),

    "demo_code": (
        "# 🧪 Test your implementation (feel free to add more debug prints)\n"
        "import torch\n"
        "torch.manual_seed(0)\n"
        "\n"
        "x = torch.rand(3, 4, 4)  # fake CIFAR-style image\n"
        "mean = (0.4914, 0.4822, 0.4465)\n"
        "std = (0.2470, 0.2435, 0.2616)\n"
        "out = my_normalize(x, mean, std)\n"
        "print('Input  shape:', x.shape, 'mean per ch:', x.mean(dim=(1,2)))\n"
        "print('Output shape:', out.shape, 'mean per ch:', out.mean(dim=(1,2)))"
    ),

    "hint": (
        "`torch.as_tensor` で mean/std を x と同じ dtype/device の tensor に変換"
        "（tuple/list/tensor を一律で扱える）。それを `(C, 1, 1)` に reshape すれば "
        "`(C, H, W)` と `(B, C, H, W)` 両方で broadcast が効く。"
    ),

    "tests": [
        {
            "name": "Shape preserved (B, C, H, W)",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 8, 8)\n"
                "out = {fn}(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n"
                "assert out.shape == x.shape, f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Matches manual (x - mean) / std",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.rand(2, 3, 4, 4)\n"
                "mean = (0.4914, 0.4822, 0.4465)\n"
                "std = (0.2470, 0.2435, 0.2616)\n"
                "out = {fn}(x, mean, std)\n"
                "m = torch.tensor(mean).view(-1, 1, 1)\n"
                "s = torch.tensor(std).view(-1, 1, 1)\n"
                "ref = (x - m) / s\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
            ),
        },
        {
            "name": "Unbatched (C, H, W) also works",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(3, 8, 8)\n"
                "out = {fn}(x, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))\n"
                "assert out.shape == x.shape, f'Shape: {out.shape}'\n"
                "ref = (x - torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)) / 0.25\n"
                "assert torch.allclose(out, ref, atol=1e-6), 'Value mismatch on unbatched input'\n"
            ),
        },
        {
            "name": "Accepts list / tuple / tensor for mean & std",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(1, 3, 4, 4)\n"
                "out_tup = {fn}(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n"
                "out_lst = {fn}(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])\n"
                "out_ten = {fn}(x, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]))\n"
                "assert torch.allclose(out_tup, out_lst, atol=1e-6), 'list vs tuple mismatch'\n"
                "assert torch.allclose(out_tup, out_ten, atol=1e-6), 'tensor vs tuple mismatch'\n"
            ),
        },
        {
            "name": "Gradient flows to x",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(1, 3, 4, 4, requires_grad=True)\n"
                "out = {fn}(x, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))\n"
                "out.sum().backward()\n"
                "assert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'\n"
                "expected = (1.0 / 0.25)\n"
                "assert torch.allclose(x.grad, torch.full_like(x, expected), atol=1e-5), f'd/dx should be 1/std = {expected}'\n"
            ),
        },
    ],
}
