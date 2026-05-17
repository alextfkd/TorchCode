"""B0 — 2D Max Pooling (Conv2d #22 の自然な相方、CNN block 後の定石 down-sample)."""

PROBLEM = {
    "id": "maxpool2d",
    "number": 41,
    "title": "2D Max Pooling",
    "difficulty": "Medium",
    "fn_name": "my_max_pool2d",

    "memo": (
        "Conv 後の標準的な down-sampling。最大値を残して空間サイズを半分にする。"
    ),
    "details": (
        "VGG 系で頻出。最近のアーキテクチャ (ResNet 後期、ViT) は stride 付き conv で代替する設計も多い。\n\n落とし穴: padding は `-inf` で（0 だと負入力で誤る）、勾配は argmax 位置にしか流れない。"
    ),
    "intro_md": (
        "**2D Max Pooling** を実装する。Conv2d (#22) の自然な相方で、ほぼ全 CNN block の後に "
        "来る down-sampling layer。\n\n"
        "$$\\text{out}[b, c, i, j] = \\max_{(p,q) \\in \\text{window}(i,j)} x[b, c, p, q]$$\n\n"
        "### AvgPool との違い\n"
        "- max は **non-smooth**（勾配は argmax 位置にだけ流れる）\n"
        "- Padding は **`-inf`** で行う — 0 だと負入力で誤る\n"
    ),

    "signature": (
        "def my_max_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    # x: (B, C, H, W)\n"
        "    # Returns: (B, C, H_out, W_out)"
    ),

    "rules": [
        "`F.max_pool2d`, `nn.MaxPool2d`, `F.adaptive_max_pool2d` は **使わない**",
        "`unfold`, `reshape`, `torch.max`/`amax` は許可",
        "`stride` の default は `kernel_size` (non-overlapping windows)",
        "`padding > 0` の場合、`float('-inf')` で pad — 0 だと negative input が pad に dominate される",
        "勾配は各 window の argmax 位置にのみ流れる",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def my_max_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    pass  # unfold で patch 抽出 → window dim で max"
    ),

    "solution_body": (
        "def my_max_pool2d(x, kernel_size, stride=None, padding=0):\n"
        "    if stride is None:\n"
        "        stride = kernel_size\n"
        "    if padding > 0:\n"
        "        x = F.pad(x, [padding] * 4, value=float('-inf'))\n"
        "    # Sliding window 抽出: (B, C, H_out, W_out, kH, kW)\n"
        "    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)\n"
        "    # window 次元で max\n"
        "    return patches.amax(dim=(-1, -2))"
    ),

    "demo_code": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "torch.manual_seed(0)\n"
        "\n"
        "x = torch.randn(1, 1, 4, 4)\n"
        "print('Input:\\n', x.squeeze())\n"
        "print('Output:\\n', my_max_pool2d(x, kernel_size=2).squeeze())\n"
        "print('Ref:\\n', F.max_pool2d(x, kernel_size=2).squeeze())"
    ),

    "hint": (
        "`x.unfold(2, k, s).unfold(3, k, s)` で patch を取って shape "
        "`(B, C, H_out, W_out, kH, kW)`、最後の 2 dim で max。padding は `float('-inf')` で "
        "（NOT 0 — negative input が pad に負けるため）。"
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
            "name": "Matches F.max_pool2d",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(2, 4, 8, 8)\n"
                "out = {fn}(x, kernel_size=2)\n"
                "ref = torch.nn.functional.max_pool2d(x, kernel_size=2)\n"
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
                "ref = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)\n"
                "assert out.shape == ref.shape, f'Shape: {out.shape} vs {ref.shape}'\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
            ),
        },
        {
            "name": "Padding with -inf (negative inputs)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = -torch.rand(1, 1, 4, 4) - 0.1\n"
                "out = {fn}(x, kernel_size=3, stride=1, padding=1)\n"
                "ref = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)\n"
                "assert out.shape == ref.shape, f'Shape: {out.shape} vs {ref.shape}'\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Padding must use -inf (not 0). Max diff: {(out-ref).abs().max():.6f}'\n"
            ),
        },
        {
            "name": "Gradient routes to argmax only",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(1, 1, 4, 4, requires_grad=True)\n"
                "out = {fn}(x, kernel_size=2)\n"
                "out.sum().backward()\n"
                "assert x.grad is not None, 'Missing gradient'\n"
                "nonzero = (x.grad != 0).sum().item()\n"
                "assert nonzero == 4, f'Expected exactly 4 non-zero grads (one per 2x2 window), got {nonzero}'\n"
            ),
        },
    ],
}
