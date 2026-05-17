"""A3 — Random Crop with Padding (CIFAR-10 pad=4, crop=32 の定石)."""

PROBLEM = {
    "id": "random_crop",
    "number": 44,
    "title": "Random Crop with Padding",
    "difficulty": "Easy",
    "fn_name": "random_crop",

    "memo": (
        "CIFAR-10 学習の定石。4 pixel zero pad → 32×32 random crop で平行移動 invariance を獲得。"
    ),
    "details": (
        "ImageNet では `RandomResizedCrop`（scale も変える版）が標準で、より強い augmentation。\n\nTest 時には center crop で固定。Crop と Normalize (#42) は Compose で並べて、Normalize は最後 (crop 領域の値を正規化)。"
    ),
    "intro_md": (
        "**Random Crop with Zero Padding** を実装する。CIFAR-10 の定石: 4 pixel 分 zero pad "
        "してから 32×32 に random crop。\n\n"
        "計算コストほぼゼロで学習データに平行移動 invariance を追加できる。\n\n"
        "### Example\n```\n"
        "x: (3, 32, 32), padding=4 → (3, 40, 40) に pad → random offset で (3, 32, 32) 切り出し\n"
        "```"
    ),

    "signature": (
        "def random_crop(x, size, padding=0):\n"
        "    # x: (C, H, W) or (B, C, H, W)\n"
        "    # size: int — output crop size (square)\n"
        "    # padding: int — zero-pad on all sides before cropping\n"
        "    # Returns: (C, size, size) or (B, C, size, size)"
    ),

    "rules": [
        "`torchvision.transforms.RandomCrop` は **使わない**",
        "最初に zero pad (`F.pad`) してから `size × size` で random crop",
        "`(C, H, W)` と `(B, C, H, W)` 両方をサポート",
        "batch 入力では各 sample が **独立** な offset で crop",
        "`size` は int 1個（正方形 output）",
    ],

    "imports": "import torch\nimport torch.nn.functional as F",

    "template_body": (
        "def random_crop(x, size, padding=0):\n"
        "    pass  # F.pad → (i, j) sampling → slice x[..., i:i+size, j:j+size]"
    ),

    "solution_body": (
        "def random_crop(x, size, padding=0):\n"
        "    if padding > 0:\n"
        "        x = F.pad(x, [padding] * 4)\n"
        "    if x.dim() == 3:\n"
        "        _, H, W = x.shape\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        return x[:, i:i+size, j:j+size]\n"
        "    # (B, C, H, W) — per-sample offset\n"
        "    B = x.shape[0]\n"
        "    H, W = x.shape[-2], x.shape[-1]\n"
        "    out = torch.empty(B, x.shape[1], size, size, dtype=x.dtype, device=x.device)\n"
        "    for b in range(B):\n"
        "        i = torch.randint(0, H - size + 1, (1,)).item()\n"
        "        j = torch.randint(0, W - size + 1, (1,)).item()\n"
        "        out[b] = x[b, :, i:i+size, j:j+size]\n"
        "    return out"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.rand(4, 3, 32, 32)\n"
        "out = random_crop(x, size=32, padding=4)\n"
        "print('In:', x.shape, '→ Out:', out.shape)"
    ),

    "hint": (
        "`F.pad` の後、`(i, j)` を `[0, H_padded - size + 1)` から uniform にサンプリングして "
        "`x[..., i:i+size, j:j+size]` で切る。batch の場合は **sample ごとに** `(i, j)` をサンプリング、"
        "各 sample が違う crop になるようにする。"
    ),

    "tests": [
        {
            "name": "Output shape (B, C, size, size)",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(4, 3, 32, 32)\n"
                "out = {fn}(x, size=32, padding=4)\n"
                "assert out.shape == (4, 3, 32, 32), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Unbatched (C, H, W) also works",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(3, 16, 16)\n"
                "out = {fn}(x, size=12)\n"
                "assert out.shape == (3, 12, 12), f'Shape: {out.shape}'\n"
            ),
        },
        {
            "name": "Identity when size == H and padding == 0",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(2, 3, 8, 8)\n"
                "out = {fn}(x, size=8, padding=0)\n"
                "assert torch.allclose(out, x), 'No-op crop must equal input'\n"
            ),
        },
        {
            "name": "Determinism with seed",
            "code": (
                "\nimport torch\n"
                "x = torch.rand(1, 3, 16, 16)\n"
                "torch.manual_seed(0)\n"
                "a = {fn}(x, size=12, padding=0)\n"
                "torch.manual_seed(0)\n"
                "b = {fn}(x, size=12, padding=0)\n"
                "assert torch.allclose(a, b), 'Same seed must give same crop'\n"
            ),
        },
        {
            "name": "Randomness — different seeds give different crops",
            "code": (
                "\nimport torch\n"
                "x = torch.arange(1*1*20*20).float().view(1, 1, 20, 20)\n"
                "torch.manual_seed(0)\n"
                "a = {fn}(x, size=5, padding=0)\n"
                "torch.manual_seed(1)\n"
                "b = {fn}(x, size=5, padding=0)\n"
                "assert not torch.allclose(a, b), 'Different seeds should give different crops'\n"
            ),
        },
    ],
}
