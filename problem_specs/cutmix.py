"""A6 — CutMix (Yun et al. 2019、Mixup の幾何版・面積比 λ で interview 頻出)."""

PROBLEM = {
    "id": "cutmix",
    "number": 47,
    "title": "CutMix",
    "difficulty": "Medium",
    "fn_name": "cutmix",

    "intro_md": (
        "**CutMix** (Yun et al. 2019) を実装する。Mixup と違って pixel 値を blend するのではなく、"
        "ある画像から矩形を **切り取って貼る**。label は paste 領域の面積比で混ぜる。\n\n"
        "### area-based λ (interview trap)\n"
        "矩形はランダムな中心位置に置かれて、**画像境界でクリップされる**ことがある。なので "
        "Beta(α,α) からサンプルした λ と、**実際の**面積比は一致しない。clipping 後の実面積から "
        "λ を再計算してから return すること — interview で grill される所。\n"
    ),

    "signature": (
        "def cutmix(x, y, alpha=1.0):\n"
        "    # x: (B, C, H, W) image batch\n"
        "    # y: (B,) class indices\n"
        "    # alpha: Beta distribution parameter\n"
        "    # Returns: (x_mixed, y_a, y_b, lam)\n"
        "    #   lam = 1 - (actual cut area) / (H*W) — recomputed after clipping"
    ),

    "rules": [
        "`λ ~ Beta(α, α)` をサンプル、cut size は `cut_h = int(H * sqrt(1 - λ))`, 同様に `cut_w`",
        "中心 `(cy, cx)` を `[0, H) × [0, W)` から uniform サンプル",
        "矩形を画像 bounds で clip: `y1 = max(0, cy - cut_h//2)` など",
        "`x[perm, :, y1:y2, x1:x2]` を `x_mixed` の同じ slice にコピー",
        "**λ を再計算** `1 - (y2-y1)*(x2-x1) / (H*W)` してから return",
    ],

    "imports": "import torch",

    "template_body": (
        "def cutmix(x, y, alpha=1.0):\n"
        "    pass  # lam サンプル、cut size 導出、中心サンプリング、画像 bounds で clip、\n"
        "          # x[perm] を paste、最後に実面積から lam を再計算"
    ),

    "solution_body": (
        "def cutmix(x, y, alpha=1.0):\n"
        "    B, C, H, W = x.shape\n"
        "    lam = torch.distributions.Beta(alpha, alpha).sample().item()\n"
        "    perm = torch.randperm(B, device=x.device)\n"
        "    cut_ratio = (1 - lam) ** 0.5\n"
        "    cut_h = int(H * cut_ratio)\n"
        "    cut_w = int(W * cut_ratio)\n"
        "    cy = torch.randint(0, H, (1,)).item()\n"
        "    cx = torch.randint(0, W, (1,)).item()\n"
        "    y1 = max(0, cy - cut_h // 2)\n"
        "    y2 = min(H, cy + cut_h // 2)\n"
        "    x1 = max(0, cx - cut_w // 2)\n"
        "    x2 = min(W, cx + cut_w // 2)\n"
        "    x_mixed = x.clone()\n"
        "    x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]\n"
        "    # 実現された面積から lam を再計算\n"
        "    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)\n"
        "    return x_mixed, y, y[perm], lam"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.zeros(2, 1, 16, 16)\n"
        "x[1] = 1.0\n"
        "y = torch.tensor([0, 1])\n"
        "x_mix, y_a, y_b, lam = cutmix(x, y, alpha=1.0)\n"
        "print('lam =', round(lam, 3))\n"
        "print('Ones in sample 0:', (x_mix[0] == 1).sum().item(), '(should be (1-lam)*256 if perm swapped)')"
    ),

    "hint": (
        "`cut_h = int(H * sqrt(1-λ))`。中心を uniform サンプリングして "
        "`y1=max(0, cy-cut_h//2)` などで clip。`x[perm, :, y1:y2, x1:x2]` を `x_mixed` の同じ "
        "slice に paste。**λ を再計算**してから return — 境界 clipping を考慮するため。"
    ),

    "tests": [
        {
            "name": "Output is a 4-tuple with correct shapes",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(4, 3, 16, 16)\n"
                "y = torch.arange(4)\n"
                "out = {fn}(x, y, alpha=1.0)\n"
                "assert len(out) == 4\n"
                "x_mix, y_a, y_b, lam = out\n"
                "assert x_mix.shape == x.shape\n"
                "assert y_a.shape == y.shape and y_b.shape == y.shape\n"
            ),
        },
        {
            "name": "y_a is original, y_b is a permutation of y",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "y = torch.tensor([5, 7, 2, 9])\n"
                "x = torch.randn(4, 3, 8, 8)\n"
                "_, y_a, y_b, _ = {fn}(x, y, alpha=1.0)\n"
                "assert torch.equal(y_a, y), 'y_a must equal y'\n"
                "assert sorted(y_b.tolist()) == sorted(y.tolist()), 'y_b not a permutation'\n"
            ),
        },
        {
            "name": "lam is float in [0, 1]",
            "code": (
                "\nimport torch\n"
                "for seed in range(8):\n"
                "    torch.manual_seed(seed)\n"
                "    x = torch.randn(4, 3, 16, 16)\n"
                "    y = torch.arange(4)\n"
                "    _, _, _, lam = {fn}(x, y, alpha=1.0)\n"
                "    assert isinstance(lam, float), f'lam should be float, got {type(lam).__name__}'\n"
                "    assert 0.0 <= lam <= 1.0, f'lam {lam} out of [0, 1]'\n"
            ),
        },
        {
            "name": "lam matches actual cut area (recomputed after clipping)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "B, C, H, W = 2, 1, 32, 32\n"
                "x = torch.zeros(B, C, H, W)\n"
                "x[1] = 1.0\n"
                "y = torch.tensor([0, 1])\n"
                "x_mix, y_a, y_b, lam = {fn}(x, y, alpha=1.0)\n"
                "if y_b[0].item() == 1:\n"
                "    # sample 0 received pasted region from sample 1 (ones)\n"
                "    ones = (x_mix[0] == 1).sum().item()\n"
                "    expected_cut = (1.0 - lam) * H * W\n"
                "    assert abs(ones - expected_cut) < 1.0, f'Cut area {ones} != (1-lam)*H*W = {expected_cut:.1f}'\n"
                "else:\n"
                "    assert (x_mix[0] == 0).all(), 'When perm[0]=0, sample 0 should be unchanged'\n"
            ),
        },
        {
            "name": "Pixels are either from x or x[perm] — no blending",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(4, 3, 16, 16)\n"
                "y = torch.arange(4)\n"
                "x_orig = x.clone()\n"
                "x_mix, _, y_b, _ = {fn}(x, y, alpha=1.0)\n"
                "perm = y_b\n"
                "is_from_x = (x_mix == x_orig)\n"
                "is_from_perm = (x_mix == x_orig[perm])\n"
                "assert (is_from_x | is_from_perm).all(), 'Every pixel must come from x or x[perm], not be blended'\n"
            ),
        },
    ],
}
