"""A5 — Mixup (Zhang et al. 2017、W5 advanced レシピ常連)."""

PROBLEM = {
    "id": "mixup",
    "number": 46,
    "title": "Mixup",
    "difficulty": "Medium",
    "fn_name": "mixup",

    "intro_md": (
        "Implement **Mixup** (Zhang et al. 2017) — linearly interpolate pairs of examples and their "
        "labels. A frequent component of top CIFAR-10 / ImageNet recipes.\n\n"
        "$$\\tilde{x} = \\lambda x + (1-\\lambda) x_{\\text{perm}}, \\quad \\lambda \\sim \\text{Beta}(\\alpha, \\alpha)$$\n\n"
        "### The 4-tuple interface\n"
        "The standard PyTorch interface returns both labels and the mixing coefficient, so the loss "
        "can be computed externally:\n"
        "```python\n"
        "x_mix, y_a, y_b, lam = mixup(x, y)\n"
        "loss = lam * CE(model(x_mix), y_a) + (1 - lam) * CE(model(x_mix), y_b)\n"
        "```"
    ),

    "signature": (
        "def mixup(x, y, alpha=1.0):\n"
        "    # x: (B, ...) inputs\n"
        "    # y: (B,) long class indices\n"
        "    # alpha: Beta distribution parameter (>0)\n"
        "    # Returns: (x_mixed, y_a, y_b, lam)"
    ),

    "rules": [
        "Sample `λ ~ Beta(α, α)` — use `torch.distributions.Beta(α, α).sample().item()`",
        "Permute the batch with `torch.randperm(B)`",
        "Return a **4-tuple** `(x_mixed, y_a, y_b, lam)`",
        "`y_a` is the original `y`, `y_b` is `y[perm]`",
        "`x_mixed = lam * x + (1 - lam) * x[perm]`",
    ],

    "imports": "import torch",

    "template_body": (
        "def mixup(x, y, alpha=1.0):\n"
        "    pass  # sample lam ~ Beta(α,α), perm = randperm(B), then mix x and y"
    ),

    "solution_body": (
        "def mixup(x, y, alpha=1.0):\n"
        "    lam = torch.distributions.Beta(alpha, alpha).sample().item()\n"
        "    B = x.shape[0]\n"
        "    perm = torch.randperm(B, device=x.device)\n"
        "    x_mixed = lam * x + (1 - lam) * x[perm]\n"
        "    return x_mixed, y, y[perm], lam"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(4, 3, 8, 8)\n"
        "y = torch.tensor([0, 1, 2, 3])\n"
        "x_mix, y_a, y_b, lam = mixup(x, y, alpha=1.0)\n"
        "print('x_mix shape:', x_mix.shape, '| lam =', round(lam, 3))\n"
        "print('y_a:', y_a.tolist(), '| y_b:', y_b.tolist())"
    ),

    "hint": (
        "Use `torch.distributions.Beta(alpha, alpha).sample().item()` for λ as a Python float. "
        "`torch.randperm(B)` gives the shuffle. Return `(x_mixed, y, y[perm], lam)` — the loss is "
        "computed externally as `λ*CE(pred, y_a) + (1-λ)*CE(pred, y_b)`."
    ),

    "tests": [
        {
            "name": "Output is a 4-tuple with correct shapes",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "x = torch.randn(4, 3, 8, 8)\n"
                "y = torch.arange(4)\n"
                "out = {fn}(x, y, alpha=1.0)\n"
                "assert len(out) == 4, f'Expected 4-tuple, got {len(out)}-tuple'\n"
                "x_mix, y_a, y_b, lam = out\n"
                "assert x_mix.shape == x.shape, f'x_mix shape {x_mix.shape}'\n"
                "assert y_a.shape == y.shape and y_b.shape == y.shape, 'label shapes wrong'\n"
            ),
        },
        {
            "name": "y_a equals original y",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "y = torch.tensor([5, 7, 2, 9])\n"
                "x = torch.randn(4, 3)\n"
                "_, y_a, _, _ = {fn}(x, y, alpha=1.0)\n"
                "assert torch.equal(y_a, y), f'y_a {y_a.tolist()} != y {y.tolist()}'\n"
            ),
        },
        {
            "name": "y_b is a permutation of y",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "y = torch.tensor([5, 7, 2, 9])\n"
                "x = torch.randn(4, 3)\n"
                "_, _, y_b, _ = {fn}(x, y, alpha=1.0)\n"
                "assert sorted(y_b.tolist()) == sorted(y.tolist()), f'y_b not a permutation: {y_b.tolist()}'\n"
            ),
        },
        {
            "name": "lam is a Python float in [0, 1]",
            "code": (
                "\nimport torch\n"
                "for seed in range(5):\n"
                "    torch.manual_seed(seed)\n"
                "    x = torch.randn(4, 3)\n"
                "    y = torch.arange(4)\n"
                "    _, _, _, lam = {fn}(x, y, alpha=1.0)\n"
                "    assert isinstance(lam, float), f'lam should be float, got {type(lam).__name__}'\n"
                "    assert 0.0 <= lam <= 1.0, f'lam {lam} out of [0,1]'\n"
            ),
        },
        {
            "name": "x_mix == lam*x + (1-lam)*x[perm] (perm inferred from y_b)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "B = 8\n"
                "x = torch.randn(B, 3, 4, 4)\n"
                "y = torch.arange(B)  # so y_b == perm directly\n"
                "x_mix, y_a, y_b, lam = {fn}(x, y, alpha=1.0)\n"
                "perm = y_b\n"
                "expected = lam * x + (1 - lam) * x[perm]\n"
                "assert torch.allclose(x_mix, expected, atol=1e-5), f'Max diff: {(x_mix-expected).abs().max():.6f}'\n"
            ),
        },
    ],
}
