"""B4 — Top-k Accuracy (ImageNet top-5、CIFAR-10 top-1 評価)."""

PROBLEM = {
    "id": "top_k_accuracy",
    "number": 52,
    "title": "Top-k Accuracy",
    "difficulty": "Easy",
    "fn_name": "top_k_accuracy",

    "intro_md": (
        "Implement **top-k accuracy** — the fraction of samples where the true label is among the "
        "model's top-k predictions. Standard for ImageNet (top-1 / top-5) and most evaluation pipelines.\n\n"
        "$$\\text{acc@k} = \\frac{1}{B} \\sum_b \\mathbb{1}[y_b \\in \\text{top-k}(\\text{logits}_b)]$$\n\n"
        "### Why top-k\n"
        "- **top-1**: strict — what most people mean by \"accuracy\"\n"
        "- **top-5**: more forgiving for fine-grained tasks (ImageNet has many similar dog breeds)\n"
        "- top-5 is always ≥ top-1 (monotone in k)\n"
    ),

    "signature": (
        "def top_k_accuracy(logits, targets, k=1):\n"
        "    # logits: (B, K_classes)\n"
        "    # targets: (B,) class indices\n"
        "    # k: int\n"
        "    # Returns: Python float in [0, 1]"
    ),

    "rules": [
        "Use `tensor.topk(k, dim=-1)` to get the top-k indices",
        "A sample is correct if `targets[b]` matches **any** of the top-k indices",
        "Return as a Python float (call `.item()`) — directly logable",
        "Clamp `k` to `num_classes` so `k > K` doesn't error (always 100% in that case)",
    ],

    "imports": "import torch",

    "template_body": (
        "def top_k_accuracy(logits, targets, k=1):\n"
        "    pass  # topk indices, compare against targets.unsqueeze(-1), .any() over k, mean.item()"
    ),

    "solution_body": (
        "def top_k_accuracy(logits, targets, k=1):\n"
        "    k = min(k, logits.size(-1))\n"
        "    _, topk_indices = logits.topk(k, dim=-1)\n"
        "    # topk_indices: (B, k); compare with targets broadcast to (B, 1)\n"
        "    correct = (topk_indices == targets.unsqueeze(-1)).any(dim=-1)\n"
        "    return correct.float().mean().item()"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "logits = torch.randn(100, 10)\n"
        "targets = torch.randint(0, 10, (100,))\n"
        "print(f'top-1 acc: {top_k_accuracy(logits, targets, k=1):.3f}')\n"
        "print(f'top-5 acc: {top_k_accuracy(logits, targets, k=5):.3f}')"
    ),

    "hint": (
        "`logits.topk(k, dim=-1)` returns `(values, indices)`. Compare `indices` (shape `(B, k)`) "
        "with `targets.unsqueeze(-1)` (shape `(B, 1)`) — `.any(dim=-1)` then `.float().mean().item()` "
        "gives the accuracy as a float."
    ),

    "tests": [
        {
            "name": "All correct → 1.0",
            "code": (
                "\nimport torch\n"
                "logits = torch.eye(10) * 100\n"
                "targets = torch.arange(10)\n"
                "acc = {fn}(logits, targets, k=1)\n"
                "assert abs(float(acc) - 1.0) < 1e-6, f'Expected 1.0, got {acc}'\n"
            ),
        },
        {
            "name": "All wrong → 0.0",
            "code": (
                "\nimport torch\n"
                "logits = torch.zeros(4, 5)\n"
                "logits[:, 0] = 10.0  # everyone predicts class 0\n"
                "targets = torch.tensor([1, 2, 3, 4])\n"
                "acc = {fn}(logits, targets, k=1)\n"
                "assert abs(float(acc) - 0.0) < 1e-6, f'Expected 0.0, got {acc}'\n"
            ),
        },
        {
            "name": "top-5 ≥ top-1 (monotone in k)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "logits = torch.randn(50, 10)\n"
                "targets = torch.randint(0, 10, (50,))\n"
                "acc1 = float({fn}(logits, targets, k=1))\n"
                "acc5 = float({fn}(logits, targets, k=5))\n"
                "assert acc5 >= acc1, f'top-5 ({acc5}) must be ≥ top-1 ({acc1})'\n"
            ),
        },
        {
            "name": "k ≥ num_classes → 1.0",
            "code": (
                "\nimport torch\n"
                "logits = torch.randn(8, 5)\n"
                "targets = torch.randint(0, 5, (8,))\n"
                "acc = {fn}(logits, targets, k=10)\n"
                "assert abs(float(acc) - 1.0) < 1e-6, f'k≥K should give 1.0, got {acc}'\n"
            ),
        },
        {
            "name": "Matches manual top-k count",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "B, K, k = 20, 10, 3\n"
                "logits = torch.randn(B, K)\n"
                "targets = torch.randint(0, K, (B,))\n"
                "_, top_idx = logits.topk(k, dim=-1)\n"
                "manual = (top_idx == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()\n"
                "out = float({fn}(logits, targets, k=k))\n"
                "assert abs(out - manual) < 1e-6, f'Got {out}, manual {manual}'\n"
            ),
        },
    ],
}
