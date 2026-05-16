"""B7 — Weight Decay (L2 regularization、AdamW との対比に向けた下準備)."""

PROBLEM = {
    "id": "weight_decay",
    "number": 55,
    "title": "Weight Decay (L2 in gradient)",
    "difficulty": "Easy",
    "fn_name": "apply_weight_decay",

    "intro_md": (
        "Implement **weight decay** — the L2 regularization term folded into the gradient before "
        "the optimizer step. This is the SGD/L2 convention; AdamW uses a different (decoupled) "
        "formulation (see problem 56).\n\n"
        "### L2-in-gradient form (this problem)\n"
        "$$g \\leftarrow g + \\lambda \\cdot p$$\n\n"
        "Equivalent to adding `(λ/2)·‖p‖²` to the loss and letting autograd add `λ·p` to the "
        "gradient. Doing it directly in the optimizer step skips the extra graph node.\n\n"
        "### vs. AdamW's decoupled form\n"
        "$$p \\leftarrow p \\cdot (1 - \\text{lr} \\cdot \\lambda)$$\n"
        "Applied to the **param** directly, not via gradient. Loshchilov & Hutter (2019) showed "
        "this works better with adaptive optimizers.\n"
    ),

    "signature": (
        "def apply_weight_decay(params, weight_decay):\n"
        "    # params: a single tensor OR iterable of tensors\n"
        "    # weight_decay: float λ\n"
        "    # In-place: adds weight_decay * p to each p.grad\n"
        "    # Returns: None"
    ),

    "rules": [
        "**In-place**: modifies `p.grad` directly, no return value",
        "Accept a single tensor OR a list/iterable",
        "Skip params with `p.grad is None`",
        "When `weight_decay == 0`, do nothing",
        "Wrap in `@torch.no_grad()` — we're modifying `.grad`, not building a graph",
    ],

    "imports": "import torch",

    "template_body": (
        "def apply_weight_decay(params, weight_decay):\n"
        "    pass  # under no_grad: for each p with p.grad, do p.grad.add_(p, alpha=weight_decay)"
    ),

    "solution_body": (
        "@torch.no_grad()\n"
        "def apply_weight_decay(params, weight_decay):\n"
        "    if isinstance(params, torch.Tensor):\n"
        "        params = [params]\n"
        "    if weight_decay == 0:\n"
        "        return\n"
        "    for p in params:\n"
        "        if p.grad is not None:\n"
        "            p.grad.add_(p, alpha=weight_decay)"
    ),

    "demo_code": (
        "import torch\n"
        "p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "p.grad = torch.ones(3)\n"
        "apply_weight_decay(p, weight_decay=0.1)\n"
        "print('grad after WD (was ones, +0.1*p):', p.grad.tolist())"
    ),

    "hint": (
        "Single-tensor case: wrap in a list. `weight_decay=0` fast path. For each param: "
        "`p.grad.add_(p, alpha=weight_decay)` is equivalent to `p.grad += weight_decay * p`. "
        "Use `@torch.no_grad()` since we're modifying `.grad` outside autograd."
    ),

    "tests": [
        {
            "name": "Adds weight_decay * p to grad",
            "code": (
                "\nimport torch\n"
                "p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
                "p.grad = torch.ones(3)\n"
                "{fn}(p, 0.1)\n"
                "expected = torch.tensor([1.1, 1.2, 1.3])\n"
                "assert torch.allclose(p.grad, expected, atol=1e-6), f'Got {p.grad}, expected {expected}'\n"
            ),
        },
        {
            "name": "weight_decay=0 → no change",
            "code": (
                "\nimport torch\n"
                "p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
                "p.grad = torch.tensor([0.5, 0.6, 0.7])\n"
                "orig = p.grad.clone()\n"
                "{fn}(p, 0.0)\n"
                "assert torch.allclose(p.grad, orig), 'WD=0 should not change grad'\n"
            ),
        },
        {
            "name": "Works on a list of params",
            "code": (
                "\nimport torch\n"
                "p1 = torch.tensor([1., 2.], requires_grad=True); p1.grad = torch.zeros(2)\n"
                "p2 = torch.tensor([3., 4.], requires_grad=True); p2.grad = torch.zeros(2)\n"
                "{fn}([p1, p2], 0.5)\n"
                "assert torch.allclose(p1.grad, torch.tensor([0.5, 1.0]))\n"
                "assert torch.allclose(p2.grad, torch.tensor([1.5, 2.0]))\n"
            ),
        },
        {
            "name": "Skips params with grad=None",
            "code": (
                "\nimport torch\n"
                "p1 = torch.tensor([1., 2.], requires_grad=True); p1.grad = torch.zeros(2)\n"
                "p2 = torch.tensor([3., 4.], requires_grad=True)  # grad is None\n"
                "{fn}([p1, p2], 0.1)  # should not raise\n"
                "assert torch.allclose(p1.grad, torch.tensor([0.1, 0.2]))\n"
                "assert p2.grad is None, 'p2.grad should still be None'\n"
            ),
        },
        {
            "name": "Does NOT modify p itself (only p.grad)",
            "code": (
                "\nimport torch\n"
                "p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
                "p.grad = torch.zeros(3)\n"
                "p_orig = p.clone()\n"
                "{fn}(p, 0.5)\n"
                "assert torch.allclose(p, p_orig), 'p itself should NOT change — only p.grad'\n"
            ),
        },
    ],
}
