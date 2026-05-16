"""B5 — NLL Loss (CrossEntropy = log_softmax + NLL の後半を分解)."""

PROBLEM = {
    "id": "nll_loss",
    "number": 53,
    "title": "Negative Log-Likelihood Loss",
    "difficulty": "Easy",
    "fn_name": "my_nll_loss",

    "intro_md": (
        "Implement **NLL loss** — the second half of cross-entropy. The input is **already** "
        "`log_softmax`'d probabilities; this function picks the log-prob at each target class "
        "and averages the negative.\n\n"
        "$$L = -\\frac{1}{B} \\sum_b \\log p_b[y_b]$$\n\n"
        "### Why PyTorch separates `log_softmax` + `nll_loss`\n"
        "- Reuse the same `log_softmax` output for both loss and analysis\n"
        "- Apply NLL to log-probabilities from anywhere (mixture models, normalizing flows)\n"
        "- `CrossEntropyLoss(logits, targets) == nll_loss(log_softmax(logits), targets)`\n"
    ),

    "signature": (
        "def my_nll_loss(log_probs, targets):\n"
        "    # log_probs: (B, K) — already log_softmax'd\n"
        "    # targets: (B,) long\n"
        "    # Returns: scalar loss (mean over batch)"
    ),

    "rules": [
        "Do **NOT** use `F.nll_loss` or `nn.NLLLoss`",
        "Input is already `log_softmax` — don't recompute softmax",
        "Mean reduction (scalar output)",
        "Use advanced indexing: `log_probs[arange(B), targets]`",
    ],

    "imports": "import torch",

    "template_body": (
        "def my_nll_loss(log_probs, targets):\n"
        "    pass  # gather log_probs[arange(B), targets], negate, mean"
    ),

    "solution_body": (
        "def my_nll_loss(log_probs, targets):\n"
        "    B = log_probs.size(0)\n"
        "    selected = log_probs[torch.arange(B, device=log_probs.device), targets]\n"
        "    return -selected.mean()"
    ),

    "demo_code": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "torch.manual_seed(0)\n"
        "logits = torch.randn(4, 10)\n"
        "log_probs = F.log_softmax(logits, dim=-1)\n"
        "targets = torch.tensor([3, 1, 7, 0])\n"
        "print('NLL: ', my_nll_loss(log_probs, targets).item())\n"
        "print('Ref: ', F.nll_loss(log_probs, targets).item())"
    ),

    "hint": (
        "Use advanced indexing: `log_probs[torch.arange(B), targets]` gives the log-prob at the "
        "correct class for each sample. Negate and take mean."
    ),

    "tests": [
        {
            "name": "Returns scalar",
            "code": (
                "\nimport torch\n"
                "import torch.nn.functional as F\n"
                "log_probs = F.log_softmax(torch.randn(4, 10), dim=-1)\n"
                "targets = torch.randint(0, 10, (4,))\n"
                "loss = {fn}(log_probs, targets)\n"
                "assert loss.dim() == 0, f'Expected scalar, got shape {loss.shape}'\n"
            ),
        },
        {
            "name": "Matches F.nll_loss",
            "code": (
                "\nimport torch\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "log_probs = F.log_softmax(torch.randn(8, 5), dim=-1)\n"
                "targets = torch.randint(0, 5, (8,))\n"
                "out = {fn}(log_probs, targets)\n"
                "ref = F.nll_loss(log_probs, targets)\n"
                "assert torch.allclose(out, ref, atol=1e-6), f'Got {out.item()}, expected {ref.item()}'\n"
            ),
        },
        {
            "name": "Equivalent to CrossEntropy when log_probs = log_softmax(logits)",
            "code": (
                "\nimport torch\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "logits = torch.randn(16, 7)\n"
                "targets = torch.randint(0, 7, (16,))\n"
                "out = {fn}(F.log_softmax(logits, dim=-1), targets)\n"
                "ref = F.cross_entropy(logits, targets)\n"
                "assert torch.allclose(out, ref, atol=1e-5), 'NLL∘log_softmax should equal CE'\n"
            ),
        },
        {
            "name": "Gradient flows back to log_probs",
            "code": (
                "\nimport torch\n"
                "log_probs = torch.randn(4, 10).log_softmax(dim=-1).detach().requires_grad_()\n"
                "targets = torch.randint(0, 10, (4,))\n"
                "loss = {fn}(log_probs, targets)\n"
                "loss.backward()\n"
                "assert log_probs.grad is not None and log_probs.grad.shape == log_probs.shape\n"
            ),
        },
        {
            "name": "Per-sample indexing correctness (handcrafted)",
            "code": (
                "\nimport torch\n"
                "log_probs = torch.tensor([\n"
                "    [-0.1, -10.0, -10.0],\n"
                "    [-10.0, -0.2, -10.0],\n"
                "    [-10.0, -10.0, -0.3],\n"
                "])\n"
                "targets = torch.tensor([0, 1, 2])\n"
                "out = {fn}(log_probs, targets)\n"
                "expected = -(-0.1 + -0.2 + -0.3) / 3\n"
                "assert abs(out.item() - expected) < 1e-6, f'Got {out.item()}, expected {expected}'\n"
            ),
        },
    ],
}
