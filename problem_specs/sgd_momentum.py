"""B6 — SGD with Momentum (Adam の前に押さえる古典、CNN で依然強い)."""

PROBLEM = {
    "id": "sgd_momentum",
    "number": 54,
    "title": "SGD with Momentum",
    "difficulty": "Medium",
    "fn_name": "MySGDMomentum",

    "intro_md": (
        "Implement **SGD with Momentum** — the optimizer everyone learns first, still competitive "
        "with Adam for many vision tasks (ResNet, VGG).\n\n"
        "### Update rule (PyTorch convention — no `(1-μ)` factor on g)\n"
        "$$v_{t+1} = \\mu \\cdot v_t + g_t$$\n"
        "$$p_{t+1} = p_t - \\text{lr} \\cdot v_{t+1}$$\n\n"
        "Watch out: the classical Heavy-Ball / EMA form uses `μ*v + (1-μ)*g`. PyTorch's `torch.optim.SGD` "
        "uses **no `(1-μ)` factor**. Match PyTorch's convention so the matching test passes.\n"
    ),

    "signature": (
        "class MySGDMomentum:\n"
        "    def __init__(self, params, lr, momentum=0.9): ...\n"
        "    def zero_grad(self): ...\n"
        "    def step(self): ..."
    ),

    "rules": [
        "Do **NOT** use `torch.optim.SGD`",
        "Implement `step()` and `zero_grad()` methods",
        "Velocity update: `v = momentum * v + p.grad` (PyTorch convention — no `(1-μ)` factor)",
        "Param update: `p -= lr * v`",
        "Initialize velocity buffers to zeros (one per param)",
        "Wrap `step()` in `@torch.no_grad()` (or manual context)",
    ],

    "imports": "import torch",

    "template_body": (
        "class MySGDMomentum:\n"
        "    def __init__(self, params, lr, momentum=0.9):\n"
        "        pass  # store params/lr/momentum, init velocity buffers as zeros_like(p)\n"
        "    \n"
        "    def zero_grad(self):\n"
        "        pass  # for each param, zero out p.grad (if not None)\n"
        "    \n"
        "    def step(self):\n"
        "        pass  # for each (p, v): v = μ*v + p.grad; p -= lr * v"
    ),

    "solution_body": (
        "class MySGDMomentum:\n"
        "    def __init__(self, params, lr, momentum=0.9):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "        self.momentum = momentum\n"
        "        self.velocities = [torch.zeros_like(p) for p in self.params]\n"
        "    \n"
        "    def zero_grad(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is not None:\n"
        "                p.grad.detach_()\n"
        "                p.grad.zero_()\n"
        "    \n"
        "    @torch.no_grad()\n"
        "    def step(self):\n"
        "        for p, v in zip(self.params, self.velocities):\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            v.mul_(self.momentum).add_(p.grad)\n"
        "            p.add_(v, alpha=-self.lr)"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "p = torch.randn(5, requires_grad=True)\n"
        "opt = MySGDMomentum([p], lr=0.1, momentum=0.9)\n"
        "for step in range(3):\n"
        "    loss = (p ** 2).sum()\n"
        "    loss.backward()\n"
        "    opt.step()\n"
        "    opt.zero_grad()\n"
        "    print(f'step {step}: |p| = {p.norm().item():.4f}')"
    ),

    "hint": (
        "Store params, lr, momentum, and velocity buffers (`zeros_like` each param). "
        "In `step()`: wrap with `@torch.no_grad()`. For each `(p, v)` pair: "
        "`v.mul_(momentum).add_(p.grad)` then `p.add_(v, alpha=-lr)`. "
        "Skip params with `p.grad is None`."
    ),

    "tests": [
        {
            "name": "Has step() and zero_grad() methods",
            "code": (
                "\nimport torch\n"
                "p = torch.randn(3, requires_grad=True)\n"
                "opt = {fn}([p], lr=0.01, momentum=0.9)\n"
                "assert hasattr(opt, 'step') and callable(opt.step), 'Missing step()'\n"
                "assert hasattr(opt, 'zero_grad') and callable(opt.zero_grad), 'Missing zero_grad()'\n"
            ),
        },
        {
            "name": "momentum=0 → plain SGD",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p1 = torch.randn(5, requires_grad=True)\n"
                "p2 = p1.detach().clone().requires_grad_()\n"
                "opt = {fn}([p2], lr=0.1, momentum=0.0)\n"
                "g = torch.ones(5)\n"
                "p1.grad = g.clone()\n"
                "p2.grad = g.clone()\n"
                "opt.step()\n"
                "expected = p1 - 0.1 * g\n"
                "assert torch.allclose(p2, expected, atol=1e-6), f'momentum=0 should be plain SGD'\n"
            ),
        },
        {
            "name": "Matches torch.optim.SGD with momentum over 5 steps",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p_ref = torch.randn(10, requires_grad=True)\n"
                "p_my = p_ref.detach().clone().requires_grad_()\n"
                "ref = torch.optim.SGD([p_ref], lr=0.05, momentum=0.9)\n"
                "my = {fn}([p_my], lr=0.05, momentum=0.9)\n"
                "for step in range(5):\n"
                "    p_ref.grad = (p_ref * 2).detach()\n"
                "    p_my.grad = (p_my * 2).detach()\n"
                "    ref.step()\n"
                "    my.step()\n"
                "diff = (p_ref - p_my).abs().max().item()\n"
                "assert diff < 1e-5, f'After 5 steps, max diff: {diff:.6f}'\n"
            ),
        },
        {
            "name": "Velocity accumulates (step 2 delta > 1.5x step 1 delta)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p = torch.randn(3, requires_grad=True)\n"
                "opt = {fn}([p], lr=0.1, momentum=0.9)\n"
                "g = torch.ones(3)\n"
                "# Step 1: v = g; |delta_1| = 0.1\n"
                "p_start = p.clone()\n"
                "p.grad = g.clone()\n"
                "opt.step()\n"
                "delta_1 = (p - p_start).abs().mean().item()\n"
                "p_mid = p.clone()\n"
                "# Step 2: v = 0.9*g + g = 1.9*g; |delta_2| = 0.19\n"
                "p.grad = g.clone()\n"
                "opt.step()\n"
                "delta_2 = (p - p_mid).abs().mean().item()\n"
                "assert delta_2 > delta_1 * 1.5, f'Step 2 delta ({delta_2:.4f}) should be ~1.9x step 1 ({delta_1:.4f})'\n"
            ),
        },
        {
            "name": "zero_grad clears gradients",
            "code": (
                "\nimport torch\n"
                "p = torch.randn(4, requires_grad=True)\n"
                "p.grad = torch.ones(4)\n"
                "opt = {fn}([p], lr=0.01, momentum=0.9)\n"
                "opt.zero_grad()\n"
                "assert (p.grad == 0).all(), f'zero_grad should zero out gradients: got {p.grad}'\n"
            ),
        },
    ],
}
