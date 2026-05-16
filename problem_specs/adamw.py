"""B8 — AdamW (Loshchilov & Hutter 2019, Transformer 時代の標準 optimizer)."""

PROBLEM = {
    "id": "adamw",
    "number": 56,
    "title": "AdamW (Decoupled Weight Decay)",
    "difficulty": "Hard",
    "fn_name": "MyAdamW",

    "intro_md": (
        "Implement **AdamW** (Loshchilov & Hutter 2019) — Adam with **decoupled weight decay**. "
        "The default modern choice for training Transformers.\n\n"
        "### The decoupled twist\n"
        "Plain Adam + L2 regularization adds `λp` to the gradient *before* the moment updates, "
        "which means the L2 term is divided by `√v_hat` — different weights get different effective "
        "decay rates. AdamW applies decay **directly to the parameter** after the Adam update, so "
        "every weight gets the same effective decay:\n\n"
        "$$p \\leftarrow p \\cdot (1 - \\text{lr} \\cdot \\lambda)$$\n"
        "$$p \\leftarrow p - \\text{lr} \\cdot \\hat{m} / (\\sqrt{\\hat{v}} + \\varepsilon)$$\n\n"
        "### Killer test (the interview line)\n"
        "With `weight_decay > 0` and **zero gradients**, AdamW still shrinks the params (decoupled "
        "WD acts on `p` regardless of `g`). Plain Adam + L2 would do nothing in that case.\n"
    ),

    "signature": (
        "class MyAdamW:\n"
        "    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01): ...\n"
        "    def zero_grad(self): ...\n"
        "    def step(self): ..."
    ),

    "rules": [
        "Do **NOT** use `torch.optim.AdamW` or `torch.optim.Adam`",
        "Apply decoupled WD: `p *= (1 - lr * weight_decay)` to the param (NOT the gradient)",
        "Then Adam: m, v EMAs with bias correction, then `p -= lr * m_hat / (sqrt(v_hat) + eps)`",
        "Initialize `m, v` to zeros and step counter `t = 0`; increment `t` at the start of each step",
        "Skip params with `p.grad is None` (do NOT apply WD to them either — matches torch.optim.AdamW)",
        "Wrap `step()` in `@torch.no_grad()`",
    ],

    "imports": "import torch",

    "template_body": (
        "class MyAdamW:\n"
        "    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):\n"
        "        pass  # store hyperparams; init m=[zeros], v=[zeros], t=0\n"
        "    \n"
        "    def zero_grad(self):\n"
        "        pass\n"
        "    \n"
        "    def step(self):\n"
        "        pass  # t += 1; for each (p, m, v) with p.grad is not None:\n"
        "              #   decoupled WD: p *= (1 - lr*wd)\n"
        "              #   m = β1*m + (1-β1)*g; v = β2*v + (1-β2)*g²\n"
        "              #   m_hat = m / (1 - β1^t); v_hat = v / (1 - β2^t)\n"
        "              #   p -= lr * m_hat / (sqrt(v_hat) + eps)"
    ),

    "solution_body": (
        "class MyAdamW:\n"
        "    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "        self.beta1, self.beta2 = betas\n"
        "        self.eps = eps\n"
        "        self.weight_decay = weight_decay\n"
        "        self.m = [torch.zeros_like(p) for p in self.params]\n"
        "        self.v = [torch.zeros_like(p) for p in self.params]\n"
        "        self.t = 0\n"
        "    \n"
        "    def zero_grad(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is not None:\n"
        "                p.grad.detach_()\n"
        "                p.grad.zero_()\n"
        "    \n"
        "    @torch.no_grad()\n"
        "    def step(self):\n"
        "        self.t += 1\n"
        "        bc1 = 1 - self.beta1 ** self.t\n"
        "        bc2 = 1 - self.beta2 ** self.t\n"
        "        for p, m, v in zip(self.params, self.m, self.v):\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            # Decoupled weight decay — applied to PARAM, not gradient\n"
        "            if self.weight_decay != 0:\n"
        "                p.mul_(1 - self.lr * self.weight_decay)\n"
        "            g = p.grad\n"
        "            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)\n"
        "            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)\n"
        "            m_hat = m / bc1\n"
        "            v_hat = v / bc2\n"
        "            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)"
    ),

    "demo_code": (
        "import torch\n"
        "torch.manual_seed(0)\n"
        "p = torch.randn(5, requires_grad=True)\n"
        "opt = MyAdamW([p], lr=0.01, weight_decay=0.1)\n"
        "for step in range(3):\n"
        "    loss = (p ** 2).sum()\n"
        "    loss.backward()\n"
        "    opt.step()\n"
        "    opt.zero_grad()\n"
        "    print(f'step {step}: |p| = {p.norm().item():.4f}')"
    ),

    "hint": (
        "**Order matters**: apply decoupled WD (`p *= (1 - lr*wd)`) BEFORE the Adam moment update. "
        "Increment `t` at the start of `step()`. Bias-correct with `1 - β^t`. "
        "Use `p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)` for the fused in-place update. "
        "Skip params with `grad is None` (do NOT apply WD to them either — that's what torch.optim.AdamW does)."
    ),

    "tests": [
        {
            "name": "Has step() and zero_grad() methods",
            "code": (
                "\nimport torch\n"
                "p = torch.randn(3, requires_grad=True)\n"
                "opt = {fn}([p], lr=0.01, weight_decay=0.01)\n"
                "assert hasattr(opt, 'step') and callable(opt.step)\n"
                "assert hasattr(opt, 'zero_grad') and callable(opt.zero_grad)\n"
            ),
        },
        {
            "name": "step() updates params (WD=0)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p = torch.randn(5, requires_grad=True)\n"
                "p_orig = p.clone()\n"
                "opt = {fn}([p], lr=0.01, weight_decay=0.0)\n"
                "p.grad = torch.ones(5)\n"
                "opt.step()\n"
                "assert not torch.allclose(p, p_orig), 'step() should change params'\n"
            ),
        },
        {
            "name": "[KILLER TEST] Zero grad still shrinks params (decoupled WD)",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p = torch.tensor([2.0, -3.0, 4.0], requires_grad=True)\n"
                "p_orig = p.clone()\n"
                "opt = {fn}([p], lr=0.1, weight_decay=0.5)\n"
                "p.grad = torch.zeros(3)  # zero gradient (not None)\n"
                "opt.step()\n"
                "# AdamW: p *= (1 - 0.1*0.5) = 0.95; Adam update with zero grad = 0\n"
                "expected = p_orig * (1 - 0.1 * 0.5)\n"
                "assert torch.allclose(p, expected, atol=1e-5), f'Decoupled WD must shrink params even with zero grad: {p.tolist()} vs {expected.tolist()}'\n"
            ),
        },
        {
            "name": "Matches torch.optim.AdamW over 5 steps",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p_ref = torch.randn(10, requires_grad=True)\n"
                "p_my = p_ref.detach().clone().requires_grad_()\n"
                "ref = torch.optim.AdamW([p_ref], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)\n"
                "my = {fn}([p_my], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)\n"
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
            "name": "Bias correction: step 1 update = lr * 1/(1+eps) for g=1, WD=0",
            "code": (
                "\nimport torch\n"
                "torch.manual_seed(0)\n"
                "p = torch.randn(5, requires_grad=True)\n"
                "p_orig = p.clone()\n"
                "opt = {fn}([p], lr=0.001, weight_decay=0.0)\n"
                "p.grad = torch.ones(5)\n"
                "opt.step()\n"
                "# Step 1: m = (1-β1)*g, m_hat = g = 1; v = (1-β2)*g², v_hat = g² = 1\n"
                "# p -= lr * 1 / (1 + eps)\n"
                "expected = p_orig - 0.001 * (1.0 / (1.0 + 1e-8))\n"
                "assert torch.allclose(p, expected, atol=1e-6), f'Step 1 bias correction wrong'\n"
            ),
        },
    ],
}
