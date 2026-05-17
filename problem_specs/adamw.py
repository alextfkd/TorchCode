"""B8 — AdamW (Loshchilov & Hutter 2019, Transformer 時代の標準 optimizer)."""

PROBLEM = {
    "id": "adamw",
    "number": 56,
    "title": "AdamW (Decoupled Weight Decay)",
    "difficulty": "Hard",
    "fn_name": "MyAdamW",

    "intro_md": (
        "**AdamW** (Loshchilov & Hutter 2019) を実装する。Adam + **decoupled weight decay**。"
        "Transformer 学習の modern default。\n\n"
        "### Decoupled の捻り\n"
        "plain Adam + L2 正則化は `λp` を moment update の **前** に gradient に足すので、"
        "L2 項が `√v_hat` で割られてしまう — 各 weight で実効 decay rate が変わる。"
        "AdamW は Adam update の後で decay を **param に直接** 適用するので、全 weight で "
        "実効 decay rate が同じになる：\n\n"
        "$$p \\leftarrow p \\cdot (1 - \\text{lr} \\cdot \\lambda)$$\n"
        "$$p \\leftarrow p - \\text{lr} \\cdot \\hat{m} / (\\sqrt{\\hat{v}} + \\varepsilon)$$\n\n"
        "### Killer test (interview の決め台詞)\n"
        "`weight_decay > 0` で **gradient = 0** でも、AdamW は param を縮める"
        "（decoupled WD は `g` に依らず `p` に作用）。plain Adam + L2 だとそのケースで何もしない。"
        "これが両者を区別する。\n"
    ),

    "signature": (
        "class MyAdamW:\n"
        "    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01): ...\n"
        "    def zero_grad(self): ...\n"
        "    def step(self): ..."
    ),

    "rules": [
        "`torch.optim.AdamW` や `torch.optim.Adam` は **使わない**",
        "Decoupled WD: `p *= (1 - lr * weight_decay)` を param に適用（gradient ではない）",
        "その後 Adam: m, v EMA + bias correction、`p -= lr * m_hat / (sqrt(v_hat) + eps)`",
        "`m, v` を zeros で、step counter `t = 0` で初期化。`step()` の最初に `t += 1`",
        "`p.grad is None` の param は skip（WD も適用しない — `torch.optim.AdamW` の挙動に合わせる）",
        "`step()` を `@torch.no_grad()` で wrap",
    ],

    "imports": "import torch",

    "template_body": (
        "class MyAdamW:\n"
        "    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):\n"
        "        pass  # hyperparam 保存、m=[zeros], v=[zeros], t=0 で初期化\n"
        "    \n"
        "    def zero_grad(self):\n"
        "        pass\n"
        "    \n"
        "    def step(self):\n"
        "        pass  # t += 1; 各 (p, m, v) で p.grad is not None なら:\n"
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
        "            # Decoupled weight decay — gradient じゃなく PARAM に適用\n"
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
        "**順序重要**: decoupled WD (`p *= (1 - lr*wd)`) を Adam moment update の **前** に適用。"
        "`step()` の最初に `t += 1`。Bias correction は `1 - β^t`。"
        "`p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)` で fused in-place update。"
        "`grad is None` の param は skip（WD も skip — `torch.optim.AdamW` の挙動）。"
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
