"""AdamW (Decoupled Weight Decay) task — AUTO-GENERATED from problem_specs/adamw.py. Do not edit directly."""

TASK = {
    "title": "AdamW (Decoupled Weight Decay)",
    "difficulty": "Hard",
    "function_name": "MyAdamW",
    "hint": "**Order matters**: apply decoupled WD (`p *= (1 - lr*wd)`) BEFORE the Adam moment update. Increment `t` at the start of `step()`. Bias-correct with `1 - β^t`. Use `p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)` for the fused in-place update. Skip params with `grad is None` (do NOT apply WD to them either — that's what torch.optim.AdamW does).",
    "tests": [
        {
            "name": "Has step() and zero_grad() methods",
            "code": "\nimport torch\np = torch.randn(3, requires_grad=True)\nopt = {fn}([p], lr=0.01, weight_decay=0.01)\nassert hasattr(opt, 'step') and callable(opt.step)\nassert hasattr(opt, 'zero_grad') and callable(opt.zero_grad)\n"
        },
        {
            "name": "step() updates params (WD=0)",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(5, requires_grad=True)\np_orig = p.clone()\nopt = {fn}([p], lr=0.01, weight_decay=0.0)\np.grad = torch.ones(5)\nopt.step()\nassert not torch.allclose(p, p_orig), 'step() should change params'\n"
        },
        {
            "name": "[KILLER TEST] Zero grad still shrinks params (decoupled WD)",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.tensor([2.0, -3.0, 4.0], requires_grad=True)\np_orig = p.clone()\nopt = {fn}([p], lr=0.1, weight_decay=0.5)\np.grad = torch.zeros(3)  # zero gradient (not None)\nopt.step()\n# AdamW: p *= (1 - 0.1*0.5) = 0.95; Adam update with zero grad = 0\nexpected = p_orig * (1 - 0.1 * 0.5)\nassert torch.allclose(p, expected, atol=1e-5), f'Decoupled WD must shrink params even with zero grad: {p.tolist()} vs {expected.tolist()}'\n"
        },
        {
            "name": "Matches torch.optim.AdamW over 5 steps",
            "code": "\nimport torch\ntorch.manual_seed(0)\np_ref = torch.randn(10, requires_grad=True)\np_my = p_ref.detach().clone().requires_grad_()\nref = torch.optim.AdamW([p_ref], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)\nmy = {fn}([p_my], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)\nfor step in range(5):\n    p_ref.grad = (p_ref * 2).detach()\n    p_my.grad = (p_my * 2).detach()\n    ref.step()\n    my.step()\ndiff = (p_ref - p_my).abs().max().item()\nassert diff < 1e-5, f'After 5 steps, max diff: {diff:.6f}'\n"
        },
        {
            "name": "Bias correction: step 1 update = lr * 1/(1+eps) for g=1, WD=0",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(5, requires_grad=True)\np_orig = p.clone()\nopt = {fn}([p], lr=0.001, weight_decay=0.0)\np.grad = torch.ones(5)\nopt.step()\n# Step 1: m = (1-β1)*g, m_hat = g = 1; v = (1-β2)*g², v_hat = g² = 1\n# p -= lr * 1 / (1 + eps)\nexpected = p_orig - 0.001 * (1.0 / (1.0 + 1e-8))\nassert torch.allclose(p, expected, atol=1e-6), f'Step 1 bias correction wrong'\n"
        }
    ]
}
