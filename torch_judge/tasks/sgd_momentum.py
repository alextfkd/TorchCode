"""SGD with Momentum task — AUTO-GENERATED from problem_specs/sgd_momentum.py. Do not edit directly."""

TASK = {
    "title": "SGD with Momentum",
    "difficulty": "Medium",
    "function_name": "MySGDMomentum",
    "hint": "Store params, lr, momentum, and velocity buffers (`zeros_like` each param). In `step()`: wrap with `@torch.no_grad()`. For each `(p, v)` pair: `v.mul_(momentum).add_(p.grad)` then `p.add_(v, alpha=-lr)`. Skip params with `p.grad is None`.",
    "tests": [
        {
            "name": "Has step() and zero_grad() methods",
            "code": "\nimport torch\np = torch.randn(3, requires_grad=True)\nopt = {fn}([p], lr=0.01, momentum=0.9)\nassert hasattr(opt, 'step') and callable(opt.step), 'Missing step()'\nassert hasattr(opt, 'zero_grad') and callable(opt.zero_grad), 'Missing zero_grad()'\n"
        },
        {
            "name": "momentum=0 → plain SGD",
            "code": "\nimport torch\ntorch.manual_seed(0)\np1 = torch.randn(5, requires_grad=True)\np2 = p1.detach().clone().requires_grad_()\nopt = {fn}([p2], lr=0.1, momentum=0.0)\ng = torch.ones(5)\np1.grad = g.clone()\np2.grad = g.clone()\nopt.step()\nexpected = p1 - 0.1 * g\nassert torch.allclose(p2, expected, atol=1e-6), f'momentum=0 should be plain SGD'\n"
        },
        {
            "name": "Matches torch.optim.SGD with momentum over 5 steps",
            "code": "\nimport torch\ntorch.manual_seed(0)\np_ref = torch.randn(10, requires_grad=True)\np_my = p_ref.detach().clone().requires_grad_()\nref = torch.optim.SGD([p_ref], lr=0.05, momentum=0.9)\nmy = {fn}([p_my], lr=0.05, momentum=0.9)\nfor step in range(5):\n    p_ref.grad = (p_ref * 2).detach()\n    p_my.grad = (p_my * 2).detach()\n    ref.step()\n    my.step()\ndiff = (p_ref - p_my).abs().max().item()\nassert diff < 1e-5, f'After 5 steps, max diff: {diff:.6f}'\n"
        },
        {
            "name": "Velocity accumulates (step 2 delta > 1.5x step 1 delta)",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(3, requires_grad=True)\nopt = {fn}([p], lr=0.1, momentum=0.9)\ng = torch.ones(3)\n# Step 1: v = g; |delta_1| = 0.1\np_start = p.clone()\np.grad = g.clone()\nopt.step()\ndelta_1 = (p - p_start).abs().mean().item()\np_mid = p.clone()\n# Step 2: v = 0.9*g + g = 1.9*g; |delta_2| = 0.19\np.grad = g.clone()\nopt.step()\ndelta_2 = (p - p_mid).abs().mean().item()\nassert delta_2 > delta_1 * 1.5, f'Step 2 delta ({delta_2:.4f}) should be ~1.9x step 1 ({delta_1:.4f})'\n"
        },
        {
            "name": "zero_grad clears gradients",
            "code": "\nimport torch\np = torch.randn(4, requires_grad=True)\np.grad = torch.ones(4)\nopt = {fn}([p], lr=0.01, momentum=0.9)\nopt.zero_grad()\nassert (p.grad == 0).all(), f'zero_grad should zero out gradients: got {p.grad}'\n"
        }
    ]
}
