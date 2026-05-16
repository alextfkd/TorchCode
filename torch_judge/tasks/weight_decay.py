"""Weight Decay (L2 in gradient) task — AUTO-GENERATED from problem_specs/weight_decay.py. Do not edit directly."""

TASK = {
    "title": "Weight Decay (L2 in gradient)",
    "difficulty": "Easy",
    "function_name": "apply_weight_decay",
    "hint": "Single-tensor case: wrap in a list. `weight_decay=0` fast path. For each param: `p.grad.add_(p, alpha=weight_decay)` is equivalent to `p.grad += weight_decay * p`. Use `@torch.no_grad()` since we're modifying `.grad` outside autograd.",
    "tests": [
        {
            "name": "Adds weight_decay * p to grad",
            "code": "\nimport torch\np = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\np.grad = torch.ones(3)\n{fn}(p, 0.1)\nexpected = torch.tensor([1.1, 1.2, 1.3])\nassert torch.allclose(p.grad, expected, atol=1e-6), f'Got {p.grad}, expected {expected}'\n"
        },
        {
            "name": "weight_decay=0 → no change",
            "code": "\nimport torch\np = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\np.grad = torch.tensor([0.5, 0.6, 0.7])\norig = p.grad.clone()\n{fn}(p, 0.0)\nassert torch.allclose(p.grad, orig), 'WD=0 should not change grad'\n"
        },
        {
            "name": "Works on a list of params",
            "code": "\nimport torch\np1 = torch.tensor([1., 2.], requires_grad=True); p1.grad = torch.zeros(2)\np2 = torch.tensor([3., 4.], requires_grad=True); p2.grad = torch.zeros(2)\n{fn}([p1, p2], 0.5)\nassert torch.allclose(p1.grad, torch.tensor([0.5, 1.0]))\nassert torch.allclose(p2.grad, torch.tensor([1.5, 2.0]))\n"
        },
        {
            "name": "Skips params with grad=None",
            "code": "\nimport torch\np1 = torch.tensor([1., 2.], requires_grad=True); p1.grad = torch.zeros(2)\np2 = torch.tensor([3., 4.], requires_grad=True)  # grad is None\n{fn}([p1, p2], 0.1)  # should not raise\nassert torch.allclose(p1.grad, torch.tensor([0.1, 0.2]))\nassert p2.grad is None, 'p2.grad should still be None'\n"
        },
        {
            "name": "Does NOT modify p itself (only p.grad)",
            "code": "\nimport torch\np = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\np.grad = torch.zeros(3)\np_orig = p.clone()\n{fn}(p, 0.5)\nassert torch.allclose(p, p_orig), 'p itself should NOT change — only p.grad'\n"
        }
    ]
}
