"""Negative Log-Likelihood Loss task — AUTO-GENERATED from problem_specs/nll_loss.py. Do not edit directly."""

TASK = {
    "title": "Negative Log-Likelihood Loss",
    "difficulty": "Easy",
    "function_name": "my_nll_loss",
    "hint": "Use advanced indexing: `log_probs[torch.arange(B), targets]` gives the log-prob at the correct class for each sample. Negate and take mean.",
    "tests": [
        {
            "name": "Returns scalar",
            "code": "\nimport torch\nimport torch.nn.functional as F\nlog_probs = F.log_softmax(torch.randn(4, 10), dim=-1)\ntargets = torch.randint(0, 10, (4,))\nloss = {fn}(log_probs, targets)\nassert loss.dim() == 0, f'Expected scalar, got shape {loss.shape}'\n"
        },
        {
            "name": "Matches F.nll_loss",
            "code": "\nimport torch\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nlog_probs = F.log_softmax(torch.randn(8, 5), dim=-1)\ntargets = torch.randint(0, 5, (8,))\nout = {fn}(log_probs, targets)\nref = F.nll_loss(log_probs, targets)\nassert torch.allclose(out, ref, atol=1e-6), f'Got {out.item()}, expected {ref.item()}'\n"
        },
        {
            "name": "Equivalent to CrossEntropy when log_probs = log_softmax(logits)",
            "code": "\nimport torch\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nlogits = torch.randn(16, 7)\ntargets = torch.randint(0, 7, (16,))\nout = {fn}(F.log_softmax(logits, dim=-1), targets)\nref = F.cross_entropy(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), 'NLL∘log_softmax should equal CE'\n"
        },
        {
            "name": "Gradient flows back to log_probs",
            "code": "\nimport torch\nlog_probs = torch.randn(4, 10).log_softmax(dim=-1).detach().requires_grad_()\ntargets = torch.randint(0, 10, (4,))\nloss = {fn}(log_probs, targets)\nloss.backward()\nassert log_probs.grad is not None and log_probs.grad.shape == log_probs.shape\n"
        },
        {
            "name": "Per-sample indexing correctness (handcrafted)",
            "code": "\nimport torch\nlog_probs = torch.tensor([\n    [-0.1, -10.0, -10.0],\n    [-10.0, -0.2, -10.0],\n    [-10.0, -10.0, -0.3],\n])\ntargets = torch.tensor([0, 1, 2])\nout = {fn}(log_probs, targets)\nexpected = -(-0.1 + -0.2 + -0.3) / 3\nassert abs(out.item() - expected) < 1e-6, f'Got {out.item()}, expected {expected}'\n"
        }
    ]
}
