"""Label Smoothing Cross-Entropy task — AUTO-GENERATED from problem_specs/label_smoothing_ce.py. Do not edit directly."""

TASK = {
    "title": "Label Smoothing Cross-Entropy",
    "difficulty": "Easy",
    "function_name": "label_smoothing_ce",
    "hint": "Use `F.log_softmax(logits, dim=-1)`. Build `target_dist = one_hot * (1 - smoothing) + smoothing / K` — true class gets `(1-ε) + ε/K`, others get `ε/K`. Loss = `-(target_dist * log_probs).sum(-1).mean()`.",
    "tests": [
        {
            "name": "Returns scalar",
            "code": "\nimport torch\nlogits = torch.randn(4, 10)\ntargets = torch.randint(0, 10, (4,))\nloss = {fn}(logits, targets, smoothing=0.1)\nassert loss.dim() == 0, f'Expected scalar, got shape {loss.shape}'\n"
        },
        {
            "name": "smoothing=0 equals standard CrossEntropy",
            "code": "\nimport torch\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nlogits = torch.randn(8, 10)\ntargets = torch.randint(0, 10, (8,))\nout = {fn}(logits, targets, smoothing=0.0)\nref = F.cross_entropy(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), f'ε=0: {out.item():.6f} vs CE {ref.item():.6f}'\n"
        },
        {
            "name": "Matches nn.CrossEntropyLoss(label_smoothing=0.1)",
            "code": "\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(0)\nlogits = torch.randn(16, 5)\ntargets = torch.randint(0, 5, (16,))\nout = {fn}(logits, targets, smoothing=0.1)\nref = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), f'Got {out.item():.6f}, expected {ref.item():.6f}'\n"
        },
        {
            "name": "Gradient flows back to logits",
            "code": "\nimport torch\nlogits = torch.randn(4, 10, requires_grad=True)\ntargets = torch.randint(0, 10, (4,))\nloss = {fn}(logits, targets, smoothing=0.1)\nloss.backward()\nassert logits.grad is not None and logits.grad.shape == logits.shape\n"
        },
        {
            "name": "Smoothing increases loss on confident-correct predictions",
            "code": "\nimport torch\ntorch.manual_seed(0)\n# Strongly correct predictions: smoothing should INCREASE the loss\nlogits = torch.zeros(4, 5)\nlogits[torch.arange(4), torch.tensor([0, 1, 2, 3])] = 10.0\ntargets = torch.tensor([0, 1, 2, 3])\nloss_0 = {fn}(logits, targets, smoothing=0.0)\nloss_1 = {fn}(logits, targets, smoothing=0.2)\nassert loss_1 > loss_0, f'Smoothing should increase loss on confident-correct: {loss_0.item()} vs {loss_1.item()}'\n"
        }
    ]
}
