"""Top-k Accuracy task — AUTO-GENERATED from problem_specs/top_k_accuracy.py. Do not edit directly."""

TASK = {
    "title": "Top-k Accuracy",
    "difficulty": "Easy",
    "function_name": "top_k_accuracy",
    "hint": "`logits.topk(k, dim=-1)` は `(values, indices)` を返す。`indices` (shape `(B, k)`) と `targets.unsqueeze(-1)` (shape `(B, 1)`) を比較 — `.any(dim=-1)` → `.float().mean().item()` で float の accuracy。",
    "tests": [
        {
            "name": "All correct → 1.0",
            "code": "\nimport torch\nlogits = torch.eye(10) * 100\ntargets = torch.arange(10)\nacc = {fn}(logits, targets, k=1)\nassert abs(float(acc) - 1.0) < 1e-6, f'Expected 1.0, got {acc}'\n"
        },
        {
            "name": "All wrong → 0.0",
            "code": "\nimport torch\nlogits = torch.zeros(4, 5)\nlogits[:, 0] = 10.0  # everyone predicts class 0\ntargets = torch.tensor([1, 2, 3, 4])\nacc = {fn}(logits, targets, k=1)\nassert abs(float(acc) - 0.0) < 1e-6, f'Expected 0.0, got {acc}'\n"
        },
        {
            "name": "top-5 ≥ top-1 (monotone in k)",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.randn(50, 10)\ntargets = torch.randint(0, 10, (50,))\nacc1 = float({fn}(logits, targets, k=1))\nacc5 = float({fn}(logits, targets, k=5))\nassert acc5 >= acc1, f'top-5 ({acc5}) must be ≥ top-1 ({acc1})'\n"
        },
        {
            "name": "k ≥ num_classes → 1.0",
            "code": "\nimport torch\nlogits = torch.randn(8, 5)\ntargets = torch.randint(0, 5, (8,))\nacc = {fn}(logits, targets, k=10)\nassert abs(float(acc) - 1.0) < 1e-6, f'k≥K should give 1.0, got {acc}'\n"
        },
        {
            "name": "Matches manual top-k count",
            "code": "\nimport torch\ntorch.manual_seed(0)\nB, K, k = 20, 10, 3\nlogits = torch.randn(B, K)\ntargets = torch.randint(0, K, (B,))\n_, top_idx = logits.topk(k, dim=-1)\nmanual = (top_idx == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()\nout = float({fn}(logits, targets, k=k))\nassert abs(out - manual) < 1e-6, f'Got {out}, manual {manual}'\n"
        }
    ]
}
