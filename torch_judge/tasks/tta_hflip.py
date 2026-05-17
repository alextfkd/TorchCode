"""TTA (Horizontal Flip Averaging) task — AUTO-GENERATED from problem_specs/tta_hflip.py. Do not edit directly."""

TASK = {
    "title": "TTA (Horizontal Flip Averaging)",
    "difficulty": "Easy",
    "function_name": "tta_hflip",
    "hint": "2 つの forward pass: `model(x)` と `model(x.flip(-1))`。それぞれを `softmax(dim=-1)` で 確率に変換、平均。推論なので `torch.no_grad()` で wrap。",
    "tests": [
        {
            "name": "Output shape (B, num_classes)",
            "code": "\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\nmodel.eval()\nx = torch.randn(4, 3, 16, 16)\nout = {fn}(model, x)\nassert out.shape == (4, 10), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Output is a probability distribution (rows sum to 1, non-negative)",
            "code": "\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\nmodel.eval()\nx = torch.randn(3, 3, 8, 8)\nout = {fn}(model, x)\nsums = out.sum(dim=-1)\nassert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f'Row sums: {sums.tolist()}'\nassert (out >= 0).all(), 'Probabilities must be non-negative'\n"
        },
        {
            "name": "Equals averaged softmax of model(x) and model(flip(x))",
            "code": "\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\nmodel.eval()\nx = torch.randn(2, 3, 8, 8)\nout = {fn}(model, x)\nwith torch.no_grad():\n    expected = (F.softmax(model(x), dim=-1) + F.softmax(model(x.flip(-1)), dim=-1)) / 2\nassert torch.allclose(out, expected, atol=1e-6), f'Max diff: {(out-expected).abs().max():.6f}'\n"
        },
        {
            "name": "Invariant to flipping the input (tta(x) == tta(flip(x)))",
            "code": "\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\nmodel.eval()\nx = torch.randn(2, 3, 8, 8)\na = {fn}(model, x)\nb = {fn}(model, x.flip(-1))\nassert torch.allclose(a, b, atol=1e-6), 'TTA must be invariant to flipping the input'\n"
        },
        {
            "name": "Differs from single-forward for asymmetric input",
            "code": "\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))\nmodel.eval()\nx = torch.randn(2, 3, 8, 8)\nout_tta = {fn}(model, x)\nwith torch.no_grad():\n    out_single = F.softmax(model(x), dim=-1)\nassert not torch.allclose(out_tta, out_single, atol=1e-4), 'TTA result should differ from single forward'\n"
        }
    ]
}
