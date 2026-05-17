"""Per-Channel Normalize task — AUTO-GENERATED from problem_specs/normalize.py. Do not edit directly."""

TASK = {
    "title": "Per-Channel Normalize",
    "difficulty": "Easy",
    "function_name": "my_normalize",
    "hint": "`torch.as_tensor` で mean/std を x と同じ dtype/device の tensor に変換（tuple/list/tensor を一律で扱える）。それを `(C, 1, 1)` に reshape すれば `(C, H, W)` と `(B, C, H, W)` 両方で broadcast が効く。",
    "tests": [
        {
            "name": "Shape preserved (B, C, H, W)",
            "code": "\nimport torch\nx = torch.rand(2, 3, 8, 8)\nout = {fn}(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\nassert out.shape == x.shape, f'Shape: {out.shape}'\n"
        },
        {
            "name": "Matches manual (x - mean) / std",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.rand(2, 3, 4, 4)\nmean = (0.4914, 0.4822, 0.4465)\nstd = (0.2470, 0.2435, 0.2616)\nout = {fn}(x, mean, std)\nm = torch.tensor(mean).view(-1, 1, 1)\ns = torch.tensor(std).view(-1, 1, 1)\nref = (x - m) / s\nassert torch.allclose(out, ref, atol=1e-6), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Unbatched (C, H, W) also works",
            "code": "\nimport torch\nx = torch.rand(3, 8, 8)\nout = {fn}(x, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))\nassert out.shape == x.shape, f'Shape: {out.shape}'\nref = (x - torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)) / 0.25\nassert torch.allclose(out, ref, atol=1e-6), 'Value mismatch on unbatched input'\n"
        },
        {
            "name": "Accepts list / tuple / tensor for mean & std",
            "code": "\nimport torch\nx = torch.rand(1, 3, 4, 4)\nout_tup = {fn}(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\nout_lst = {fn}(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])\nout_ten = {fn}(x, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]))\nassert torch.allclose(out_tup, out_lst, atol=1e-6), 'list vs tuple mismatch'\nassert torch.allclose(out_tup, out_ten, atol=1e-6), 'tensor vs tuple mismatch'\n"
        },
        {
            "name": "Gradient flows to x",
            "code": "\nimport torch\nx = torch.rand(1, 3, 4, 4, requires_grad=True)\nout = {fn}(x, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))\nout.sum().backward()\nassert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'\nexpected = (1.0 / 0.25)\nassert torch.allclose(x.grad, torch.full_like(x, expected), atol=1e-5), f'd/dx should be 1/std = {expected}'\n"
        }
    ]
}
