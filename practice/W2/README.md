# W2 — MLP / 基本分類 / 基礎 optimization

MLP の forward → activation → loss → optimizer → 評価指標 までの一周。基本的な MNIST / 分類パイプラインの構成部品を一つずつ自前実装する週。

**8 problems.** Solve each template in place, then run the last cell `check("...")` to grade. Stuck? Open the linked solution in the rightmost column.

## Study order

| Order | # | Problem | Difficulty | Solution |
|:-----:|:-:|---------|:----------:|:--------:|
| 1 | 40 | [`40_linear_regression.ipynb`](40_linear_regression.ipynb) — Linear Regression | Medium | [↗](../../solutions/40_linear_regression_solution.ipynb) |
| 2 | 03 | [`03_linear.ipynb`](03_linear.ipynb) — Simple Linear Layer | Medium | [↗](../../solutions/03_linear_solution.ipynb) |
| 3 | 01 | [`01_relu.ipynb`](01_relu.ipynb) — Implement ReLU | Easy | [↗](../../solutions/01_relu_solution.ipynb) |
| 4 | 02 | [`02_softmax.ipynb`](02_softmax.ipynb) — Implement Softmax | Easy | [↗](../../solutions/02_softmax_solution.ipynb) |
| 5 | 16 | [`16_cross_entropy.ipynb`](16_cross_entropy.ipynb) — Cross-Entropy Loss | Easy | [↗](../../solutions/16_cross_entropy_solution.ipynb) |
| 6 | 53 | [`53_nll_loss.ipynb`](53_nll_loss.ipynb) — Negative Log-Likelihood Loss | Easy | [↗](../../solutions/53_nll_loss_solution.ipynb) |
| 7 | 54 | [`54_sgd_momentum.ipynb`](54_sgd_momentum.ipynb) — SGD with Momentum | Medium | [↗](../../solutions/54_sgd_momentum_solution.ipynb) |
| 8 | 52 | [`52_top_k_accuracy.ipynb`](52_top_k_accuracy.ipynb) — Top-k Accuracy | Easy | [↗](../../solutions/52_top_k_accuracy_solution.ipynb) |

## How to use

```bash
# from the repo root
make run                # or `docker compose up`
# then in JupyterLab, navigate to practice/W2/ and open a notebook
```

If you accidentally break a template, delete it and rerun `python scripts/build_weeks.py` to restore — it only copies missing files, so your other in-progress notebooks are preserved.
