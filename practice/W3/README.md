# W3 — 正則化 / 正規化 / advanced optimization

深いネットを安定して訓練するための道具一式。init → normalization → regularization → optimizer → schedule → stability の順で学習。

**9 problems.** Solve each template in place, then run the last cell `check("...")` to grade. Stuck? Open the linked solution in the rightmost column.

## Study order

| Order | # | Problem | Difficulty | Solution |
|:-----:|:-:|---------|:----------:|:--------:|
| 1 | 20 | [`20_weight_init.ipynb`](20_weight_init.ipynb) — Kaiming Initialization | Easy | [↗](../../solutions/20_weight_init_solution.ipynb) |
| 2 | 07 | [`07_batchnorm.ipynb`](07_batchnorm.ipynb) — Implement BatchNorm | Medium | [↗](../../solutions/07_batchnorm_solution.ipynb) |
| 3 | 04 | [`04_layernorm.ipynb`](04_layernorm.ipynb) — Implement LayerNorm | Medium | [↗](../../solutions/04_layernorm_solution.ipynb) |
| 4 | 17 | [`17_dropout.ipynb`](17_dropout.ipynb) — Implement Dropout | Easy | [↗](../../solutions/17_dropout_solution.ipynb) |
| 5 | 55 | [`55_weight_decay.ipynb`](55_weight_decay.ipynb) — Weight Decay (L2 in gradient) | Easy | [↗](../../solutions/55_weight_decay_solution.ipynb) |
| 6 | 29 | [`29_adam.ipynb`](29_adam.ipynb) — Adam Optimizer | Medium | [↗](../../solutions/29_adam_solution.ipynb) |
| 7 | 56 | [`56_adamw.ipynb`](56_adamw.ipynb) — AdamW (Decoupled Weight Decay) | Hard | [↗](../../solutions/56_adamw_solution.ipynb) |
| 8 | 30 | [`30_cosine_lr.ipynb`](30_cosine_lr.ipynb) — Cosine LR Scheduler with Warmup | Medium | [↗](../../solutions/30_cosine_lr_solution.ipynb) |
| 9 | 21 | [`21_gradient_clipping.ipynb`](21_gradient_clipping.ipynb) — Gradient Norm Clipping | Easy | [↗](../../solutions/21_gradient_clipping_solution.ipynb) |

## How to use

```bash
# from the repo root
make run                # or `docker compose up`
# then in JupyterLab, navigate to practice/W3/ and open a notebook
```

If you accidentally break a template, delete it and rerun `python scripts/build_weeks.py` to restore — it only copies missing files, so your other in-progress notebooks are preserved.
