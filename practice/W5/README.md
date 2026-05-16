# W5 — CIFAR-10 advanced レシピ

CIFAR-10 で 95% → 97% に押し上げる「効くやつ」5点。Label smoothing + 強い augmentation (Cutout / Mixup / CutMix) + 推論時 TTA。

**5 problems.** Solve each template in place, then run the last cell `check("...")` to grade. Stuck? Open the linked solution in the rightmost column.

## Study order

| Order | # | Problem | Difficulty | Solution |
|:-----:|:-:|---------|:----------:|:--------:|
| 1 | 51 | [`51_label_smoothing_ce.ipynb`](51_label_smoothing_ce.ipynb) — Label Smoothing Cross-Entropy | Easy | [↗](../../solutions/51_label_smoothing_ce_solution.ipynb) |
| 2 | 45 | [`45_cutout.ipynb`](45_cutout.ipynb) — Cutout / RandomErasing | Medium | [↗](../../solutions/45_cutout_solution.ipynb) |
| 3 | 46 | [`46_mixup.ipynb`](46_mixup.ipynb) — Mixup | Medium | [↗](../../solutions/46_mixup_solution.ipynb) |
| 4 | 47 | [`47_cutmix.ipynb`](47_cutmix.ipynb) — CutMix | Medium | [↗](../../solutions/47_cutmix_solution.ipynb) |
| 5 | 48 | [`48_tta_hflip.ipynb`](48_tta_hflip.ipynb) — TTA (Horizontal Flip Averaging) | Easy | [↗](../../solutions/48_tta_hflip_solution.ipynb) |

## How to use

```bash
# from the repo root
make run                # or `docker compose up`
# then in JupyterLab, navigate to practice/W5/ and open a notebook
```

If you accidentally break a template, delete it and rerun `python scripts/build_weeks.py` to restore — it only copies missing files, so your other in-progress notebooks are preserved.
