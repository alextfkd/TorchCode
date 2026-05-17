# W4 — CNN 基礎 + 基本 transforms

CIFAR-10 CNN を構成する全部品。Conv → Pool → GAP の forward + 基本 data augmentation (Normalize / HFlip / RandomCrop) のセット。

**7 problems.** Solve each template in place, then run the last cell `check("...")` to grade. Stuck? Open the linked solution in the rightmost column.

## Study order

| Order | # | Problem | Difficulty | Solution | Colab |
|:-----:|:-:|---------|:----------:|:--------:|:-----:|
| 1 | 22 | [`22_conv2d.ipynb`](22_conv2d.ipynb) — 2D Convolution | Medium | [↗](../../solutions/22_conv2d_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/22_conv2d.ipynb) |
| 2 | 41 | [`41_maxpool2d.ipynb`](41_maxpool2d.ipynb) — 2D Max Pooling | Medium | [↗](../../solutions/41_maxpool2d_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/41_maxpool2d.ipynb) |
| 3 | 49 | [`49_avg_pool2d.ipynb`](49_avg_pool2d.ipynb) — 2D Average Pooling | Easy | [↗](../../solutions/49_avg_pool2d_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/49_avg_pool2d.ipynb) |
| 4 | 50 | [`50_global_avg_pool.ipynb`](50_global_avg_pool.ipynb) — Global Average Pooling | Easy | [↗](../../solutions/50_global_avg_pool_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/50_global_avg_pool.ipynb) |
| 5 | 42 | [`42_normalize.ipynb`](42_normalize.ipynb) — Per-Channel Normalize | Easy | [↗](../../solutions/42_normalize_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/42_normalize.ipynb) |
| 6 | 43 | [`43_random_hflip.ipynb`](43_random_hflip.ipynb) — Random Horizontal Flip | Easy | [↗](../../solutions/43_random_hflip_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/43_random_hflip.ipynb) |
| 7 | 44 | [`44_random_crop.ipynb`](44_random_crop.ipynb) — Random Crop with Padding | Easy | [↗](../../solutions/44_random_crop_solution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alextfkd/TorchCode/blob/master/practice/W4/44_random_crop.ipynb) |

## How to use

```bash
# from the repo root
make run                # or `docker compose up`
# then in JupyterLab, navigate to practice/W4/ and open a notebook
```

If you accidentally break a template, delete it and rerun `python scripts/build_weeks.py` to restore — it only copies missing files, so your other in-progress notebooks are preserved.
