# Practice — DL基礎 W2-W5 復習トラック

週ごとの練習問題セット。合計 **29 問**、TorchCode の 56 問から DL基礎の 週次内容に直結するものだけを pick up。

## Weeks

| Week | テーマ | 問題数 |
|:----:|--------|:------:|
| [W2](W2/) | MLP / 基本分類 / 基礎 optimization | 8 |
| [W3](W3/) | 正則化 / 正規化 / advanced optimization | 9 |
| [W4](W4/) | CNN 基礎 + 基本 transforms | 7 |
| [W5](W5/) | CIFAR-10 advanced レシピ | 5 |

## Source of truth

マッピングは `scripts/week_mapping.py` で管理。問題の追加・削除・並び替えは そこを編集して `python scripts/build_weeks.py` を再実行。

週フォルダ内の `.ipynb` は `templates/` からのコピーで、判定エンジン (`check("...")`) はファイルの場所に依存しないのでそのまま動く。
