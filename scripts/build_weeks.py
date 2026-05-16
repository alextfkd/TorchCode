#!/usr/bin/env python3
"""Build practice/W{2,3,4,5}/ from scripts/week_mapping.py.

For each week:
- Copy templates/{file_id}.ipynb → practice/{week}/{file_id}.ipynb (only if missing,
  so in-progress solutions aren't clobbered)
- Always (re)generate practice/{week}/README.md with the study order, links,
  and links back to the central solutions/.
- Always (re)generate practice/README.md as the top-level index.

Usage:
    python scripts/build_weeks.py              # build all weeks (preserves in-progress)
    python scripts/build_weeks.py --reset      # wipe practice/W*/ first (loses your work)
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRACTICE_DIR = ROOT / "practice"
TEMPLATES_DIR = ROOT / "templates"
SOLUTIONS_DIR = ROOT / "solutions"
MAPPING_PATH = ROOT / "scripts" / "week_mapping.py"


def load_mapping() -> dict:
    spec = importlib.util.spec_from_file_location("week_mapping", MAPPING_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WEEK_MAPPING


def load_tasks() -> dict:
    """Import the live task registry to look up titles/difficulty."""
    sys.path.insert(0, str(ROOT))
    from torch_judge.tasks import TASKS
    return TASKS


def file_to_task_id(file_id: str) -> str:
    """`01_relu` → `relu`."""
    return file_id.split("_", 1)[1]


def render_week_readme(week_id: str, week_info: dict, tasks: dict) -> str:
    lines = [f"# {week_id} — {week_info['title']}", ""]
    lines.append(week_info["intro"])
    lines.append("")
    lines.append(
        f"**{len(week_info['problems'])} problems.** "
        "Solve each template in place, then run the last cell `check(\"...\")` to grade. "
        "Stuck? Open the linked solution in the rightmost column."
    )
    lines.append("")
    lines.append("## Study order")
    lines.append("")
    lines.append("| Order | # | Problem | Difficulty | Solution |")
    lines.append("|:-----:|:-:|---------|:----------:|:--------:|")
    for i, file_id in enumerate(week_info["problems"], 1):
        task_id = file_to_task_id(file_id)
        task = tasks.get(task_id)
        title = task["title"] if task else task_id
        diff = task["difficulty"] if task else "?"
        num = file_id.split("_", 1)[0]
        template_link = f"[`{file_id}.ipynb`]({file_id}.ipynb)"
        solution_link = f"[↗](../../solutions/{file_id}_solution.ipynb)"
        lines.append(f"| {i} | {num} | {template_link} — {title} | {diff} | {solution_link} |")
    lines.append("")
    lines.append("## How to use")
    lines.append("")
    lines.append("```bash")
    lines.append("# from the repo root")
    lines.append("make run                # or `docker compose up`")
    lines.append(f"# then in JupyterLab, navigate to practice/{week_id}/ and open a notebook")
    lines.append("```")
    lines.append("")
    lines.append(
        "If you accidentally break a template, delete it and rerun "
        "`python scripts/build_weeks.py` to restore — it only copies missing files, "
        "so your other in-progress notebooks are preserved."
    )
    return "\n".join(lines) + "\n"


def render_index(mapping: dict, tasks: dict) -> str:
    total = sum(len(w["problems"]) for w in mapping.values())
    lines = [
        "# Practice — DL基礎 W2-W5 復習トラック",
        "",
        f"週ごとの練習問題セット。合計 **{total} 問**、TorchCode の 56 問から DL基礎の "
        "週次内容に直結するものだけを pick up。",
        "",
        "## Weeks",
        "",
    ]
    lines.append("| Week | テーマ | 問題数 |")
    lines.append("|:----:|--------|:------:|")
    for week_id, info in mapping.items():
        lines.append(f"| [{week_id}]({week_id}/) | {info['title']} | {len(info['problems'])} |")
    lines.append("")
    lines.append("## Source of truth")
    lines.append("")
    lines.append(
        "マッピングは `scripts/week_mapping.py` で管理。問題の追加・削除・並び替えは "
        "そこを編集して `python scripts/build_weeks.py` を再実行。"
    )
    lines.append("")
    lines.append(
        "週フォルダ内の `.ipynb` は `templates/` からのコピーで、判定エンジン "
        "(`check(\"...\")`) はファイルの場所に依存しないのでそのまま動く。"
    )
    return "\n".join(lines) + "\n"


def build_week(week_id: str, week_info: dict, tasks: dict, reset: bool) -> tuple[int, int]:
    """Return (copied, kept) counts."""
    week_dir = PRACTICE_DIR / week_id
    if reset and week_dir.exists():
        shutil.rmtree(week_dir)
    week_dir.mkdir(parents=True, exist_ok=True)

    copied = kept = 0
    for file_id in week_info["problems"]:
        src = TEMPLATES_DIR / f"{file_id}.ipynb"
        if not src.exists():
            print(f"  ✗ {file_id}.ipynb missing in templates/", file=sys.stderr)
            continue
        dst = week_dir / f"{file_id}.ipynb"
        if dst.exists():
            kept += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    # README is always regenerated (no user state in it)
    readme = render_week_readme(week_id, week_info, tasks)
    (week_dir / "README.md").write_text(readme, encoding="utf-8")
    return copied, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Wipe practice/W*/ before rebuilding (loses in-progress work)")
    args = ap.parse_args()

    if not TEMPLATES_DIR.exists():
        print(f"templates/ not found at {TEMPLATES_DIR}", file=sys.stderr)
        return 1

    mapping = load_mapping()
    tasks = load_tasks()
    PRACTICE_DIR.mkdir(exist_ok=True)

    for week_id, week_info in mapping.items():
        copied, kept = build_week(week_id, week_info, tasks, reset=args.reset)
        print(f"✓ {week_id}: {copied} copied, {kept} kept ({len(week_info['problems'])} total) — README regenerated")

    (PRACTICE_DIR / "README.md").write_text(render_index(mapping, tasks), encoding="utf-8")
    print(f"✓ index: practice/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
