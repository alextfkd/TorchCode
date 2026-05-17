#!/usr/bin/env python3
"""Build TorchCode problems from problem_specs/ — single source of truth.

Each problem_specs/{id}.py exports a PROBLEM dict with all fields needed to
generate the task definition + template notebook + solution notebook.

Usage:
    python scripts/build.py              # build all specs
    python scripts/build.py normalize    # build a single spec by id
    python scripts/build.py --verify     # build + run judge on every solution
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = "alextfkd/TorchCode"
BRANCH = "master"
ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "problem_specs"
TASKS_DIR = ROOT / "torch_judge" / "tasks"
TEMPLATES_DIR = ROOT / "templates"
SOLUTIONS_DIR = ROOT / "solutions"

BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"

DIFFICULTY_EMOJI = {"Easy": "🟢", "Medium": "🟠", "Hard": "🔴"}

COLAB_INSTALL = """# Install torch-judge in Colab (no-op in JupyterLab/Docker)
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q --force-reinstall --no-deps git+https://github.com/alextfkd/TorchCode.git')
except ImportError:
    pass"""


def colab_url(folder: str, filename: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{folder}/{filename}"


def md_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
    }


def code_cell(src: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
        "execution_count": None,
    }


def nb(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def load_spec(spec_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(spec_path.stem, spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PROBLEM"):
        raise ValueError(f"{spec_path} missing PROBLEM dict")
    return mod.PROBLEM


def render_intro(p: dict[str, Any], folder: str, filename: str) -> str:
    import re as _re
    emoji = DIFFICULTY_EMOJI.get(p["difficulty"], "")
    badge = f"[![Open In Colab]({BADGE_IMG})]({colab_url(folder, filename)})"
    rules_md = "\n".join(f"- {r}" for r in p.get("rules", []))

    memo_block = ""
    if p.get("memo"):
        memo_block += f"> 💡 **どこで使う**：{p['memo']}"
    if p.get("details"):
        if memo_block:
            memo_block += "\n\n"
        memo_block += (
            f"<details>\n<summary>📖 もっと詳しく</summary>\n\n"
            f"{p['details']}\n\n</details>"
        )

    # Split intro_md at the first "### ..." section header so memo/details
    # land right after the formula (which is in the top section), not after
    # any sub-sections like "### Example" / "### Algorithm".
    intro_text = p["intro_md"]
    m = _re.search(r"\n###\s", intro_text)
    if m and memo_block:
        intro_top = intro_text[:m.start()].rstrip()
        intro_bottom = intro_text[m.start():].lstrip("\n")
        intro_combined = f"{intro_top}\n\n{memo_block}\n\n{intro_bottom}"
    elif memo_block:
        intro_combined = f"{intro_text.rstrip()}\n\n{memo_block}"
    else:
        intro_combined = intro_text

    return (
        f"{badge}\n\n"
        f"# {emoji} {p['difficulty']}: {p['title']}\n\n"
        f"{intro_combined}\n\n"
        f"### Signature\n```python\n{p['signature']}\n```\n\n"
        f"### Rules\n{rules_md}"
    )


def render_solution_intro(p: dict[str, Any], folder: str, filename: str) -> str:
    badge = f"[![Open In Colab]({BADGE_IMG})]({colab_url(folder, filename)})"
    return f"{badge}\n\n# Solution: {p['title']}\n\nReference solution."


def write_task(p: dict[str, Any]) -> Path:
    """Emit torch_judge/tasks/{id}.py — the minimal runtime artifact."""
    task = {
        "title": p["title"],
        "difficulty": p["difficulty"],
        "function_name": p["fn_name"],
        "hint": p["hint"],
        "tests": p["tests"],
    }
    path = TASKS_DIR / f"{p['id']}.py"
    body = f'"""{p["title"]} task — AUTO-GENERATED from problem_specs/{p["id"]}.py. Do not edit directly."""\n\nTASK = {json.dumps(task, indent=4, ensure_ascii=False)}\n'
    path.write_text(body, encoding="utf-8")
    return path


def write_template(p: dict[str, Any]) -> Path:
    filename = f"{p['number']:02d}_{p['id']}.ipynb"
    cells = [
        md_cell(render_intro(p, "templates", filename)),
        code_cell(COLAB_INSTALL),
        code_cell(p["imports"]),
        code_cell(f"# ✏️ YOUR IMPLEMENTATION HERE\n\n{p['template_body']}"),
        code_cell(p["demo_code"]),
        code_cell(f'# ✅ SUBMIT — Run this cell to check your solution\nfrom torch_judge import check\ncheck("{p["id"]}")'),
    ]
    path = TEMPLATES_DIR / filename
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def write_solution(p: dict[str, Any]) -> Path:
    filename = f"{p['number']:02d}_{p['id']}_solution.ipynb"
    cells = [
        md_cell(render_solution_intro(p, "solutions", filename)),
        code_cell(COLAB_INSTALL),
        code_cell(p["imports"]),
        code_cell(f"# ✅ SOLUTION\n\n{p['solution_body']}"),
        code_cell(p["demo_code"]),
        code_cell(f'from torch_judge import check\ncheck("{p["id"]}")'),
    ]
    path = SOLUTIONS_DIR / filename
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def verify_spec(p: dict[str, Any]) -> tuple[bool, str]:
    """Run the solution through the judge — must pass all tests."""
    sys.path.insert(0, str(ROOT))
    # Force reload to pick up newly-written tasks/{id}.py
    import importlib
    import torch_judge.tasks._registry as registry
    importlib.reload(registry)
    from torch_judge.tasks._registry import TASKS, get_task
    from torch_judge import engine

    task = get_task(p["id"])
    if task is None:
        return False, f"task '{p['id']}' not found in registry after build"

    # Execute the solution body to get the user function
    ns: dict[str, Any] = {}
    exec(compile(p["imports"], "<imports>", "exec"), ns)
    exec(compile(p["solution_body"], "<solution>", "exec"), ns)
    fn = ns.get(p["fn_name"])
    if fn is None:
        return False, f"solution body did not define {p['fn_name']}"

    # Inject into engine and run check
    engine._get_user_namespace = lambda: {p["fn_name"]: fn}
    passed = 0
    failed: list[str] = []
    for test in task["tests"]:
        test_code = test["code"].replace("{fn}", p["fn_name"])
        test_ns: dict[str, Any] = {p["fn_name"]: fn}
        try:
            exec(compile(test_code, f"<test:{test['name']}>", "exec"), test_ns)
            passed += 1
        except Exception as e:  # noqa: BLE001
            failed.append(f"{test['name']}: {type(e).__name__}: {e}")
    total = len(task["tests"])
    if passed == total:
        return True, f"{passed}/{total} passed"
    return False, f"{passed}/{total} passed; failures: " + "; ".join(failed)


def build_one(spec_path: Path, verify: bool) -> bool:
    p = load_spec(spec_path)
    t = write_task(p)
    tmpl = write_template(p)
    sol = write_solution(p)
    print(f"✓ {p['id']}: wrote {t.name}, {tmpl.name}, {sol.name}")
    if verify:
        ok, msg = verify_spec(p)
        marker = "✓" if ok else "✗"
        print(f"  {marker} verify: {msg}")
        return ok
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?", help="Build only this spec (default: all)")
    ap.add_argument("--verify", action="store_true", help="Run judge on each solution after building")
    args = ap.parse_args()

    if not SPECS_DIR.exists():
        print(f"No specs directory at {SPECS_DIR}", file=sys.stderr)
        return 1

    specs = sorted(SPECS_DIR.glob("*.py"))
    specs = [s for s in specs if not s.name.startswith("_")]
    if args.id:
        specs = [s for s in specs if s.stem == args.id]
        if not specs:
            print(f"No spec named '{args.id}' in {SPECS_DIR}", file=sys.stderr)
            return 1

    all_ok = True
    for spec_path in specs:
        ok = build_one(spec_path, verify=args.verify)
        all_ok = all_ok and ok

    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
