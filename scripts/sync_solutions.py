#!/usr/bin/env python3
"""Sync template intro markdown into solution notebooks (idempotent).

For each templates/{file}.ipynb, copy cell 0 (intro markdown) into the
corresponding solutions/{file}_solution.ipynb, with two adjustments:

- Title prefix: "🟢 Easy:" / "🟡 Medium:" / "🟠 Medium:" / "🔴 Hard:"
  → "✅ Solution:"
- Colab badge URL: templates/X.ipynb → solutions/X_solution.ipynb

Use this after editing a template's intro markdown so the corresponding
solution notebook stays self-contained.

Note: spec-driven problems (16 in problem_specs/) are also handled
end-to-end by `scripts/build.py`. Running this script after build.py is
a no-op for those, so it's safe to always run.

Usage:
    python scripts/sync_solutions.py          # sync where solution drifts
    python scripts/sync_solutions.py --force  # rewrite even if in sync
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"
SOLUTIONS_DIR = ROOT / "solutions"

TITLE_RE = re.compile(r"^# [🟢🟡🟠🔴⬜️] ?(Easy|Medium|Hard): ", re.MULTILINE)
TEMPLATES_URL_RE = re.compile(r"(blob/master/)templates/(\w+\.ipynb)")


def transform_intro(tmpl_intro: str) -> str:
    new_intro = TITLE_RE.sub("# ✅ Solution: ", tmpl_intro)

    def fix_url(m: re.Match) -> str:
        prefix = m.group(1)
        fn = m.group(2)
        sol_fn = fn.replace(".ipynb", "_solution.ipynb")
        return f"{prefix}solutions/{sol_fn}"

    new_intro = TEMPLATES_URL_RE.sub(fix_url, new_intro)
    return new_intro


def sync(force: bool) -> int:
    updated = 0
    in_sync = 0
    no_solution = 0
    for tmpl_path in sorted(TEMPLATES_DIR.glob("*.ipynb")):
        sol_name = tmpl_path.stem + "_solution.ipynb"
        sol_path = SOLUTIONS_DIR / sol_name
        if not sol_path.exists():
            no_solution += 1
            continue

        tmpl_nb = json.loads(tmpl_path.read_text())
        sol_nb = json.loads(sol_path.read_text())
        tmpl_intro = "".join(tmpl_nb["cells"][0]["source"])
        new_intro = transform_intro(tmpl_intro)

        sol_intro = "".join(sol_nb["cells"][0]["source"])
        if not force and sol_intro == new_intro:
            in_sync += 1
            continue

        sol_nb["cells"][0]["source"] = new_intro.splitlines(keepends=True)
        sol_path.write_text(
            json.dumps(sol_nb, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        updated += 1
        print(f"  ✓ {sol_name}")

    print(
        f"\nUpdated {updated}, in sync {in_sync}, no solution file {no_solution}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true", help="Rewrite even if intro is already in sync"
    )
    args = ap.parse_args()
    return sync(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
