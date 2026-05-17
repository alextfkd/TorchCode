#!/usr/bin/env python3
"""Verify every solution notebook in solutions/ passes its task's tests.

Extracts the imports + solution code from each `solutions/*_solution.ipynb`,
runs them, then executes the corresponding task's test suite from
`torch_judge.tasks`. Reports per-problem pass count and a final summary.

Exit code: 0 if all solutions pass all tests, 1 if any failure.

Usage:
    python scripts/verify_all_solutions.py            # silent on pass, terse on fail
    python scripts/verify_all_solutions.py -v         # show failure details
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS_DIR = ROOT / "solutions"

sys.path.insert(0, str(ROOT))


def extract_solution_code(sol_nb: dict) -> tuple[str, str]:
    """Return (imports_code, solution_code) extracted from a solution ipynb.

    Heuristics:
    - imports cell: a small code cell containing `import torch` (and only imports)
    - solution cell: the code cell that starts with `# ✅ SOLUTION`
    """
    imports_code = ""
    solution_code = ""
    for cell in sol_nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "google.colab" in src:
            continue
        if src.startswith("from torch_judge import check"):
            continue
        if "# ✅ SOLUTION" in src:
            solution_code = src.replace("# ✅ SOLUTION", "", 1).strip()
            continue
        if not solution_code and "import" in src and len(src) < 200:
            imports_code = src
    return imports_code, solution_code


def run_tests(task: dict[str, Any], fn_name: str, fn: Any) -> tuple[int, int, list[str]]:
    passed = 0
    failures: list[str] = []
    for test in task["tests"]:
        test_code = test["code"].replace("{fn}", fn_name)
        test_ns: dict[str, Any] = {fn_name: fn}
        try:
            exec(compile(test_code, f"<test:{test['name']}>", "exec"), test_ns)
            passed += 1
        except Exception as e:  # noqa: BLE001
            failures.append(f"{test['name']}: {type(e).__name__}: {e}")
    return passed, len(task["tests"]), failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="Show failure details"
    )
    args = ap.parse_args()

    from torch_judge.tasks import TASKS

    all_pass = True
    total_pass = 0
    total_tests = 0
    sol_count = 0

    for sol_path in sorted(SOLUTIONS_DIR.glob("*_solution.ipynb")):
        sol_count += 1
        stem = sol_path.stem.replace("_solution", "")
        task_id = stem.split("_", 1)[1]
        task = TASKS.get(task_id)
        if task is None:
            print(f"  ? {task_id:30s} task '{task_id}' not in registry")
            continue

        sol_nb = json.loads(sol_path.read_text())
        imports_code, solution_code = extract_solution_code(sol_nb)

        ns: dict[str, Any] = {}
        try:
            if imports_code:
                exec(compile(imports_code, "<imports>", "exec"), ns)
            if not solution_code:
                print(f"  ✗ {task_id:30s} no '# ✅ SOLUTION' cell found")
                all_pass = False
                continue
            exec(compile(solution_code, "<solution>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {task_id:30s} solution exec error — {type(e).__name__}: {e}")
            all_pass = False
            continue

        fn_name = task["function_name"]
        fn = ns.get(fn_name)
        if fn is None:
            print(f"  ✗ {task_id:30s} function/class '{fn_name}' not defined")
            all_pass = False
            continue

        passed, total, failures = run_tests(task, fn_name, fn)
        total_pass += passed
        total_tests += total
        if passed == total:
            print(f"  ✓ {task_id:30s} {passed}/{total}")
        else:
            all_pass = False
            print(f"  ✗ {task_id:30s} {passed}/{total}")
            if args.verbose:
                for f in failures:
                    print(f"      - {f}")

    print(
        f"\nChecked {sol_count} solutions: {total_pass}/{total_tests} tests passing"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
