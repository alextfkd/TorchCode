#!/usr/bin/env python3
"""Add Colab-only pip install of torch-judge to all notebooks that use torch_judge."""

import json
from pathlib import Path

INSTALL_CELL_SOURCE = [
    "# Install torch-judge in Colab (no-op in JupyterLab/Docker)\n",
    "try:\n",
    "    import google.colab\n",
    "    get_ipython().run_line_magic('pip', 'install -q --force-reinstall --no-deps git+https://github.com/alextfkd/TorchCode.git')\n",
    "except ImportError:\n",
    "    pass\n",
]

MARKER = "get_ipython().run_line_magic('pip', 'install"


def has_torch_judge(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        flat = "".join(src) if isinstance(src, list) else str(src)
        if "torch_judge" in flat:
            return True
    return False


def already_has_install(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        flat = "".join(src) if isinstance(src, list) else str(src)
        if MARKER in flat and "torch-judge" in flat:
            return True
    return False


def process_notebook(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if not has_torch_judge(nb):
        return False
    if already_has_install(nb):
        return False

    cells = nb["cells"]
    if not cells:
        return False

    # Insert install cell at index 1 (after first cell, usually markdown title)
    install_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": INSTALL_CELL_SOURCE,
        "outputs": [],
        "execution_count": None,
    }
    cells.insert(1, install_cell)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    return True


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    updated = 0
    for pattern in ["templates/*.ipynb", "solutions/*.ipynb"]:
        for path in sorted(root.glob(pattern)):
            if process_notebook(path):
                print(f"  + {path.relative_to(root)}")
                updated += 1
    print(f"Updated {updated} notebooks.")


if __name__ == "__main__":
    main()
