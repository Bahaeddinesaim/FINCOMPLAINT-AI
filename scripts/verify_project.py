"""Run quality checks and offline model tests; optionally refresh notebook test outputs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat


def verify(root):
    root = Path(root)
    runtime = root / ".runtime"
    runtime.mkdir(exist_ok=True)
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TEMP": str(runtime),
        "TMP": str(runtime),
        "NO_COLOR": "1",
        "FORCE_COLOR": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=src",
            "--cov-branch",
            "--cov-fail-under=100",
            "--cov-report=term-missing",
            "--cov-report=html:reports/final/coverage",
            "--cov-report=json:reports/final/coverage.json",
            "--junitxml=reports/final/pytest.xml",
        ],
    ]
    outputs = []
    for command in commands:
        result = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
        print(output, end="")
        outputs.append(output)
        if result.returncode:
            raise RuntimeError(f"Check failed: {command}")
    return "".join(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-notebook", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = verify(root)
    if args.update_notebook:
        path = root / "notebooks/00_projet_fincomplaint.ipynb"
        notebook = nbformat.read(path, as_version=4)
        matches = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and ("'pytest'" in cell.source or '"pytest"' in cell.source or "verify_project" in cell.source)
        ]
        if len(matches) != 1:
            raise ValueError("Expected one verification cell")
        cell = matches[0]
        cell.source = "from scripts.verify_project import verify\n\nverify(ROOT)\n"
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output)]
        nbformat.validate(notebook)
        nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
