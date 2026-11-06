"""Execute notebooks and preserve their outputs: python -m scripts.execute_notebooks."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    with tempfile.TemporaryDirectory() as directory:
        previous = os.environ.get("IPYTHONDIR")
        os.environ["IPYTHONDIR"] = directory
        try:
            for path in sorted(Path("notebooks").glob("*.ipynb")):
                print(f"Execution : {path}", flush=True)
                notebook = nbformat.read(path, as_version=4)
                NotebookClient(
                    notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}
                ).execute()
                nbformat.write(notebook, path)
        finally:
            if previous is None:
                os.environ.pop("IPYTHONDIR", None)
            else:
                os.environ["IPYTHONDIR"] = previous


if __name__ == "__main__":
    main()
