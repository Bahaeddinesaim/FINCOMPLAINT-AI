"""Execute the complete project notebook and save outputs even on failure."""

import asyncio
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".runtime"
    runtime.mkdir(exist_ok=True)
    for variable in ["IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "TEMP", "TMP"]:
        os.environ[variable] = str(runtime)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    manager = KernelManager(connection_file=str(runtime / "connection.json"))
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="FinComplaint",
        language="python",
    )
    path = root / "notebooks" / "00_projet_fincomplaint.ipynb"
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(root)}}).execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    print("Notebook execute et enregistre.")


if __name__ == "__main__":
    main()
