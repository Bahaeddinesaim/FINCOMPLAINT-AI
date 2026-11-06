"""Download the public pretrained models into project-local directories."""

import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["HF_HOME"] = str(root / ".models")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    specs = [
        ("sentence-transformers/all-MiniLM-L6-v2", "minilm", ["*.json", "*.safetensors", "vocab.txt", "*.txt"]),
        ("google/flan-t5-small", "flan", ["*.json", "*.safetensors", "spiece.model"]),
        ("distilbert/distilbert-base-uncased", "distilbert", ["*.json", "*.safetensors", "vocab.txt"]),
    ]
    for name, directory, files in specs:
        snapshot_download(name, local_dir=root / ".models" / directory, allow_patterns=files)


if __name__ == "__main__":
    main()
