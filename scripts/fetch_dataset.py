"""Download public repository LFS files with SHA-256 verification."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPOSITORY = "Bahaeddinesaim/FINCOMPLAINT-AI"


def download(entry, destination):
    name = entry["path"]
    fields = dict(line.split(" ", 1) for line in entry["pointer"].strip().splitlines())
    digest = fields["oid"].removeprefix("sha256:")
    size = int(fields["size"])
    target = destination / name
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual == digest:
            return {"file": name, "bytes": size, "sha256": digest, "verified": True}
    stage = target.with_suffix(".csv.partial")
    url = f"https://media.githubusercontent.com/media/{REPOSITORY}/main/{name}"
    print(f"Download: {name} ({size / 1e6:.1f} MB)", flush=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with stage.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    with stage.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if stage.stat().st_size != size or actual != digest:
        raise ValueError(f"LFS integrity mismatch: {name}")
    stage.replace(target)
    print(f"Verified: {name}", flush=True)
    return {"file": name, "bytes": size, "sha256": digest, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("."))
    parser.add_argument("--pointer-dir", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": p.name,
            "pointer": subprocess.check_output(
                ["git", "-C", str(args.pointer_dir), "show", f"HEAD:{p.name}"], text=True
            ),
        }
        for p in args.pointer_dir.glob("CCDB*.csv")
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda e: download(e, args.destination), entries))
    manifest = {"repository": REPOSITORY, "files": results}
    path = args.destination / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
