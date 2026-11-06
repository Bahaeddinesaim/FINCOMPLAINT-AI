"""Stream CFPB exports into SQLite with quality reports and temporal partitions.

Run: python -m src.ingestion --input-dir .
"""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd

from src.preprocessing import narrative_hash, normalize_text

REQUIRED = {"Complaint ID", "Date received", "Product", "Consumer complaint narrative"}


def prepare(input_dir: Path, output: Path, report_path: Path, chunksize: int = 10000) -> dict:
    # Export sequence is numeric (9 precedes 10), so earlier periods win ties.
    files = sorted(input_dir.glob("CCDB*.csv"), key=lambda p: (
        int(re.search(r"CCDB_Export_(\d+)", p.name).group(1))
        if re.search(r"CCDB_Export_(\d+)", p.name) else 0, p.name))
    if not files:
        raise ValueError(f"Aucun export CCDB*.csv dans {input_dir}")
    # Validate every input before replacing a previous generated corpus.
    inventory = []
    for path in files:
        with path.open("rb") as stream:
            if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path.name}: pointeur Git LFS. Executer git lfs pull.")
        columns = pd.read_csv(path, nrows=0).columns
        missing = REQUIRED - set(columns)
        if missing:
            raise ValueError(f"{path.name}: colonnes absentes: {sorted(missing)}")
        inventory.append({"file": path.name, "bytes": path.stat().st_size, "columns": list(columns)})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(".building.sqlite")
    counts = Counter()
    with closing(sqlite3.connect(staging)) as db:
        db.execute("DROP TABLE IF EXISTS complaints")
        db.execute("""CREATE TABLE complaints (
            complaint_id TEXT PRIMARY KEY, date_received TEXT NOT NULL,
            product TEXT NOT NULL, issue TEXT NOT NULL, narrative TEXT NOT NULL,
            text_hash TEXT UNIQUE NOT NULL, split TEXT NOT NULL,
            source_file TEXT NOT NULL, char_count INTEGER NOT NULL)""")
        for entry, path in zip(inventory, files):
            file_counts = Counter()
            print(f"Lecture {path.name}", flush=True)
            for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize):
                dates = pd.to_datetime(chunk["Date received"], errors="coerce", format="mixed")
                for record, date in zip(chunk.to_dict("records"), dates):
                    file_counts["rows_read"] += 1
                    if pd.isna(date):
                        file_counts["invalid_date"] += 1
                        continue
                    if date.year not in (2025, 2026):
                        file_counts["outside_period"] += 1
                        continue
                    text = normalize_text(record["Consumer complaint narrative"])
                    if not text:
                        file_counts["empty_narrative"] += 1
                        continue
                    identifier = record["Complaint ID"].strip()
                    product = record["Product"].strip()
                    if not identifier or not product:
                        file_counts["missing_id_or_product"] += 1
                        continue
                    digest = narrative_hash(text)
                    if db.execute("SELECT 1 FROM complaints WHERE complaint_id=?", (identifier,)).fetchone():
                        file_counts["duplicate_id"] += 1
                        continue
                    if db.execute("SELECT 1 FROM complaints WHERE text_hash=?", (digest,)).fetchone():
                        file_counts["duplicate_text"] += 1
                        continue
                    # Only 2025 is used for fitting and model selection.
                    split = "test" if date.year == 2026 else (
                        "validation" if int(digest[:8], 16) % 5 == 0 else "train")
                    db.execute("INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?,?)", (
                        identifier, date.strftime("%Y-%m-%d"), product,
                        record.get("Issue", "").strip(), record["Consumer complaint narrative"],
                        digest, split, path.name, len(text)))
                    file_counts["kept"] += 1
                db.commit()
            entry["counts"] = dict(file_counts)
            counts.update(file_counts)
        db.execute("CREATE INDEX complaints_split ON complaints(split)")
        db.commit()
        def rows(query):
            cursor = db.execute(query)
            return [dict(zip([col[0] for col in cursor.description], row)) for row in cursor]
        report = {
            "counts": dict(counts), "files": inventory,
            "date_range": rows("SELECT MIN(date_received) AS first, MAX(date_received) AS last FROM complaints"),
            "by_split": rows("SELECT split, COUNT(*) AS count FROM complaints GROUP BY split"),
            "by_product": rows("SELECT product, split, COUNT(*) AS count FROM complaints GROUP BY product, split ORDER BY count DESC"),
            "lengths": rows("SELECT split, MIN(char_count) AS min, AVG(char_count) AS mean, MAX(char_count) AS max FROM complaints GROUP BY split"),
            "limitations": ["Quasi-doublons non detectes; seuls ID et textes normalises identiques sont dedupliques.",
                             "La premiere occurrence dans l'ordre numerique des exports est conservee.",
                             "Validation deterministe par hash du texte, environ 20 % de 2025."],
        }
    staging.replace(output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/processed/complaints.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_quality.json"))
    parser.add_argument("--chunksize", type=int, default=10000)
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize doit etre positif")
    report = prepare(args.input_dir, args.output, args.report, args.chunksize)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
