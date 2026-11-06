import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.ingestion import prepare
from src.preprocessing import narrative_hash


class IngestionTests(unittest.TestCase):
    def test_quality_dedup_and_temporal_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ["Complaint ID", "Date received", "Product", "Consumer complaint narrative"]
            pd.DataFrame(
                [
                    ["1", "2025-02-01", "Credit card", "A disputed payment"],
                    ["1", "2025-02-01", "Credit card", "Different wording"],
                    ["2", "bad", "Credit card", "Invalid date"],
                    ["3", "2024-02-01", "Credit card", "Old complaint"],
                    ["4", "2025-02-01", "Credit card", "  "],
                ],
                columns=columns,
            ).to_csv(root / "CCDB_Export_9_2025.csv", index=False)
            pd.DataFrame(
                [
                    ["5", "2026-02-01", "Credit card", " a DISPUTED   payment "],
                    ["6", "2026-02-01", "Mortgage", "Mortgage servicing problem"],
                    ["7", "2026-02-01", "", "Missing label"],
                ],
                columns=columns,
            ).to_csv(root / "CCDB_Export_10_2026.csv", index=False)
            report = prepare(root, root / "out.sqlite", root / "report.json", chunksize=2)
            self.assertEqual(
                report["counts"],
                {
                    "rows_read": 8,
                    "kept": 2,
                    "duplicate_id": 1,
                    "invalid_date": 1,
                    "outside_period": 1,
                    "empty_narrative": 1,
                    "duplicate_text": 1,
                    "missing_id_or_product": 1,
                },
            )
            with closing(sqlite3.connect(root / "out.sqlite")) as db:
                records = db.execute("SELECT complaint_id, split FROM complaints ORDER BY complaint_id").fetchall()
            self.assertIn(records[0][1], ("train", "validation"))
            self.assertEqual(records[1], ("6", "test"))
            self.assertEqual(json.loads((root / "report.json").read_text())["counts"]["kept"], 2)

    def test_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CCDB_pointer.csv").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Git LFS"):
                prepare(root, root / "out.sqlite", root / "report.json")
            self.assertFalse((root / "out.sqlite").exists())

    def test_normalized_hash(self):
        self.assertEqual(narrative_hash(" Hello\nworld "), narrative_hash("hello world"))


if __name__ == "__main__":
    unittest.main()
