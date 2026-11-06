import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import joblib

from src.classifier import train


class ClassifierTests(unittest.TestCase):
    def test_training_excludes_validation_and_temporal_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "corpus.sqlite"
            with closing(sqlite3.connect(database)) as db:
                db.execute("CREATE TABLE complaints (narrative TEXT, product TEXT, split TEXT, text_hash TEXT)")
                db.executemany(
                    "INSERT INTO complaints VALUES (?,?,?,?)",
                    [
                        ("card charge disputed payment", "Card", "train", "a"),
                        ("card charge disputed payment late", "Card", "train", "b"),
                        ("mortgage servicing housing loan", "Mortgage", "train", "c"),
                        ("mortgage servicing housing loan late", "Mortgage", "train", "d"),
                        ("card charge validationonly", "Card", "validation", "e"),
                        ("mortgage loan validationonly", "Mortgage", "validation", "f"),
                        ("futureonly futureonly", "Unseen", "test", "g"),
                    ],
                )
                db.commit()
            results = train(database, root / "models", root / "reports", 10, 10)
            self.assertFalse(results["test_evaluated"])
            self.assertNotIn("test", results["splits"])
            self.assertEqual(results["splits"]["validation"]["rows"], 2)
            model = joblib.load(root / "models/baseline.joblib")
            self.assertNotIn("validationonly", model["tfidf"].vocabulary_)
            self.assertNotIn("futureonly", model["tfidf"].vocabulary_)
            self.assertNotIn("Unseen", model.classes_)


if __name__ == "__main__":
    unittest.main()
