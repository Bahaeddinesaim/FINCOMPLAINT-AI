from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.rag import ask
from src.retriever import build, search


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        database = root / "data.sqlite"
        with closing(sqlite3.connect(database)) as db:
            db.execute("CREATE TABLE complaints (complaint_id TEXT, date_received TEXT, product TEXT, "
                       "issue TEXT, narrative TEXT, source_file TEXT, split TEXT, text_hash TEXT)")
            db.executemany("INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?)", [
                ("1", "2025-01-01", "Card", "Fee", "card charge disputed payment", "source.csv", "train", "a"),
                ("2", "2025-01-01", "Mortgage", "Loan", "mortgage escrow housing loan", "source.csv", "train", "b"),
                ("3", "2025-02-01", "Card", "Fee", "validationonly card", "source.csv", "validation", "c"),
                ("4", "2026-01-01", "Card", "Fee", "futureonly card", "future.csv", "test", "d")])
            db.commit()
        self.index = build(database, root / "index.joblib")

    def test_training_index_and_ranked_sources(self):
        self.assertEqual({d["complaint_id"] for d in self.index["documents"]}, {"1", "2"})
        self.assertNotIn("futureonly", self.index["vectorizer"].vocabulary_)
        self.assertNotIn("validationonly", self.index["vectorizer"].vocabulary_)
        self.assertEqual(search(self.index, "mortgage escrow")[0]["complaint_id"], "2")

    def test_oov_abstention_and_extractive_traceability(self):
        self.assertEqual(ask(self.index, "quasar nebula")["mode"], "abstention")
        result = ask(self.index, "card charge")
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["sources"][0]["source_file"], "source.csv")
        self.assertIn(result["sources"][0]["excerpt"], result["answer"])
        with self.assertRaises(ValueError):
            search(self.index, " ")
        with self.assertRaises(ValueError):
            search(self.index, "card", 0)

    @patch.dict("os.environ", {"OLLAMA_MODEL": "mock-model"})
    @patch("src.rag.urlopen")
    def test_generated_citation_ids_are_checked(self, request):
        response = MagicMock()
        response.read.return_value = json.dumps({"message": {"content": "Unsupported [999]"}}).encode()
        request.return_value.__enter__.return_value = response
        result = ask(self.index, "mortgage", generate=True)
        self.assertFalse(result["citations_valid"])
        sent = json.loads(request.call_args.args[0].data)
        self.assertFalse(sent["stream"])
        self.assertIn("[complaint_id]", sent["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
