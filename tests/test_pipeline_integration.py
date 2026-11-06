"""Exercise the repository's public pipeline with a temporary CSV corpus."""

import sqlite3

import pandas as pd
import pytest

from src.classifier import train
from src.ingestion import prepare
from src.rag import ask
from src.retriever import build


@pytest.mark.integration
def test_csv_sqlite_training_and_sourced_answer(tmp_path):
    rows = []
    for index in range(40):
        product = "Mortgage" if index % 2 else "Card"
        topic = "mortgage escrow loan" if index % 2 else "credit card charge"
        rows.append(
            {
                "Complaint ID": str(index),
                "Date received": "2025-04-01",
                "Product": product,
                "Consumer complaint narrative": f"{topic} problem case {index}",
            }
        )
    rows.append(
        {
            "Complaint ID": "future",
            "Date received": "2026-04-01",
            "Product": "Card",
            "Consumer complaint narrative": "futureonly disputed account",
        }
    )
    pd.DataFrame(rows).to_csv(tmp_path / "CCDB_Export_9.csv", index=False)
    database = tmp_path / "corpus.sqlite"
    quality = prepare(tmp_path, database, tmp_path / "quality.json", chunksize=7)
    assert quality["counts"]["kept"] == 41
    metrics = train(database, tmp_path / "artifacts", tmp_path / "reports", 100, 100)
    assert metrics["test_evaluated"] is False
    index = build(database, tmp_path / "retriever.joblib")
    assert "futureonly" not in index["vectorizer"].vocabulary_
    result = ask(index, "mortgage escrow")
    assert result["sources"] and result["citations_valid"]
    with sqlite3.connect(database) as connection:
        for source in result["sources"]:
            narrative, split = connection.execute(
                "SELECT narrative, split FROM complaints WHERE complaint_id=?",
                (source["complaint_id"],),
            ).fetchone()
            assert split == "train" and narrative.startswith(source["excerpt"])
