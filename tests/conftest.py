"""Isolated miniature corpus shared by integration and command tests."""

import sqlite3

import pytest


@pytest.fixture
def corpus_db(tmp_path):
    path = tmp_path / "corpus.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE complaints (complaint_id TEXT, date_received TEXT, product TEXT, "
            "issue TEXT, narrative TEXT, source_file TEXT, split TEXT, text_hash TEXT)"
        )
        db.executemany(
            "INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?)",
            [
                ("1", "2025-01-01", "Card", "Fee", "card charge disputed payment", "x.csv", "train", "a"),
                ("2", "2025-01-02", "Card", "Fee", "card charge disputed payment late", "x.csv", "train", "b"),
                ("3", "2025-01-03", "Mortgage", "Escrow", "mortgage escrow housing loan", "x.csv", "train", "c"),
                ("4", "2025-01-04", "Mortgage", "Escrow", "mortgage escrow housing loan late", "x.csv", "train", "d"),
                ("5", "2025-02-01", "Card", "Fee", "card validationonly", "x.csv", "validation", "e"),
                ("6", "2026-01-01", "Unseen", "Other", "futureonly report", "y.csv", "test", "f"),
            ],
        )
    return path
