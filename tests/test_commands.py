"""CLI behavior, real artifacts and explicit holdout evaluation."""

import json
import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

from src import classifier, demo, ingestion
from src.retriever import build


@pytest.mark.parametrize("module", ["classifier", "ingestion", "demo"])
def test_module_entrypoints_help(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    script = Path(__file__).resolve().parents[1] / "src" / f"{module}.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("action", ["classify", "search", "ask"])
def test_cli_real_artifacts(action, corpus_db, tmp_path, monkeypatch, capsys):
    directory = tmp_path / "models"
    classifier.train(corpus_db, directory, tmp_path / "reports", 20, 20)
    build(corpus_db, directory / "retriever.joblib")
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["demo", action, "mortgage escrow", "--artifacts", str(directory)])
    assert demo.main() == 0
    result = json.loads(capsys.readouterr().out)
    if action == "classify":
        assert result["product"] == "Mortgage"
    elif action == "search":
        assert result[0]["complaint_id"] in {"3", "4"}
    else:
        assert result["sources"] and result["citations_valid"]


def test_cli_missing_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo", "classify", "card", "--artifacts", str(tmp_path)])
    assert demo.main() == 1
    assert "Demonstrateur" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["classify", " "], ["search", "card", "--top-k", "0"]])
def test_cli_rejects_invalid_input(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo", *arguments])
    with pytest.raises(SystemExit) as error:
        demo.main()
    assert error.value.code == 2


def test_classifier_main_explicit_test(corpus_db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classifier",
            "--database",
            str(corpus_db),
            "--output",
            str(tmp_path / "models"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--evaluate-test",
        ],
    )
    classifier.main()
    report = json.loads((tmp_path / "reports/baseline_metrics.json").read_text())
    assert report["test_evaluated"] and report["splits"]["test"]["unseen_labels"] == ["Unseen"]
    assert (tmp_path / "reports/confusion_test.csv").exists()


@pytest.mark.parametrize("flag,value", [("--train-limit", "1"), ("--eval-limit", "0")])
def test_classifier_invalid_limits(flag, value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["classifier", flag, value])
    with pytest.raises(SystemExit):
        classifier.main()


def test_classifier_single_class_and_empty_evaluation(corpus_db, tmp_path):
    with sqlite3.connect(corpus_db) as db:
        db.execute("DELETE FROM complaints WHERE split != 'train'")
    result = classifier.train(corpus_db, tmp_path / "m", tmp_path / "r", 20, 20, True)
    assert result["splits"] == {"validation": {"rows": 0}, "test": {"rows": 0}}
    with sqlite3.connect(corpus_db) as db:
        db.execute("DELETE FROM complaints WHERE product='Mortgage'")
    with pytest.raises(ValueError, match="deux classes"):
        classifier.train(corpus_db, tmp_path / "m", tmp_path / "r", 20, 20)


def test_ingestion_main_and_missing_schema(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(sys, "argv", ["ingestion", "--chunksize", "0"])
    with pytest.raises(SystemExit):
        ingestion.main()
    with pytest.raises(ValueError, match="Aucun export"):
        ingestion.prepare(tmp_path, tmp_path / "db", tmp_path / "report")
    output = tmp_path / "preserved.sqlite"
    output.write_bytes(b"existing artifact")
    pd.DataFrame({"bad": ["schema"]}).to_csv(tmp_path / "CCDB.csv", index=False)
    with pytest.raises(ValueError, match="colonnes absentes"):
        ingestion.prepare(tmp_path, output, tmp_path / "report")
    assert output.read_bytes() == b"existing artifact"
    pd.DataFrame(
        {
            "Complaint ID": ["1"],
            "Date received": ["2025-01-01"],
            "Product": ["Card"],
            "Consumer complaint narrative": ["card fee"],
        }
    ).to_csv(tmp_path / "CCDB.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingestion", "--input-dir", str(tmp_path), "--output", str(output), "--report", str(tmp_path / "report.json")],
    )
    ingestion.main()
    assert json.loads((tmp_path / "report.json").read_text())["counts"]["kept"] == 1


def test_retriever_empty_and_bad_limit(corpus_db, tmp_path):
    with pytest.raises(ValueError, match="positif"):
        build(corpus_db, tmp_path / "index", limit=0)
    with sqlite3.connect(corpus_db) as db:
        db.execute("DELETE FROM complaints")
    with pytest.raises(ValueError, match="vide"):
        build(corpus_db, tmp_path / "index")
