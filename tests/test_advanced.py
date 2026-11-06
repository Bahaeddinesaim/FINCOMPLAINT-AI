"""Cheap unit tests for model-independent contracts."""

import pytest

from src.advanced import answer_question, chunk_records, ranking_metrics


def test_chunk_offsets_and_sources():
    records = [{"complaint_id": "1", "narrative": "a b c d e", "source_file": "x.csv"}]
    chunks = chunk_records(records, words=3, overlap=1)
    assert [p["excerpt"] for p in chunks] == ["a b c", "c d e"]
    assert [p["offset_words"] for p in chunks] == [0, 2]
    assert all(p["complaint_id"] == "1" for p in chunks)
    with pytest.raises(ValueError):
        chunk_records(records, words=2, overlap=2)
    assert chunk_records([{"complaint_id": "empty", "narrative": ""}]) == []


def test_metrics_deduplicate_recall():
    assert ranking_metrics(["x", "a", "a"], {"a", "b"}) == {"recall": 0.5, "mrr": 0.5}
    with pytest.raises(ValueError):
        ranking_metrics([], set())


def test_no_model_loaded_for_abstention():
    class EmptyIndex:
        def search(self, *args, **kwargs):
            return []

    result = answer_question(EmptyIndex(), "unknown")
    assert result["mode"] == "abstention" and not result["sources"]
