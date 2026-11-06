"""Real local checkpoints, bounded inputs, temporary outputs and no network access."""

import numpy as np
import pandas as pd
import pytest

from src import advanced, api


@pytest.mark.integration
def test_semantic_ranking_and_threshold():
    records = [
        {"complaint_id": "1", "narrative": "My mortgage escrow payment increased unexpectedly."},
        {"complaint_id": "2", "narrative": "A debt collector repeatedly called my phone."},
    ]
    engine = advanced.SemanticIndex(records)
    assert engine.index.ntotal == 2
    hits = engine.search("mortgage escrow payment", k=10)
    assert hits[0]["complaint_id"] == "1"
    assert hits[0]["score"] >= hits[1]["score"]
    assert engine.search("mortgage escrow payment", threshold=1.1) == []
    with pytest.raises(ValueError):
        engine.search(" ")
    with pytest.raises(ValueError):
        engine.search("query", k=0)
    with pytest.raises(ValueError):
        advanced.SemanticIndex([])


@pytest.mark.integration
def test_real_generator_cache_and_generation():
    advanced.generator.cache_clear()
    first = advanced.generator()
    assert first is advanced.generator()

    class Evidence:
        def search(self, *args, **kwargs):
            return [{"complaint_id": "42", "excerpt": "The consumer disputes an escrow payment."}]

    result = advanced.answer_question(Evidence(), "What does the consumer dispute?")
    assert result["sources"][0]["complaint_id"] == "42"
    assert result["answer"]
    assert result["mode"] in {"flan-t5", "abstention_invalid_citations"}


@pytest.mark.parametrize(
    "draft,valid",
    [
        ("Escrow disputed [42]", True),
        ("Unsupported [999]", False),
        ("No references", False),
        ("Mixed [42] [999]", False),
    ],
)
def test_generation_citation_guard(draft, valid, monkeypatch):
    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            assert "[42]" in prompt and kwargs["max_length"] == 512
            return {"input_ids": "test"}

        def decode(self, *args, **kwargs):
            return draft

    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return [[0]]

    class Evidence:
        def search(self, question, k, threshold):
            assert k == 3 and threshold == 0.35
            return [{"complaint_id": "42", "excerpt": "Escrow disputed"}]

    monkeypatch.setattr(advanced, "generator", lambda: (Tokenizer(), Model()))
    result = advanced.answer_question(Evidence(), "escrow?")
    assert result["citations_valid"] is valid
    if not valid:
        assert result["generated_draft"] == draft
        assert result["answer"] != draft


@pytest.mark.integration
def test_distilbert_training_freeze_and_saved_reload(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    real_root = advanced.ROOT
    model_loader = AutoModelForSequenceClassification.from_pretrained
    tokenizer_loader = AutoTokenizer.from_pretrained
    captured = {}

    def load_model(path, **kwargs):
        model = model_loader(real_root / ".models/distilbert", local_files_only=True, **kwargs)
        captured["model"] = model
        captured["before"] = model.classifier.weight.detach().clone()
        return model

    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", load_model)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda path: tokenizer_loader(real_root / ".models/distilbert", local_files_only=True),
    )
    monkeypatch.setattr(advanced, "ROOT", tmp_path)
    training = pd.DataFrame(
        {"narrative": ["card fee dispute", "mortgage escrow payment"] * 5, "product": ["Card", "Mortgage"] * 5}
    )
    validation = training.iloc[:2]
    predictions, history = advanced.adapt_distilbert(training, validation, epochs=2)
    model = captured["model"]
    assert len(predictions) == 2 and set(predictions) <= {"Card", "Mortgage"}
    assert np.isfinite(history.training_loss).all() and len(history) == 2
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.distilbert.parameters())
    assert not torch.equal(captured["before"], model.classifier.weight.detach())
    restored = model_loader(tmp_path / "artifacts/distilbert", local_files_only=True)
    assert restored.config.id2label == model.config.id2label
    assert torch.equal(restored.classifier.weight, model.classifier.weight)


@pytest.mark.integration
def test_api_loads_local_artifacts_once():
    api.service.cache_clear()
    models = api.service()
    assert models is api.service()
    assert models[1].index.ntotal == len(models[1].passages)
    assert models[0].predict(["mortgage escrow"])[0] in models[0].classes_
    api.service.cache_clear()


def test_api_missing_artifacts_is_explicit(tmp_path, monkeypatch):
    api.service.cache_clear()
    monkeypatch.setattr(api, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        api.service()
    api.service.cache_clear()


def test_ollama_requires_configured_model(monkeypatch):
    from src import rag

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(rag, "search", lambda *args: [{"complaint_id": "1", "excerpt": "card fee"}])
    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        rag.ask({}, "card", generate=True)
