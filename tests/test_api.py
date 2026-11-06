"""API contract tests with dependency injection, no model downloads."""

from fastapi.testclient import TestClient

from src.api import app, service


def test_http_contracts_and_validation():
    class Classifier:
        def predict(self, texts):
            return ["Card"] * len(texts)

    class EmptySearch:
        def search(self, *args, **kwargs):
            return []

    app.dependency_overrides[service] = lambda: (Classifier(), EmptySearch())
    try:
        with TestClient(app) as client:
            assert client.post("/classify", json={"text": "card"}).json()["product"] == "Card"
            assert client.post("/search", json={"text": "card"}).json()["sources"] == []
            assert client.post("/ask", json={"text": "unknown"}).json()["mode"] == "abstention"
            assert client.post("/ask", json={"text": " "}).status_code == 422
            assert client.post("/search", json={"text": "x", "k": 0}).status_code == 422
    finally:
        app.dependency_overrides.clear()
