"""Web assets and measured project metadata, without loading inference models."""

import json

from fastapi.testclient import TestClient

from src import api


def test_workspace_and_assets():
    with TestClient(api.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "FinComplaint AI" in response.text
        assert 'id="analysis-form"' in response.text
        for filename in ["app.js", "style.css", "mark.png"]:
            assert client.get(f"/static/{filename}").status_code == 200


def test_project_metadata_from_reports(tmp_path, monkeypatch):
    reports = tmp_path / "reports/final"
    reports.mkdir(parents=True)
    values = {
        "temporal_2026.json": {"rows_evaluated": 1000, "macro_f1": 0.47, "private": "excluded"},
        "data_quality.json": {"counts": {"kept": 2521}},
        "coverage.json": {"totals": {"percent_covered": 100}},
    }
    for filename, value in values.items():
        (reports / filename).write_text(json.dumps(value))
    (reports / "pytest.xml").write_text('<testsuites><testsuite tests="39"/></testsuites>')
    monkeypatch.setattr(api, "ROOT", tmp_path)
    with TestClient(api.app) as client:
        result = client.get("/project").json()
        assert result["tests"] == 39 and result["coverage"] == 100
        assert result["corpus"]["kept"] == 2521
        assert result["model_ready"] is False
        assert "private" not in result["temporal"]
