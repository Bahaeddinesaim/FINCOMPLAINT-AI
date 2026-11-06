"""REST demonstrator for the trained pipeline. Launch with uvicorn src.api:app."""

import json
from functools import lru_cache
from xml.etree import ElementTree

import joblib
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.advanced import ROOT, SemanticIndex, answer_question

app = FastAPI(title="FinComplaint AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/", include_in_schema=False)
def workspace():
    return FileResponse(ROOT / "web/index.html")


@app.get("/project")
def project():
    reports = ROOT / "reports/final"
    temporal = json.loads((reports / "temporal_2026.json").read_text(encoding="utf-8"))
    quality = json.loads((reports / "data_quality.json").read_text(encoding="utf-8"))
    coverage = json.loads((reports / "coverage.json").read_text(encoding="utf-8"))
    suite = ElementTree.parse(reports / "pytest.xml").getroot().find("testsuite")
    return {
        "temporal": {key: temporal[key] for key in ["rows_evaluated", "macro_f1"]},
        "corpus": quality["counts"],
        "tests": int(suite.attrib["tests"]),
        "coverage": coverage["totals"]["percent_covered"],
        "model_ready": (ROOT / "artifacts/final/baseline.joblib").exists(),
    }


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=10000, pattern=r"\S")
    k: int = Field(default=3, ge=1, le=5)


@lru_cache(maxsize=1)
def service():
    import faiss
    from sentence_transformers import SentenceTransformer

    semantic = SemanticIndex.__new__(SemanticIndex)
    semantic.passages = json.loads((ROOT / "artifacts/final/passages.json").read_text())
    semantic.model = SentenceTransformer(str(ROOT / ".models/minilm"))
    semantic.index = faiss.read_index(str(ROOT / "artifacts/final/semantic.faiss"))
    baseline = joblib.load(ROOT / "artifacts/final/baseline.joblib")
    return baseline, semantic


@app.post("/classify")
def classify(query: Query, models=Depends(service)):
    return {"product": models[0].predict([query.text])[0], "model": "TF-IDF logistic regression"}


@app.post("/search")
def search(query: Query, models=Depends(service)):
    return {"sources": models[1].search(query.text, query.k)}


@app.post("/ask")
def ask(query: Query, models=Depends(service)):
    return answer_question(models[1], query.text, query.k)
