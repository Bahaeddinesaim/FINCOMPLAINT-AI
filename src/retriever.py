"""Reproducible lexical retrieval over training complaints only."""

import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build(database: Path, output: Path, limit: int = 10000):
    if limit < 1:
        raise ValueError("limit doit etre positif")
    with closing(sqlite3.connect(database)) as db:
        db.row_factory = sqlite3.Row
        documents = [
            dict(row)
            for row in db.execute(
                "SELECT complaint_id, date_received, product, issue, narrative, source_file "
                "FROM complaints WHERE split='train' ORDER BY text_hash LIMIT ?",
                (limit,),
            )
        ]
    if not documents:
        raise ValueError("Corpus train vide")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=60000)
    matrix = vectorizer.fit_transform([doc["narrative"] for doc in documents])
    index = {
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "method": "TF-IDF cosine",
        "split": "train",
        "limit": limit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, output)
    return index


def search(index, query: str, top_k: int = 3):
    if not query.strip() or not 1 <= top_k <= 20:
        raise ValueError("Question vide ou top_k hors de [1, 20]")
    scores = (index["matrix"] @ index["vectorizer"].transform([query]).T).toarray().ravel()
    # Stable sorting preserves hash order for equal scores.
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [
        {key: value for key, value in index["documents"][i].items() if key != "narrative"}
        | {"score": float(scores[i]), "excerpt": index["documents"][i]["narrative"][:1200]}
        for i in ranked
        if scores[i] > 0
    ]
