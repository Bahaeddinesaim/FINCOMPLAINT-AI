"""Evaluate the fixed baseline on a bounded 2026 sample, without refitting."""

import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.preprocessing import narrative_hash


def main():
    root = Path(__file__).resolve().parents[1]
    frames = [
        pd.read_csv(path, nrows=3000, dtype=str, keep_default_na=False) for path in sorted(root.glob("CCDB*2026*.csv"))
    ]
    frame = pd.concat(frames, ignore_index=True)
    rows_read = len(frame)
    dates = pd.to_datetime(frame["Date received"], format="mixed", errors="coerce")
    frame = frame[
        (dates.dt.year == 2026)
        & frame["Consumer complaint narrative"].str.strip().ne("")
        & frame["Product"].str.strip().ne("")
        & frame["Complaint ID"].str.strip().ne("")
    ].copy()
    frame["text_hash"] = frame["Consumer complaint narrative"].map(narrative_hash)
    frame = frame.drop_duplicates("Complaint ID").drop_duplicates("text_hash")
    with sqlite3.connect(root / "data/processed/demo.sqlite") as connection:
        prior = pd.read_sql_query("SELECT complaint_id, text_hash FROM complaints", connection)
    frame = frame[~frame["Complaint ID"].isin(prior.complaint_id) & ~frame.text_hash.isin(prior.text_hash)]
    frame = frame.sort_values("text_hash").head(1000)
    if frame.empty:
        raise ValueError("No independent 2026 records")
    model = joblib.load(root / "artifacts/final/baseline.joblib")
    predicted = model.predict(frame["Consumer complaint narrative"])
    labels = sorted(set(model.classes_) | set(frame["Product"]))
    report = {
        "model": "fixed baseline trained on 2025 demo",
        "scope": "first 3000 rows per 2026 export; at most 1000 retained hash-ordered narratives",
        "rows_read": rows_read,
        "rows_evaluated": len(frame),
        "refitted": False,
        "macro_f1": f1_score(frame["Product"], predicted, labels=labels, average="macro", zero_division=0),
        "unseen_labels": sorted(set(frame["Product"]) - set(model.classes_)),
        "classification_report": classification_report(
            frame["Product"], predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "limitations": "Convenience sample, no near-duplicate audit; not a full-corpus benchmark.",
    }
    (root / "reports/final/temporal_2026.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
