"""TF-IDF + logistic regression baseline; temporal evaluation stays explicit."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


def train(database: Path, output: Path, report_dir: Path, train_limit: int, eval_limit: int, evaluate_test: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db:
        def sample(split, limit):
            # Stable hash order gives a reproducible sample independent of export order.
            return pd.read_sql_query(
                "SELECT narrative, product FROM complaints WHERE split=? ORDER BY text_hash LIMIT ?",
                db, params=(split, limit))
        training = sample("train", train_limit)
        if len(training) < 2 or training["product"].nunique() < 2:
            raise ValueError("Au moins deux classes sont necessaires dans le train.")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                                     max_features=60000, sublinear_tf=True, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ])
        print(f"Entrainement: {len(training)} textes, {training['product'].nunique()} classes", flush=True)
        model.fit(training["narrative"], training["product"])
        results = {"train_rows": len(training), "train_limit": train_limit,
                   "eval_limit": eval_limit, "test_evaluated": evaluate_test,
                   "sampling": "First N text hashes per split; not stratified.", "splits": {}}
        for split in (["validation", "test"] if evaluate_test else ["validation"]):
            frame = sample(split, eval_limit)
            if frame.empty:
                results["splits"][split] = {"rows": 0}
                continue
            predictions = model.predict(frame["narrative"])
            labels = sorted(set(model.classes_) | set(frame["product"]))
            results["splits"][split] = {
                "rows": len(frame),
                "macro_f1": f1_score(frame["product"], predictions, labels=labels, average="macro", zero_division=0),
                "report": classification_report(frame["product"], predictions, labels=labels, output_dict=True, zero_division=0),
                "unseen_labels": sorted(set(frame["product"]) - set(model.classes_)),
            }
            pd.DataFrame(confusion_matrix(frame["product"], predictions, labels=labels), index=labels, columns=labels).to_csv(
                report_dir / f"confusion_{split}.csv")
        joblib.dump(model, output / "baseline.joblib")
        (report_dir / "baseline_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/processed/complaints.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--eval-limit", type=int, default=10000)
    parser.add_argument("--evaluate-test", action="store_true", help="Ouvre explicitement le test final 2026")
    args = parser.parse_args()
    if args.train_limit < 2 or args.eval_limit < 1:
        parser.error("Limites invalides")
    results = train(args.database, args.output, args.report_dir, args.train_limit, args.eval_limit, args.evaluate_test)
    print(json.dumps({k: {"rows": v["rows"], "macro_f1": v.get("macro_f1")} for k, v in results["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
