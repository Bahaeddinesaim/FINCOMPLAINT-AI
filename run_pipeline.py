"""Prepare the complete corpus, then fit and validate the CPU baseline."""

import argparse
from pathlib import Path

from src.ingestion import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--eval-limit", type=int, default=10000)
    parser.add_argument("--index-limit", type=int, default=10000)
    args = parser.parse_args()
    if args.train_limit < 2 or args.eval_limit < 1 or args.index_limit < 1:
        parser.error("Limites invalides")
    database = Path("data/processed/complaints.sqlite")
    if not args.skip_ingestion:
        prepare(Path("."), database, Path("reports/data_quality.json"))
    if not database.is_file():
        parser.error("Corpus absent : relancer sans --skip-ingestion")
    from src.classifier import train
    results = train(database, Path("artifacts"), Path("reports"), args.train_limit, args.eval_limit)
    from src.retriever import build
    build(database, Path("artifacts/retriever.joblib"), args.index_limit)
    validation = results["splits"]["validation"]
    print(f"Validation: {validation['rows']} textes, macro-F1={validation.get('macro_f1')}")


if __name__ == "__main__":
    main()
