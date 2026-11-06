"""CLI demonstrator: classify, search, ask. No server or API key required."""

import argparse
import json
import sys
from pathlib import Path

import joblib

from src.rag import ask
from src.retriever import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["classify", "search", "ask"])
    parser.add_argument("text")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--generate", action="store_true", help="Appel explicite au modele local Ollama")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Texte vide")
    try:
        if args.action == "classify":
            model = joblib.load(args.artifacts / "baseline.joblib")
            result = {"text": args.text, "product": str(model.predict([args.text])[0])}
        else:
            index = joblib.load(args.artifacts / "retriever.joblib")
            result = (
                search(index, args.text, args.top_k)
                if args.action == "search"
                else ask(index, args.text, args.top_k, args.generate)
            )
    except (OSError, ValueError) as error:
        print(f"Demonstrateur: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
