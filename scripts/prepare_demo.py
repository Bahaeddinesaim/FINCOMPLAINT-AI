"""Build a bounded REAL-data demonstration, never a final benchmark.

Run from repository root: python -m scripts.prepare_demo
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from importlib.metadata import version

import pandas as pd

from src.classifier import train
from src.ingestion import prepare
from src.retriever import build
from src.rag import ask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-file", type=int, default=3000)
    args = parser.parse_args()
    if args.rows_per_file < 100:
        parser.error("Au moins 100 lignes par fichier")
    reports = Path("reports/demo")
    reports.mkdir(parents=True, exist_ok=True)
    inputs = []
    # No 2026 record is opened in this demonstration: preserve the temporal holdout.
    with tempfile.TemporaryDirectory() as directory:
        for source in sorted(Path(".").glob("CCDB*2025*.csv")):
            frame = pd.read_csv(source, nrows=args.rows_per_file, dtype=str, keep_default_na=False)
            target = Path(directory) / source.name
            frame.to_csv(target, index=False)
            inputs.append({"file": source.name, "original_bytes": source.stat().st_size,
                           "rows_read": len(frame), "sample_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        quality = prepare(Path(directory), Path("data/processed/demo.sqlite"), reports / "data_quality.json")
    metrics = train(Path("data/processed/demo.sqlite"), Path("artifacts/demo"), reports, 30000, 10000)
    index = build(Path("data/processed/demo.sqlite"), Path("artifacts/demo/retriever.joblib"))
    manifest = {"scope": "Demonstration sur les premieres lignes de chaque export 2025, non representatif",
                "rows_per_file": args.rows_per_file, "inputs": inputs, "test_2026_opened": False,
                "python": platform.python_version(),
                "packages": {name: version(name) for name in ["pandas", "scikit-learn", "joblib"]},
                "indexed_documents": len(index["documents"])}
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Store references and short previews; full excerpts are generated locally by the CLI.
    examples = []
    for query in ["credit report identity theft", "mortgage payment escrow", "quasar nebula astrophysics"]:
        answer = ask(index, query)
        examples.append({"query": query, "mode": answer["mode"],
                         "answer_preview": answer["answer"][:240],
                         "sources": [{k: s[k] for k in ["complaint_id", "source_file", "score", "product"]}
                                     for s in answer["sources"]]})
    (reports / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = metrics["splits"]["validation"]
    summary = ("# Preuves de demonstration\n\n"
               f"Corpus reel echantillonne : {quality['counts']['kept']} recits dedupliques. "
               f"Train : {metrics['train_rows']}. Validation : {validation['rows']}.\n\n"
               "| Modele | Partition | Macro-F1 |\n| --- | --- | --- |\n"
               f"| TF-IDF + regression logistique | Validation 2025 echantillonnee | {validation['macro_f1']:.4f} |\n\n"
               "Ces scores ne sont pas les performances sur le corpus complet. Les premieres lignes "
               "des six exports 2025 constituent un echantillon de convenance. Test 2026 non ouvert.\n\n"
               "Recherche : TF-IDF lexical, index train uniquement. Voir examples.json pour les IDs et fichiers sources. "
               "Aucun Recall@k/MRR de pertinence humaine ni score de fidelite LLM n'est revendique. "
               "Le mode extractif ne genere aucune synthese. Le mode Ollama n'a pas ete execute.\n")
    (reports / "RESULTS.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
