"""Assemble the repository notebooks into one reproducible project notebook."""

from pathlib import Path

import nbformat as nbf


def main():
    root = Path(__file__).resolve().parents[1]
    cells = [
        nbf.v4.new_markdown_cell(
            "# FinComplaint AI - projet sur donnees reelles\n\n"
            "SAIM Bahaeddine et OUNI Hazem. Ce notebook suit l'architecture du README : "
            "CSV -> SQLite -> classification TF-IDF -> recherche lexicale -> reponses sourcees.\n\n"
            "Les CSV sont recuperes du depot Git LFS. La demonstration utilise les 3 000 premieres "
            "lignes des six exports 2025 ; les donnees 2026 restent reservees au test final. "
            "Aucune donnee synthetique n'est substituee. Les extensions Transformer, FAISS et "
            "generation ne sont pas implementees dans cette architecture du depot."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nimport os\nimport subprocess\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "os.chdir(ROOT)\n"
            "ROWS_PER_FILE = 3000\n"
            "REBUILD_DEMO = True\n"
            "# False conserve les artefacts existants. True reproduit les calculs.\n"
            "if REBUILD_DEMO:\n"
            "    subprocess.run([sys.executable, '-m', 'scripts.prepare_demo',\n"
            "                    '--rows-per-file', str(ROWS_PER_FILE)], check=True, cwd=ROOT)\n"
        ),
    ]
    for filename in ["01_preparation.ipynb", "02_modeles.ipynb", "03_rag.ipynb", "04_evaluation.ipynb"]:
        original = nbf.read(root / "notebooks" / filename, as_version=4)
        for cell in original.cells:
            cell = nbf.from_dict(dict(cell))
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
            cells.append(cell)
    cells.extend(
        [
            nbf.v4.new_markdown_cell(
                "# Qualite du code et tests\n\n"
                "Le code metier reste dans src ; le notebook orchestre et analyse. "
                "pytest execute les tests unitaires existants ainsi qu'une integration "
                "CSV -> SQLite -> modele -> sources. La couverture n'est pas une mesure de qualite NLP."
            ),
            nbf.v4.new_code_cell(
                "commands = [\n"
                "    [sys.executable, '-m', 'ruff', 'check', '.'],\n"
                "    [sys.executable, '-m', 'ruff', 'format', '--check', '.'],\n"
                "    [sys.executable, '-m', 'pytest', '-q', '--cov=src', '--cov-report=term-missing'],\n"
                "]\n"
                "for command in commands:\n"
                "    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)\n"
                "    print(result.stdout, result.stderr)\n"
                "    assert result.returncode == 0, command\n"
            ),
            nbf.v4.new_markdown_cell(
                "# Limites et reproductibilite\n\n"
                "La validation sur cet echantillon de convenance ne mesure pas le corpus complet. "
                "Les valeurs affichees sont recalculees ; elles peuvent differer des rapports du depot. "
                "Le test 2026, le benchmark de recherche annote et l'evaluation humaine du RAG "
                "ne sont pas revendiques. Pour traiter tous les exports : `python run_pipeline.py`.\n\n"
                "Usage IA : notebook d'orchestration et tests complementaires prepares avec Codex ; "
                "les auteurs doivent verifier et interpreter les resultats avant remise. Aucun push effectue."
            ),
        ]
    )
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "name": "python3", "language": "python"}
    target = root / "notebooks" / "00_projet_fincomplaint.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
