"""Generate the four commented notebooks; execute them separately with Jupyter."""

from pathlib import Path
import nbformat as nbf


SETUP = """from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'src').is_dir(), 'Ouvrir depuis la racine ou notebooks/'
sys.path.insert(0, str(ROOT))
import json
import sqlite3
from contextlib import closing
import pandas as pd
REPORTS = ROOT / 'reports/demo'
DATABASE = ROOT / 'data/processed/demo.sqlite'
assert DATABASE.is_file(), 'Executer python -m scripts.prepare_demo depuis la racine'
"""


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(cells[0]),
                                         nbf.v4.new_code_cell(SETUP)] + [
        nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text)
        for kind, text in cells[1:]])
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    target = Path('notebooks') / name
    target.parent.mkdir(exist_ok=True)
    nbf.write(notebook, target)


def main():
    save('01_preparation.ipynb', [
        '# 01 — Préparation et exploration\n\nSAIM Bahaeddine et OUNI Hazem. '
        'Corpus : archives CFPB 2025. Ce notebook explore une démonstration réelle limitée '
        'aux 3 000 premières lignes de chaque export 2025. Aucun résultat sur le corpus complet '
        'ou sur 2026 n’est déduit de cet échantillon. Préparation reproductible : '
        '`python -m scripts.prepare_demo`. Pour le corpus complet : `python run_pipeline.py`.',
        ('md', '## Origine et sélection\nLes récits sont conservés avec leur ID et fichier source. '
         'Les dates invalides, textes vides et doublons normalisés sont exclus. La partition 2025 '
         'est déterminée par hash (environ 80 % train, 20 % validation). Voir le README pour les conditions de réutilisation.'),
        ('code', "quality = json.loads((REPORTS / 'data_quality.json').read_text(encoding='utf-8'))\n"
         "pd.DataFrame([quality['counts']]).T.rename(columns={0: 'effectif'})"),
        ('code', "with closing(sqlite3.connect(DATABASE)) as db:\n"
         "    distribution = pd.read_sql_query('SELECT product, split, COUNT(*) AS n FROM complaints GROUP BY product, split', db)\n"
         "distribution.pivot(index='product', columns='split', values='n').fillna(0).astype(int)"),
        ('code', "pd.DataFrame(quality['lengths'])"),
        ('md', 'Les textes sont déséquilibrés et les premières lignes des fichiers ne constituent '
         'pas un échantillon représentatif. Les quasi-doublons ne sont pas détectés. '
         'La longueur et les effectifs par classe doivent guider les expériences suivantes.')])
    save('02_modeles.ipynb', [
        '# 02 — Classification et comparaison\n\nBaseline entraînée sur les textes train uniquement : '
        'TF-IDF uni/bigrammes + régression logistique pondérée. Comparaison avec une prédiction '
        'de classe majoritaire, sans ajuster de paramètres sur la validation. DistilBERT reste une extension.',
        ('code', "import joblib\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.metrics import f1_score, classification_report\n"
         "model = joblib.load(ROOT / 'artifacts/demo/baseline.joblib')\n"
         "with closing(sqlite3.connect(DATABASE)) as db:\n"
         "    train = pd.read_sql_query(\"SELECT narrative, product FROM complaints WHERE split='train' ORDER BY text_hash LIMIT 30000\", db)\n"
         "    validation = pd.read_sql_query(\"SELECT narrative, product FROM complaints WHERE split='validation' ORDER BY text_hash LIMIT 10000\", db)\n"
         "labels = sorted(set(train['product']) | set(validation['product']))\n"
         "dummy = DummyClassifier(strategy='most_frequent').fit(train[['narrative']], train['product'])\n"
         "predictions = model.predict(validation['narrative'])\n"
         "comparison = pd.DataFrame([\n"
         "    {'modele': 'Classe majoritaire', 'macro_f1': f1_score(validation['product'], dummy.predict(validation[['narrative']]), labels=labels, average='macro', zero_division=0)},\n"
         "    {'modele': 'TF-IDF + regression logistique', 'macro_f1': f1_score(validation['product'], predictions, labels=labels, average='macro', zero_division=0)}\n"
         "])\ncomparison"),
        ('code', "pd.DataFrame(classification_report(validation['product'], predictions, labels=labels, output_dict=True, zero_division=0)).T"),
        ('md', '## Analyse des erreurs\nLes cinq premières erreurs dans l’ordre stable des hash sont '
         'affichées sans publier les textes. Une classe sans effectif de validation ne permet '
         'pas d’estimer sa performance. La macro-F1 donne le même poids à chaque classe.'),
        ('code', "errors = validation.assign(prediction=predictions)\n"
         "errors.loc[errors['product'] != errors['prediction'], ['product', 'prediction']].head()"),
        ('md', 'Le modèle Joblib doit provenir d’une source de confiance. Le test temporel 2026 '
         'reste fermé pendant ces expériences. La reproductibilité dépend des versions consignées dans manifest.json.')])
    save('03_rag.ipynb', [
        '# 03 — Recherche et RAG\n\nRecherche lexicale TF-IDF avec similarité cosinus sur le train 2025. '
        'Le mode par défaut retourne des extraits avec leurs sources ; il ne réalise pas de génération. '
        'Une synthèse par modèle local Ollama peut être activée explicitement. '
        'Aucun embedding ou modèle externe n’est téléchargé par ce notebook.',
        ('code', "import joblib\nfrom src.retriever import search\nfrom src.rag import ask\n"
         "index = joblib.load(ROOT / 'artifacts/demo/retriever.joblib')\n"
         "question = 'credit report identity theft'\n"
         "# Comparer k=3 et k=5 : ces scores sont des similarites, pas des probabilites.\n"
         "experiments = []\nfor k in [3, 5]:\n"
         "    for source in search(index, question, k):\n"
         "        experiments.append({'k': k, **{field: source[field] for field in ['complaint_id', 'source_file', 'score']}})\n"
         "pd.DataFrame(experiments)"),
        ('code', "answer = ask(index, question)\n"
         "# Apercu bref; les extraits complets sont consultables avec la CLI.\n"
         "print(answer['mode'])\nprint(answer['answer'][:240])\n"
         "print('Sources :', [source['complaint_id'] for source in answer['sources']])"),
        ('code', "ask(index, 'quasar nebula astrophysics')"),
        ('md', '## Génération locale facultative\nAprès installation d’Ollama et d’un modèle compatible, '
         'définir `OLLAMA_MODEL` dans le terminal puis appeler '
         '`python -m src.demo ask "credit report identity theft" --artifacts artifacts/demo --generate`. '
         'La requête envoyée au modèle contient la question et les extraits sourcés. '
         'Le contrôle des IDs cités ne garantit pas la fidélité des affirmations. '
         'La génération Ollama n’est pas exécutée ni évaluée ici. '
         '[Documentation du protocole](https://docs.ollama.com/api/chat).'),
        ('md', 'L’abstention est déclenchée uniquement en l’absence de recouvrement lexical. '
         'Une question hors domaine partageant des mots du corpus peut donc produire des résultats. '
         'Comparer k=3 et k=5 ne mesure pas leur pertinence : il faut des jugements humains indépendants.')])
    save('04_evaluation.ipynb', [
        '# 04 — Évaluation et preuves de fonctionnement\n\nCette évaluation sépare les scores de '
        'classification sur validation et les contrôles fonctionnels de traçabilité. '
        'Aucun score de pertinence ou de fidélité humaine n’est inventé.',
        ('code', "metrics = json.loads((REPORTS / 'baseline_metrics.json').read_text(encoding='utf-8'))\n"
         "pd.DataFrame([{'partition': split, 'effectif': score['rows'], 'macro_f1': score.get('macro_f1')}\n"
         "              for split, score in metrics['splits'].items()])"),
        ('code', "pd.read_csv(REPORTS / 'confusion_validation.csv', index_col=0)"),
        ('md', '## Contrôles fonctionnels\nVérifier que chaque extrait provient du récit stocké et '
         'que ses métadonnées correspondent au corpus. Ces assertions évaluent la traçabilité du mode extractif, '
         'sans évaluer la compréhension ou la qualité d’une synthèse.'),
        ('code', "import joblib\nfrom src.rag import ask\n"
         "index = joblib.load(ROOT / 'artifacts/demo/retriever.joblib')\nchecks = []\n"
         "with closing(sqlite3.connect(DATABASE)) as db:\n"
         "    for question in ['credit report identity theft', 'mortgage payment escrow']:\n"
         "        result = ask(index, question)\n        assert result['sources']\n"
         "        for source in result['sources']:\n"
         "            row = db.execute('SELECT narrative, source_file, split FROM complaints WHERE complaint_id=?', (source['complaint_id'],)).fetchone()\n"
         "            assert row and row[0].startswith(source['excerpt'])\n"
         "            assert row[1] == source['source_file'] and row[2] == 'train'\n"
         "        checks.append({'question': question, 'sources_verifiees': len(result['sources'])})\n"
         "assert ask(index, 'quasar nebula astrophysics')['mode'] == 'abstention'\n"
         "assert not metrics['test_evaluated']\npd.DataFrame(checks)"),
        ('code', "manifest = json.loads((REPORTS / 'manifest.json').read_text(encoding='utf-8'))\n"
         "{'scope': manifest['scope'], 'python': manifest['python'], 'packages': manifest['packages'], 'test_2026_opened': manifest['test_2026_opened']}"),
        ('md', '## Évaluation finale à réaliser\nConstituer un jeu de questions indépendant avec sources '
         'pertinentes annotées avant la sélection du moteur. Rapporter Recall@3, Recall@5 et MRR, '
         'puis vérifier manuellement chaque affirmation et citation des synthèses générées. '
         'Ouvrir le test 2026 seulement après fixation des choix : '
         '`python -m src.classifier --evaluate-test`. Les résultats de démonstration ne remplacent pas cette évaluation finale.')])


if __name__ == '__main__':
    main()
