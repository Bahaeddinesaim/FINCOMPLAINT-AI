"""Extend the repository notebook with executed pretrained-model experiments."""

from pathlib import Path

import nbformat as nbf


def main():
    from scripts.create_project_notebook import main as create_base

    create_base()
    root = Path(__file__).resolve().parents[1]
    notebook = nbf.read(root / "notebooks/00_projet_fincomplaint.ipynb", as_version=4)
    # Preserve the repository experiment; place advanced results before quality checks.
    cells = [
        nbf.v4.new_markdown_cell(
            "# Experience complete : spaCy, DistilBERT, FAISS et RAG generatif\n\n"
            "Les donnees sont reelles. Les limites CPU sont explicites et identiques pour la "
            "comparaison de classification. Adaptation supervisee de la tete DistilBERT : "
            "encodeur pretrained gele, pas de fine-tuning integral. Le test 2026 reste ferme. "
            "La couverture fonctionnelle ne garantit pas la qualite scientifique d'une synthese."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "os.environ['HF_HOME'] = str(ROOT / '.models')\n"
            "os.environ['HF_HUB_DISABLE_XET'] = '1'\n"
            "import torch\n"
            "torch.set_num_threads(4)\n"
            "import spacy\n"
            "from src.advanced import SemanticIndex, adapt_distilbert, answer_question, ranking_metrics\n"
            "nlp = spacy.load('en_core_web_sm')\n"
            "doc = nlp(train.narrative.iloc[0][:2000])\n"
            "display(pd.DataFrame([(t.text, t.lemma_, t.pos_) for t in doc][:30],\n"
            "                     columns=['token', 'lemme', 'categorie']))\n"
            "print('Entites:', [(e.text, e.label_) for e in doc.ents])\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparaison supervisee sur les memes effectifs\n"
            "Maximum 24 textes par classe en apprentissage et 8 par classe en validation, "
            "choisis par l'ordre deterministe des hash deja etabli. Aucun parametre ne sera "
            "choisi sur 2026. Les textes du Transformer sont tronques a 128 tokens ; TF-IDF "
            "utilise les textes complets. Cette difference est une limite de comparaison."
        ),
        nbf.v4.new_code_cell(
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "small_train = train.groupby('product', group_keys=False).head(24).copy()\n"
            "small_validation = validation.groupby('product', group_keys=False).head(8).copy()\n"
            "print('Train / validation:', len(small_train), len(small_validation))\n"
            "baseline_small = make_pipeline(TfidfVectorizer(ngram_range=(1,2)),\n"
            "    LogisticRegression(max_iter=500, class_weight='balanced', random_state=42))\n"
            "baseline_small.fit(small_train.narrative, small_train['product'])\n"
            "advanced_predictions, history = adapt_distilbert(small_train, small_validation, epochs=30)\n"
            "common_labels = sorted(set(small_train['product']) | set(small_validation['product']))\n"
            "model_comparison = pd.DataFrame([\n"
            "    {'modele': name, 'macro_f1': f1_score(small_validation['product'], pred,\n"
            "        labels=common_labels, average='macro', zero_division=0)}\n"
            "    for name, pred in [('TF-IDF memes donnees', baseline_small.predict(small_validation.narrative)),\n"
            "                       ('DistilBERT tete adaptee', advanced_predictions)]])\n"
            "display(model_comparison)\ndisplay(history.iloc[[0,-1]])\n"
            "display(pd.DataFrame(classification_report(small_validation['product'],\n"
            "    advanced_predictions, labels=common_labels, output_dict=True, zero_division=0)).T)\n"
            "model_comparison.to_csv(REPORTS / 'advanced_classification.csv', index=False)\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Recherche : decoupage, MiniLM et FAISS\n"
            "Index limite aux 300 premiers documents train par hash, segments de 80 mots "
            "avec recouvrement de 15 mots. MiniLM produit des embeddings normalises ; FAISS "
            "IndexFlatIP calcule le cosinus. L'index lexical est compare sur les memes documents. "
            "Les requetes sont des exemples de demonstration, pas un benchmark humain independant."
        ),
        nbf.v4.new_code_cell(
            "from src.retriever import build as build_lexical, search as search_lexical\n"
            "with closing(sqlite3.connect(DATABASE)) as db:\n"
            '    records = pd.read_sql_query("SELECT complaint_id, narrative, product, source_file, "\n'
            "        \"date_received FROM complaints WHERE split='train' ORDER BY text_hash LIMIT 300\", db)\n"
            "semantic = SemanticIndex(records.to_dict('records'))\n"
            "lexical = build_lexical(DATABASE, ROOT / 'artifacts/final/lexical300.joblib', limit=300)\n"
            "print('Documents:', len(records), 'segments:', len(semantic.passages))\n"
            "questions = ['credit report identity theft', 'mortgage payment escrow',\n"
            "             'credit card unauthorized charge', 'quasar nebula astrophysics']\n"
            "retrieval_comparison = []\n"
            "for question in questions:\n"
            "    for k in [3,5]:\n"
            "        for name, hits in [('TF-IDF', search_lexical(lexical, question, k)),\n"
            "                           ('MiniLM-FAISS', semantic.search(question, k))]:\n"
            "            for hit in hits:\n"
            "                retrieval_comparison.append({'question':question, 'k':k, 'engine':name,\n"
            "                    'id':hit['complaint_id'], 'score':hit['score']})\n"
            "display(pd.DataFrame(retrieval_comparison))\n"
            "import faiss\n"
            "faiss.write_index(semantic.index, str(ROOT / 'artifacts/final/semantic.faiss'))\n"
            "pd.DataFrame(semantic.passages).to_json(ROOT / 'artifacts/final/passages.json', orient='records')\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Evaluation de recherche : controle automatique et annotation humaine\n"
            "Le controle ci-dessous utilise 12 questions construites a partir d'un extrait "
            "de document indexe (known-item retrieval). Il mesure la recuperation de l'element "
            "connu, sans estimer la pertinence des questions utilisateurs. Les annotations "
            "humaines independantes restent necessaires pour un benchmark metier."
        ),
        nbf.v4.new_code_cell(
            "retrieval_scores = []\n"
            "for record in records.head(12).to_dict('records'):\n"
            "    query = ' '.join(record['narrative'].split()[:20])\n"
            "    for k in [3,5]:\n"
            "        for name, hits in [('TF-IDF', search_lexical(lexical, query, k)),\n"
            "                           ('MiniLM-FAISS', semantic.search(query, k))]:\n"
            "            ids = list(dict.fromkeys(str(h['complaint_id']) for h in hits))\n"
            "            retrieval_scores.append({'engine':name, 'k':k,\n"
            "                **ranking_metrics(ids,{str(record['complaint_id'])})})\n"
            "retrieval_summary = pd.DataFrame(retrieval_scores).groupby(['engine','k']).mean()\n"
            "display(retrieval_summary)\n"
            "retrieval_summary.to_csv(REPORTS / 'known_item_retrieval.csv')\n"
        ),
        nbf.v4.new_markdown_cell(
            "## RAG generatif local avec preuves consultables\n"
            "FLAN-T5-small genere une reponse conditionnee sur les extraits retrouves. "
            "Comparaison top-3/top-5. Seuil cosinus fixe a priori a 0,35 pour l'abstention ; "
            "ce seuil exploratoire n'est pas calibre. Les IDs cites sont controles, mais "
            "cela ne garantit pas la fidelite. Les citations absentes ou invalides sont "
            "signalees : les sources recuperees sont toujours affichees separement."
        ),
        nbf.v4.new_code_cell(
            "rag_results = []\n"
            "for question in ['What problems are reported about mortgage escrow payments?',\n"
            "                 'Who won the football world cup?', 'How do I bake a chocolate cake?']:\n"
            "    for k in [3,5]:\n"
            "        result = answer_question(semantic, question, k=k)\n"
            "        print('QUESTION:', question, 'k=', k)\n"
            "        print('MODE:', result['mode'], 'REPONSE:', result['answer'])\n"
            "        print('CITATIONS VALIDES:', result['citations_valid'])\n"
            "        display(pd.DataFrame(result['sources']))\n"
            "        rag_results.append({'question':question, 'k':k, **result})\n"
            "(REPORTS / 'generated_examples.json').write_text(\n"
            "    json.dumps(rag_results, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "display(pd.DataFrame([{key:value for key,value in r.items() if key!='sources'}\n"
            "                      for r in rag_results]))\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Analyse critique des modeles avances\n"
            "Un encodeur gele est une adaptation legere economique mais moins expressive "
            "qu'un fine-tuning complet. Les classes rares, textes tronques et biais de "
            "l'echantillon limitent les resultats. Le petit generateur peut halluciner, "
            "repondre a cote ou omettre ses citations ; ces erreurs font partie des resultats, "
            "pas d'une panne a masquer. Une evaluation humaine de chaque affirmation est "
            "necessaire avant tout usage. Aucun resultat final 2026 n'est annonce."
        ),
    ]
    notebook.cells[-4:-4] = cells
    notebook.cells[0].source = (
        "# FinComplaint AI - pipeline NLP complet sur donnees reelles\n\n"
        "SAIM Bahaeddine et OUNI Hazem. Preparation, classification baseline et Transformer, "
        "recherche lexicale et semantique, RAG generatif, evaluation et tests. "
        "Les effectifs et limites sont explicites. Les donnees 2026 restent reservees. "
        "Les resultats d'evaluation ne sont pas une promesse de performance en production."
    )
    notebook.cells.extend(
        [
            nbf.v4.new_markdown_cell(
                "# Test temporel final 2026 - echantillon borne\n\n"
                "Les choix precedents sont maintenant fixes. La baseline conservee est evaluee "
                "sans reentrainement sur les 3 000 premieres lignes de chaque export 2026, "
                "apres filtrage et exclusion des doublons connus avec 2025. Au maximum 1 000 "
                "recits sont retenus par ordre de hash. Cette section ouvre explicitement le "
                "test temporel, contrairement aux sections de demonstration historique ci-dessus. "
                "Le resultat ne mesure pas l'ensemble de 2026."
            ),
            nbf.v4.new_code_cell(
                "from scripts.evaluate_temporal import main as evaluate_temporal\n"
                "evaluate_temporal()\n"
                "temporal = json.loads((REPORTS / 'temporal_2026.json').read_text(encoding='utf-8'))\n"
                "display(pd.DataFrame(temporal['classification_report']).T)\n"
                "assert not temporal['refitted']\n"
            ),
            nbf.v4.new_markdown_cell(
                "# Demonstrateur REST\n\n"
                "Lancer `python -m uvicorn src.api:app --host 127.0.0.1 --port 8011`. "
                "Ouvrir http://127.0.0.1:8011/docs. `/classify` utilise la baseline retenue ; "
                "`/search` interroge FAISS ; `/ask` genere avec FLAN-T5 et refuse les brouillons "
                "sans citations valides. Les tests de contrats HTTP utilisent une injection "
                "de dependances ; le notebook execute separement les vrais modeles."
            ),
        ]
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(notebook, root / "notebooks/00_projet_fincomplaint.ipynb")


if __name__ == "__main__":
    main()
