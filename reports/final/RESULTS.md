# Preuves de demonstration

Corpus reel echantillonne : 2521 recits dedupliques. Train : 2024. Validation : 497.

| Modele | Partition | Macro-F1 |
| --- | --- | --- |
| TF-IDF + regression logistique | Validation 2025 echantillonnee | 0.4966 |

Ces scores ne sont pas les performances sur le corpus complet. Les premieres lignes des six exports 2025 constituent un echantillon de convenance. Test 2026 non ouvert.

Recherche : TF-IDF lexical, index train uniquement. Voir examples.json pour les IDs et fichiers sources. Aucun Recall@k/MRR de pertinence humaine ni score de fidelite LLM n'est revendique. Le mode extractif ne genere aucune synthese. Le mode Ollama n'a pas ete execute.
