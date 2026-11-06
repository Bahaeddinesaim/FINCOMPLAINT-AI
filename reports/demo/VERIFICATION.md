# Vérifications de la remise

Vérifications effectuées le 6 novembre 2026 :

| Vérification | Résultat observé |
| --- | --- |
| `python -m scripts.prepare_demo` | Exécution réussie ; 18 000 lignes lues, 2 521 récits retenus |
| `python -m unittest discover -s tests -v` | 7 tests réussis |
| Notebooks 01 à 04 | Tous exécutés ; 17 cellules de code avec compteurs d'exécution, aucune sortie d'erreur |
| Validation des formats avec `nbformat.validate` | 4 notebooks valides |
| CLI `classify`, texte « I dispute a charge on my credit card » | Produit prédit : `Credit card` |
| CLI `ask`, texte « quasar nebula astrophysics » | Mode `abstention`, aucune source |
| Traçabilité extractive | Vérification des extraits, ID, fichiers sources et appartenance au train dans le notebook 04 |

Le test du protocole Ollama utilise une réponse simulée et vérifie la détection d'un ID inventé. Aucun modèle génératif réel n'a été exécuté. Les tests fonctionnels ne mesurent pas la pertinence humaine ni la fidélité d'une synthèse.

Une première exécution automatique des notebooks a été interrompue par un problème de noyau Jupyter sous Windows. Les notebooks restants ont ensuite été exécutés avec la politique de boucle événementielle Windows adaptée et un répertoire IPython temporaire. `scripts/execute_notebooks.py` reprend cette configuration pour la reproduction.
