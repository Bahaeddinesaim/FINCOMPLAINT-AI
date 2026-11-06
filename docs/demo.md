# Parcours de demonstration

1. Lancer `python -m uvicorn src.api:app --host 127.0.0.1 --port 8012`.
2. Ouvrir http://127.0.0.1:8012 et expliquer les indicateurs mesures.
3. Classifier l'exemple de frais de carte, sans presenter de confiance calibree.
4. Rechercher `mortgage escrow payment`, consulter extraits et identifiants.
5. Demander `What problems are reported about mortgage escrow payments?` en mode RAG.
   Comparer k=3 et k=5 et distinguer reponse, sources et brouillon retenu.
6. Tester `Who won the football world cup?` et expliquer l'abstention observee.
7. Montrer les comparaisons du notebook et l'evaluation temporelle 2026.
8. Lancer `python -m scripts.verify_project` et distinguer qualite logicielle et NLP.

Limites a exposer : echantillons de convenance, encodeur gele, index limite,
generation imparfaite et besoin d'annotations humaines independantes.
