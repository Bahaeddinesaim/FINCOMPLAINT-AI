# Verification du projet

Depuis la racine du projet, avec son environnement Python :

```powershell
.venv\Scripts\python.exe -m scripts.verify_project
```

Pour enregistrer aussi la nouvelle sortie dans la cellule de verification du notebook :

```powershell
.venv\Scripts\python.exe -m scripts.verify_project --update-notebook
```

La suite contient 39 tests et couvre toutes les lignes et branches des modules `src`.
Ruff, son formatage et un seuil de couverture de 100 % sont controles par la commande.
Rapports : `reports/final/pytest.xml`, `coverage.json` et `coverage/index.html`.

Les tests rapides verifient les contrats, le nettoyage, les erreurs, les commandes,
les citations et l'isolation temporelle. Les integrations executent MiniLM et FAISS,
FLAN-T5 et l'adaptation DistilBERT avec les vrais checkpoints locaux. Le test
d'entrainement ecrit ses poids dans un dossier temporaire, sans toucher au modele livre.
Il verifie le gel de l'encodeur, la modification de la tete et le rechargement des poids.

Les modeles doivent etre presents dans `.models` et les artefacts generes dans
`artifacts/final`. Leur absence provoque un echec explicite, pas un test ignore.
Les tests passent les bibliotheques Hugging Face en mode hors ligne. Les sockets locaux
restent necessaires au client de test asynchrone FastAPI. Le plugin Dash externe est
desactive dans pytest pour eviter son avertissement de deprecation sans rapport au projet.

La couverture a 100 % ne signifie pas une absence de bugs ni une fidelite RAG garantie.
Elle porte sur le code de `src`, pas sur chaque script auxiliaire ou cellule notebook.
La qualite des predictions et des reponses reste une question d'evaluation NLP distincte.
