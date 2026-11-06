# Resultats de la version integree

Le notebook de 48 cellules a ete execute integralement sans cellule en erreur.
Les treize exports sont verifies par SHA-256 ; leur total est de 4 559 027 495 octets.

| Experience | Resultat |
| --- | --- |
| Baseline, validation 2025 de demonstration | macro-F1 0,4966 sur 497 textes |
| Baseline, petit jeu commun de comparaison | macro-F1 0,5609 |
| DistilBERT, tete adaptee, meme petit jeu | macro-F1 0,4025 |
| Baseline fixe, echantillon temporel 2026 | macro-F1 0,4722 sur 1 000 textes |
| Tests pytest | 39 reussis, avec integrations sur les modeles locaux et routes web |
| Couverture pytest des modules src | 100 % des lignes et des 82 branches |
| Ruff lint et format | passes |
| API /docs, /classify, /ask hors domaine | HTTP 200 verifies avec le service reel |

Les scores des deux premieres experiences n'utilisent pas les memes effectifs : ne pas
comparer leurs valeurs pour conclure sur un effet du modele. Le comparatif avance utilise
les memes petits jeux pour TF-IDF et DistilBERT. L'encodeur DistilBERT est gele : seule la
tete est adaptee, sans choisir les parametres sur le test 2026.

## Generation et erreurs observees

Le petit generateur a produit un brouillon court sans citation pour la question mortgage
avec top-3. Le controle de references le retient et affiche une abstention explicite ; le
brouillon reste consultable dans le JSON pour analyse. Avec top-5, une reponse contenant
un ID du contexte a ete produite. Verifier un ID ne prouve pas la fidelite de toutes les
affirmations. Les deux questions hors domaine declenchent une abstention pour les deux k.

Les preuves sont dans generated_examples.json, advanced_classification.csv,
known_item_retrieval.csv et temporal_2026.json. Le protocole de recherche known-item
est un controle automatique, pas un benchmark humain independant. L'evaluation humaine
de pertinence et de fidelite reste necessaire pour soutenir des conclusions scientifiques.
L'experience executee est bornee pour CPU ; elle ne traite pas tous les recits du corpus.
