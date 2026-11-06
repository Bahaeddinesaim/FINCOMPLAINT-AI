# Protocole et limites

Macro-F1, precision, rappel par classe et matrices de confusion. La comparaison avancee
utilise les memes 235 textes train et 74 validation. La baseline globale utilise 2 024
textes train et 497 validation : ces deux experiences ne sont pas equivalentes.

Le test 2026 retient au maximum 1 000 textes par hash apres filtrage des premieres
lignes des sept exports et exclusion des doublons connus. Le modele n'est pas reajuste.

La recherche known-item retrouve un document depuis ses propres 20 premiers mots.
Cette mesure reproductible est optimiste et ne mesure pas la pertinence de requetes
utilisateur nouvelles. Un jeu humain annote independant reste necessaire.

Le RAG est observe sur une question metier et deux questions hors domaine, avec k=3/5.
Les erreurs, brouillons retenus et references sont enregistres. Le seuil et le petit
generateur ne garantissent pas l'abstention sur toute question hors corpus.

La couverture logicielle ne prouve ni absence de bugs ni validite scientifique.
Les vrais modeles sont executes dans les integrations locales, separement des tests
unitaires avec dependances injectees. Les captures web ne constituent pas une evaluation NLP.
