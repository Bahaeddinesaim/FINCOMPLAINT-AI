# Architecture

Ingestion et schema dans `ingestion.py`, texte et empreintes dans `preprocessing.py`,
baseline dans `classifier.py`, index lexical dans `retriever.py`, modeles dans
`advanced.py`, HTTP et interface dans `api.py`. Le notebook orchestre ces composants.

Lecture par blocs de 10 000 lignes et stockage SQLite. Les identifiants et textes
normalises sont uniques. Le hash divise 2025 en train et validation ; 2026 reste un
test temporel. Les quasi-doublons restent une limite.

DistilBERT et TF-IDF sont compares sur les memes petits sous-jeux. L'encodeur est gele
pour limiter le cout CPU. MiniLM normalise ses embeddings ; FAISS IndexFlatIP calcule
le cosinus. Les segments conservent identifiant, date et fichier source.

FLAN-T5 recoit les extraits et une consigne de citation. Le controle des IDs ne
remplace pas l'evaluation de la fidelite. L'interface consomme les memes endpoints
que Swagger et insere les textes avec textContent, sans executer de HTML retourne.
