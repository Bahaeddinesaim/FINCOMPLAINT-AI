"""Conservative normalization; original narratives remain available."""

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def narrative_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()
