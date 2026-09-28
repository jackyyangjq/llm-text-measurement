"""The pre-LLM baseline: a dictionary grown from seed words with word embeddings.

Train Word2Vec (skip-gram) on a large corpus, take a handful of seed words per category, and add
every word whose cosine similarity to the category's seed centroid passes a threshold. A text
mentions a category if it contains any of its words. This is how many text-based measures in
economics and management are built; `evaluate` compares it with the LLM's labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import tokenize

SEEDS = {
    "location": ["location", "located", "metro", "walk", "central"],
    "room": ["room", "bed", "rooms", "beds", "spacious"],
    "cleanliness": ["clean", "dirty", "dust", "stains", "spotless"],
    "staff": ["staff", "reception", "receptionist", "friendly", "helpful"],
    "food": ["breakfast", "restaurant", "food", "coffee", "bar"],
    "value": ["price", "expensive", "value", "overpriced", "cheap"],
    "noise": ["noise", "noisy", "loud", "quiet", "soundproof"],
    "facilities": ["wifi", "pool", "gym", "parking", "lift"],
}


def train_embeddings(texts: list[str], seed: int = 0):
    """Skip-gram Word2Vec, 200 dimensions, window 5, words seen at least 20 times."""
    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    return Word2Vec(sentences, vector_size=200, window=5, min_count=20, sg=1, workers=4, seed=seed, epochs=5)


def expand(model, seeds: dict[str, list[str]] = SEEDS, threshold: float = 0.6, top_n: int = 60) -> dict[str, list[str]]:
    """Seed words plus the neighbours of each category's seed centroid above `threshold`. A word
    closer to another category's centroid is left to that category, so the lists do not overlap."""
    wv = model.wv
    centroids = {k: np.mean([wv[w] for w in v if w in wv], axis=0) for k, v in seeds.items()}
    unit = {k: c / np.linalg.norm(c) for k, c in centroids.items()}
    out = {}
    for cat, seed_words in seeds.items():
        words = [w for w in seed_words if w in wv]
        for word, sim in wv.similar_by_vector(centroids[cat], topn=top_n):
            if sim < threshold or word in words:
                continue
            v = wv[word] / np.linalg.norm(wv[word])
            if max(unit, key=lambda k: float(v @ unit[k])) == cat:
                words.append(word)
        out[cat] = words
    return out


def save(dictionary: dict[str, list[str]], path: Path) -> None:
    path.write_text(json.dumps(dictionary, indent=1) + "\n")


def load(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def tag(texts: pd.Series, dictionary: dict[str, list[str]]) -> pd.DataFrame:
    """1 where a text contains any word of the category."""
    tokens = texts.map(lambda t: set(tokenize(t)))
    return pd.DataFrame(
        {cat: tokens.map(lambda s, w=set(words): int(bool(s & w))) for cat, words in dictionary.items()}
    )
