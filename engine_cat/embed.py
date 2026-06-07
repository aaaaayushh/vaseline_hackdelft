"""
Embedding tier — semantic mapping of free text into the taxonomy.

We embed each taxonomy anchor once, then embed any input text (an MCC
description, or a merchant name) and assign it to the nearest anchor by cosine
similarity. This replaces a hand-maintained MCC->category lookup with a
principled, generalising mapping, and the returned similarity doubles as a
confidence score: low-similarity assignments are the ones we escalate to the
LLM tier.

Model: sentence-transformers/all-MiniLM-L6-v2 (small, fast, local, no API key).
"""
from __future__ import annotations

import functools

import numpy as np

from .taxonomy import CATEGORIES, TAXONOMY

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def _anchor_matrix() -> np.ndarray:
    anchors = [TAXONOMY[c] for c in CATEGORIES]
    return _model().encode(anchors, normalize_embeddings=True)


def map_texts_to_taxonomy(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """
    Map each input text to its nearest taxonomy category.

    Returns (categories, similarities) where similarities[i] in [-1, 1] is the
    cosine similarity of texts[i] to its assigned anchor — i.e. the confidence.
    """
    if not texts:
        return [], np.array([])
    emb = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    sims = emb @ _anchor_matrix().T            # (n_texts, n_categories)
    idx = sims.argmax(axis=1)
    cats = [CATEGORIES[i] for i in idx]
    conf = sims[np.arange(len(idx)), idx]
    return cats, conf
