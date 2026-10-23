"""Leakage checks of training questions against the eval set and the few-shot bank (task 8.2).

Training states use other cameras than the eval catalog, so a literal comparison would miss a
test question reworded for another camera. Every question is first masked with its own
catalog: camera, zone and class names (as listed and with spaces for underscores) become
`<camera>`, `<zone>`, `<class>`; numbers become `<n>`; month names `<month>`. On masked
questions three checks run against every eval question (dev and test) and few-shot example:
- template: the masked questions are identical;
- near duplicate: difflib ratio or word Jaccard at or above the eval's thresholds;
- lexical embedding: cosine of character 3-gram TF-IDF vectors at or above `min_cosine`.
A neural sentence embedding would also catch paraphrases with different words; none is a
project dependency, which is a stated limitation of this check.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from vms.eval.dataset import near_duplicates
from vms.query.catalog import QueryCatalog
from vms.query.fewshots import normalize_question

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
]
_NUMBER = re.compile(r"\b\d+(?:[.:]\d+)?(?:st|nd|rd|th|am|pm)?\b")


def _names(catalog: QueryCatalog) -> list[tuple[str, str]]:
    pairs = [(c.name, "<camera>") for c in catalog.cameras]
    pairs += [(z.name, "<zone>") for z in catalog.zones]
    pairs += [(k, "<class>") for k in catalog.classes]
    variants = []
    for name, tag in pairs:
        variants.append((name.lower(), tag))
        if "_" in name:
            variants.append((name.lower().replace("_", " "), tag))
    # Longest first, so "cell phone" is masked before "phone"-like substrings of other names.
    return sorted(set(variants), key=lambda p: -len(p[0]))


def mask(question: str, catalog: QueryCatalog) -> str:
    text = normalize_question(question)
    for name, tag in _names(catalog):
        text = re.sub(rf"(?<![a-z0-9_]){re.escape(name)}s?(?![a-z0-9_])", tag, text)
    text = _NUMBER.sub("<n>", text)
    for month in MONTHS:
        text = re.sub(rf"\b{month}\b", "<month>", text)
    return " ".join(text.split())


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    padded = f" {text} "
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def tfidf_matrix(docs: Sequence[str], vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    m = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        for gram, count in char_ngrams(doc).items():
            j = vocab.get(gram)
            if j is not None:
                m[i, j] = (1.0 + math.log(count)) * idf[j]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def max_cosines(
    train: Sequence[str], ref: Sequence[str], chunk: int = 512
) -> list[tuple[float, int]]:
    """For each training doc, the highest cosine to any reference doc and its index."""
    if not train or not ref:
        return [(0.0, -1)] * len(train)
    df: Counter[str] = Counter()
    for doc in [*train, *ref]:
        df.update(set(char_ngrams(doc)))
    vocab = {g: i for i, g in enumerate(sorted(df))}
    n = len(train) + len(ref)
    idf = np.array([math.log((1 + n) / (1 + df[g])) + 1.0 for g in sorted(df)], dtype=np.float32)
    r = tfidf_matrix(ref, vocab, idf)
    out: list[tuple[float, int]] = []
    for start in range(0, len(train), chunk):
        t = tfidf_matrix(train[start : start + chunk], vocab, idf)
        sims = t @ r.T
        best = sims.argmax(axis=1)
        out += [(float(sims[i, b]), int(b)) for i, b in enumerate(best)]
    return out


@dataclass(frozen=True)
class Hit:
    train_id: str
    ref_id: str
    check: str  # template | near_duplicate | lexical_embedding
    score: float


def leakage_hits(
    train: Sequence[tuple[str, str]],  # (id, masked question)
    ref: Sequence[tuple[str, str]],
    *,
    min_ratio: float,
    min_jaccard: float,
    min_cosine: float,
) -> list[Hit]:
    hits: list[Hit] = []
    by_text: dict[str, str] = {}
    for rid, text in ref:
        by_text.setdefault(text, rid)
    for tid, text in train:
        if text in by_text:
            hits.append(Hit(tid, by_text[text], "template", 1.0))
    for d in near_duplicates(train, ref, min_ratio=min_ratio, min_jaccard=min_jaccard):
        hits.append(Hit(d.a, d.b, "near_duplicate", max(d.ratio, d.jaccard)))
    cosines = max_cosines([t for _, t in train], [t for _, t in ref])
    for (tid, _), (score, j) in zip(train, cosines, strict=True):
        if score >= min_cosine:
            hits.append(Hit(tid, ref[j][0], "lexical_embedding", round(score, 4)))
    return hits


def masked(rows: Iterable[tuple[str, str, QueryCatalog]]) -> list[tuple[str, str]]:
    return [(rid, mask(q, catalog)) for rid, q, catalog in rows]
