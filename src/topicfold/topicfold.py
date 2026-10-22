"""Offline note clustering with feature-hash vectors and average-linkage
agglomerative clustering. No network, no model downloads, no dependencies.

The pipeline is fully deterministic: tokens are hashed into a fixed-width
count vector with a stable (process-independent) hash, vectors are L2
normalized, and clusters are merged by average-linkage cosine similarity
until no pair meets the merge threshold.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its of on
    or that the their then there these this to was were will with you your not
    we us our i he she they them his her had do does did so no yes than too very
    can could should would about after before over under again once each few all
    any both more most other some such only own same s t don now""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _stable_hash(token: str) -> int:
    # md5 hexdigest -> int is stable across processes and platforms, unlike
    # Python's built-in hash() which is salted per process.
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def feature_hash(token: str, dim: int) -> int:
    """Map a token to a bucket in [0, dim)."""
    return _stable_hash(token) % dim


def vectorize(tokens: Iterable[str], dim: int = 512) -> list[float]:
    """Bag-of-words count vector hashed into `dim` buckets, L2 normalized."""
    vec = [0.0] * dim
    for tok in tokens:
        vec[feature_hash(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two L2-normalized vectors (cosine similarity)."""
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Document:
    name: str
    text: str
    tokens: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)


@dataclass
class Cluster:
    docs: list[str]
    keywords: list[str]


def _cluster_similarity(vecs: list[list[float]], c1: list[int], c2: list[int]) -> float:
    total = 0.0
    count = 0
    for i in c1:
        for j in c2:
            total += cosine(vecs[i], vecs[j])
            count += 1
    return total / count if count else 0.0


def agglomerative_cluster(
    vectors: list[list[float]], threshold: float = 0.3
) -> list[list[int]]:
    """Average-linkage agglomerative clustering. Returns index groups.

    Starts with each vector in its own cluster and repeatedly merges the pair
    with the highest average-linkage cosine similarity while that similarity is
    >= `threshold`.
    """
    clusters: list[list[int]] = [[i] for i in range(len(vectors))]
    while len(clusters) > 1:
        best_pair = None
        best_sim = -1.0
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                sim = _cluster_similarity(vectors, clusters[a], clusters[b])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (a, b)
        if best_pair is None or best_sim < threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters


def _keywords_for(
    docs: list[Document], indices: list[int], top_n: int
) -> list[str]:
    counter: Counter[str] = Counter()
    for i in indices:
        counter.update(t for t in docs[i].tokens if t not in _STOPWORDS)
    return [tok for tok, _ in counter.most_common(top_n)]


def fold(
    notes: Iterable[tuple[str, str]],
    threshold: float = 0.3,
    dim: int = 512,
    keywords: int = 5,
) -> list[Cluster]:
    """Cluster (name, text) notes into topic groups.

    Returns a list of Clusters, each with the member note names and the most
    common distinguishing keywords across its members.
    """
    docs: list[Document] = []
    for name, text in notes:
        toks = tokenize(text)
        docs.append(Document(name=name, text=text, tokens=toks, vector=vectorize(toks, dim)))
    if not docs:
        return []
    vectors = [d.vector for d in docs]
    groups = agglomerative_cluster(vectors, threshold)
    clusters = [
        Cluster(
            docs=[docs[i].name for i in group],
            keywords=_keywords_for(docs, group, keywords),
        )
        for group in groups
    ]
    # Deterministic ordering: by size desc, then first member name.
    clusters.sort(key=lambda c: (-len(c.docs), c.docs[0]))
    return clusters


def fold_dir(
    directory: str | Path,
    threshold: float = 0.3,
    dim: int = 512,
    keywords: int = 5,
    extensions: tuple[str, ...] = (".txt", ".md", ".markdown"),
) -> list[Cluster]:
    """Cluster every matching text file in a directory (by filename)."""
    d = Path(directory)
    notes = [
        (p.name, p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(d.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return fold(notes, threshold=threshold, dim=dim, keywords=keywords)


def render_report(clusters: list[Cluster]) -> str:
    """Human-readable grouped report."""
    lines: list[str] = []
    lines.append(f"{len(clusters)} topic cluster(s)")
    lines.append("")
    for idx, cl in enumerate(clusters, 1):
        kw = ", ".join(cl.keywords) if cl.keywords else "(no keywords)"
        lines.append(f"[{idx}] {kw}  ({len(cl.docs)} notes)")
        for name in cl.docs:
            lines.append(f"    - {name}")
    return "\n".join(lines)
