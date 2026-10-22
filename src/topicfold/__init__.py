"""Offline note clustering into topics. Deterministic, dependency-free."""
from .topicfold import (
    Cluster,
    Document,
    agglomerative_cluster,
    cosine,
    feature_hash,
    fold,
    fold_dir,
    render_report,
    tokenize,
    vectorize,
)

__all__ = [
    "Cluster",
    "Document",
    "agglomerative_cluster",
    "cosine",
    "feature_hash",
    "fold",
    "fold_dir",
    "render_report",
    "tokenize",
    "vectorize",
]
