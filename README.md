# topicfold

Offline note clustering. Point it at a folder of `.txt`/`.md` notes and it groups
them into topics — no API keys, no model downloads, no network. Deterministic and
dependency-free.

## How it works

1. **Tokenize** each note (lowercase, split on non-alphanumerics, drop stopwords).
2. **Feature-hash** the token bag into a fixed-width count vector using a stable
   (process-independent) hash, then L2-normalize.
3. **Cluster** with average-linkage agglomerative clustering on cosine similarity,
   merging pairs until no pair meets the threshold.
4. **Label** each cluster by its most common distinguishing keywords.

Because the hash is stable and every step is deterministic, the same input always
produces the same clusters and the same report.

## Install

```
pip install -e .
```

## Usage

```
topicfold path/to/notes --threshold 0.2 --keywords 5
```

Options:
- `--threshold` — minimum average-linkage cosine similarity to merge (default 0.3).
  Lower merges more aggressively.
- `--dim` — feature-hash vector width (default 512).
- `--keywords` — keywords shown per topic (default 5).

## Python API

```python
from topicfold import fold, render_report

notes = [
    ("db1.txt", "database migration rollback strategy"),
    ("db2.txt", "postgres schema migration plan"),
    ("cake.txt", "chocolate cake recipe with flour"),
]
clusters = fold(notes, threshold=0.15)
print(render_report(clusters))
```

## License

MIT
