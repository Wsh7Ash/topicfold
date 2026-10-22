"""Command-line interface for topicfold."""
from __future__ import annotations

import argparse
import sys

from .topicfold import fold_dir, render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="topicfold",
        description="Cluster text notes in a directory into topics (offline).",
    )
    parser.add_argument("directory", help="Directory containing .txt/.md notes")
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Minimum average-linkage cosine similarity to merge (default 0.3)",
    )
    parser.add_argument(
        "--dim", type=int, default=512, help="Feature-hash vector width (default 512)"
    )
    parser.add_argument(
        "--keywords", type=int, default=5, help="Keywords per topic (default 5)"
    )
    args = parser.parse_args(argv)

    clusters = fold_dir(
        args.directory, threshold=args.threshold, dim=args.dim, keywords=args.keywords
    )
    print(render_report(clusters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
