"""
One-time data fix: reset stale `cluster_label_source` values so the full
pipeline is allowed to re-evaluate them.

Why this exists:
  The pipeline's SVM predict-all step refuses to overwrite a post's
  cluster_id when cluster_label_source is in {"reviewed", "corex_enriched"}.
  After the threshold-loosening + cluster-a-default fix, we want every
  heuristic-labeled post to be re-evaluated. Marking those rows with a
  neutral source value makes them eligible for the refinement passes.

What it does NOT touch:
  - Rows where cluster_label_source == "reviewed" (human-curated)
  - Rows where cluster_label_source == "corex_enriched" (strong-signal
    decision from a prior pipeline run that we trust)
  - Post.cluster_id values — only the *source* tag changes.

Usage (from the backend/ directory):
    python scripts/reset_label_sources.py
"""
from __future__ import annotations

import os
import sys

# Make the backend/ directory importable when run as a script
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app import app, ensure_database  # noqa: E402
from models import Post, db  # noqa: E402


PROTECTED_SOURCES = ("reviewed", "corex_enriched")
NEW_SOURCE = "pending_refine"  # neutral tag — pipeline treats this same as "heuristic"


def main() -> int:
    ensure_database()
    with app.app_context():
        total = Post.query.count()
        protected = Post.query.filter(Post.cluster_label_source.in_(PROTECTED_SOURCES)).count()

        eligible = Post.query.filter(
            (Post.cluster_label_source.is_(None))
            | (~Post.cluster_label_source.in_(PROTECTED_SOURCES))
        )
        updated = eligible.update(
            {Post.cluster_label_source: NEW_SOURCE},
            synchronize_session=False,
        )
        db.session.commit()

        print(f"Posts total:                 {total}")
        print(f"Posts protected (untouched): {protected}")
        print(f"Posts reset to '{NEW_SOURCE}': {updated}")
        print()
        print("Next step: POST /api/admin/pipeline/run-all  body: {\"force_retrain\": true}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
