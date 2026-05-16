"""
Sync reviewed manual priority labels from a CSV into the database.

Usage:
    python .\\backend\\scripts\\sync_manual_priorities.py --csv "C:\\path\\to\\posts_rows_manual_priority.csv"

Behavior:
  - Matches rows by Post.id using the CSV `id` column
  - Updates Post.priority and Post.severity_rank from `manual_priority`
  - Optionally mirrors the same label into PostPriority.priority_label
  - Skips rows with missing IDs or invalid manual_priority values
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import app
from models import Post, PostPriority, db

VALID_LABELS = {"High", "Medium", "Low"}
SEVERITY_MAP = {"High": 3, "Medium": 2, "Low": 1}


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sync_manual_priorities(csv_path: str, update_rf_rows: bool = True) -> dict:
    rows = load_rows(csv_path)

    seen_ids: set[str] = set()
    valid_pairs: dict[str, str] = {}
    skipped_missing = 0
    skipped_invalid = 0
    skipped_duplicate = 0

    for row in rows:
        post_id = (row.get("id") or "").strip()
        label = (row.get("manual_priority") or "").strip()

        if not post_id:
            skipped_missing += 1
            continue
        if label not in VALID_LABELS:
            skipped_invalid += 1
            continue
        if post_id in seen_ids:
            skipped_duplicate += 1
            continue

        seen_ids.add(post_id)
        valid_pairs[post_id] = label

    with app.app_context():
        posts = {
            post.id: post
            for post in Post.query.filter(Post.id.in_(list(valid_pairs))).all()
        }
        rf_rows = {}
        if update_rf_rows and valid_pairs:
            rf_rows = {
                row.post_id: row
                for row in PostPriority.query.filter(PostPriority.post_id.in_(list(valid_pairs))).all()
            }

        matched = 0
        posts_changed = 0
        rf_rows_changed = 0
        not_found = 0

        for post_id, label in valid_pairs.items():
            post = posts.get(post_id)
            if not post:
                not_found += 1
                continue

            matched += 1
            severity = SEVERITY_MAP[label]
            if post.priority != label or post.severity_rank != severity:
                post.priority = label
                post.severity_rank = severity
                posts_changed += 1

            rf_row = rf_rows.get(post_id)
            if rf_row and rf_row.priority_label != label:
                rf_row.priority_label = label
                rf_rows_changed += 1

        db.session.commit()

    return {
        "csv_rows": len(rows),
        "valid_rows": len(valid_pairs),
        "matched_posts": matched,
        "posts_changed": posts_changed,
        "rf_rows_changed": rf_rows_changed,
        "not_found": not_found,
        "skipped_missing_id": skipped_missing,
        "skipped_invalid_label": skipped_invalid,
        "skipped_duplicate_id": skipped_duplicate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync manual priority labels into the database")
    parser.add_argument("--csv", required=True, help="Path to CSV with id and manual_priority columns")
    parser.add_argument(
        "--skip-rf-rows",
        action="store_true",
        help="Do not mirror manual labels into existing post_priorities rows",
    )
    args = parser.parse_args()

    result = sync_manual_priorities(
        csv_path=args.csv,
        update_rf_rows=not args.skip_rf_rows,
    )

    print("Manual priority sync complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
