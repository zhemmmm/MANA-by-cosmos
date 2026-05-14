"""
Import an exported Apify Facebook comments dataset into the local SQLite database.

Example:
    python import_facebook_comments_dataset.py --file "C:\\Users\\USER\\Downloads\\dataset_facebook-comments-scraper.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app import app, ensure_database
from data import extract_location, infer_cluster, now_utc
from facebook_matching import build_post_match_index, find_post_match, normalize_facebook_url
from models import Comment, Post, PostSentiment, PreprocessedText, db
from preprocessing import save_preprocessed_text


def safe_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def comment_id_for(item: dict):
    fingerprint = "||".join(
        [
            normalize_facebook_url(item.get("facebookUrl") or ""),
            item.get("postTitle") or "",
            item.get("text") or item.get("comment") or item.get("body") or item.get("message") or "",
            str(item.get("likesCount") or "0"),
        ]
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def normalize_item(item: dict, post: Post | None):
    text = (item.get("text") or item.get("comment") or item.get("body") or item.get("message") or "").strip()
    title = (item.get("postTitle") or "").strip()
    combined = f"{title}\n{text}".strip()
    inferred_cluster, _keywords = infer_cluster(combined)

    return {
        "id": comment_id_for(item),
        "post_id": post.id if post else None,
        "source": "Facebook",
        "page_source": post.page_source if post else "Facebook Source",
        "author": (item.get("authorName") or item.get("author") or "Facebook user").strip() or "Facebook user",
        "text": text,
        "likes": safe_int(item.get("likesCount")),
        "post_title": title,
        "post_url": item.get("facebookUrl") or (post.source_url if post else ""),
        "cluster_id": post.cluster_id if post else inferred_cluster["id"],
        "location": post.location if post else extract_location(combined),
        "date": post.date if post and post.date else now_utc(),
        "raw_payload_json": json.dumps(item),
    }


def import_items(payload: list[dict]):
    inserted = updated = skipped = processed = errors = 0
    translated = translation_failed = negation_handled = lemmatized = 0
    bigrams_detected = emotion_only_flagged = irrelevant_flagged = 0

    ensure_database()
    with app.app_context():
        post_match_index = build_post_match_index(Post.query.all())
        affected_post_ids: set[str] = set()
        for item in payload:
            text = (item.get("text") or item.get("comment") or item.get("body") or item.get("message") or "").strip()
            post_url = item.get("facebookUrl") or ""
            if not text or not post_url:
                processed_row, processed_payload = save_preprocessed_text(
                    item=item,
                    raw_id=comment_id_for(item),
                    record_type="comment",
                    fallback_text=text,
                    parent_context_text=(item.get("postTitle") or "").strip(),
                )
                db.session.add(processed_row)
                stats = processed_payload["stats"]
                errors += stats["errors"] or (1 if processed_payload["preprocessing_status"] == "error" else 0)
                translated += stats["translated"]
                translation_failed += stats["translation_failed"]
                negation_handled += stats["negation_handled"]
                lemmatized += stats["lemmatized"]
                bigrams_detected += stats["bigrams_detected"]
                emotion_only_flagged += stats["emotion_only_flagged"]
                irrelevant_flagged += stats["irrelevant_flagged"]
                skipped += 1
                continue

            post = find_post_match(
                post_match_index,
                url=post_url,
                external_id=item.get("postId"),
            )
            normalized = normalize_item(item, post)
            if normalized["post_id"]:
                affected_post_ids.add(normalized["post_id"])
            comment = db.session.get(Comment, normalized["id"])

            if comment:
                for field, value in normalized.items():
                    setattr(comment, field, value)
                updated += 1
            else:
                db.session.add(Comment(**normalized))
                inserted += 1

            processed_row, processed_payload = save_preprocessed_text(
                item=item,
                raw_id=normalized["id"],
                record_type="comment",
                fallback_text=normalized["text"],
                parent_post_id=normalized["post_id"],
                parent_context_text=(post.caption if post else normalized["post_title"]),
            )
            db.session.add(processed_row)
            status = processed_payload["preprocessing_status"]
            stats = processed_payload["stats"]
            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
            translated += stats["translated"]
            translation_failed += stats["translation_failed"]
            negation_handled += stats["negation_handled"]
            lemmatized += stats["lemmatized"]
            bigrams_detected += stats["bigrams_detected"]
            emotion_only_flagged += stats["emotion_only_flagged"]
            irrelevant_flagged += stats["irrelevant_flagged"]
            if status != "error":
                errors += stats["errors"]

        db.session.commit()

        # Recompute parent post sentiment after comments are imported. This keeps
        # the existing post-level sentiment fields current without adding tables.
        if affected_post_ids:
            try:
                from services.vader.sentiment_analyzer import analyze_post_with_comments

                post_texts = {
                    row.raw_id: (row.vader_text or row.clean_text)
                    for row in PreprocessedText.query
                        .filter(PreprocessedText.raw_id.in_(affected_post_ids))
                        .filter_by(record_type="post")
                        .all()
                    if (row.vader_text or row.clean_text)
                }
                comments_by_post: dict[str, list[str]] = {}
                for comment in Comment.query.filter(Comment.post_id.in_(affected_post_ids)).all():
                    comments_by_post.setdefault(comment.post_id, []).append(comment.text or "")

                for post in Post.query.filter(Post.id.in_(affected_post_ids)).all():
                    result = analyze_post_with_comments(
                        post_texts.get(post.id) or post.caption or "",
                        post.cluster_id,
                        comments_by_post.get(post.id, []),
                    )
                    existing = PostSentiment.query.filter_by(post_id=post.id).first()
                    if existing:
                        existing.compound = result["compound"]
                        existing.positive = result["positive"]
                        existing.negative = result["negative"]
                        existing.neutral = result["neutral"]
                        existing.sarcasm_flag = result["sarcasm_flag"]
                    else:
                        db.session.add(PostSentiment(
                            post_id=post.id,
                            compound=result["compound"],
                            positive=result["positive"],
                            negative=result["negative"],
                            neutral=result["neutral"],
                            sarcasm_flag=result["sarcasm_flag"],
                        ))
                    post.sentiment_score = result["sentiment_score"]
                    post.sentiment_compound = result["compound"]
                db.session.commit()
            except Exception:
                db.session.rollback()

    summary = {
        "total_records_loaded": len(payload),
        "inserted": inserted,
        "updated": updated,
        "processed": processed,
        "skipped": skipped,
        "translated": translated,
        "translation_failed": translation_failed,
        "negation_handled": negation_handled,
        "lemmatized": lemmatized,
        "bigrams_detected": bigrams_detected,
        "emotion_only_flagged": emotion_only_flagged,
        "irrelevant_flagged": irrelevant_flagged,
        "errors": errors,
    }
    return summary


def import_dataset(file_path: Path):
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return import_items(payload)


def main():
    parser = argparse.ArgumentParser(description="Import exported Apify Facebook comments dataset into MANA SQLite DB.")
    parser.add_argument("--file", required=True, help="Path to the exported JSON comments dataset file.")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Dataset file not found: {file_path}")

    summary = import_dataset(file_path)
    print(f"Imported comments dataset from {file_path}")
    print(f"Total records loaded: {summary['total_records_loaded']}")
    print(f"Inserted: {summary['inserted']}")
    print(f"Updated: {summary['updated']}")
    print(f"Total records processed: {summary['processed']}")
    print(f"Total records skipped: {summary['skipped']}")
    print(f"Translated count: {summary['translated']}")
    print(f"Translation failed count: {summary['translation_failed']}")
    print(f"Negation handled count: {summary['negation_handled']}")
    print(f"Lemmatized count: {summary['lemmatized']}")
    print(f"Bigrams detected count: {summary['bigrams_detected']}")
    print(f"Emotion-only flagged count: {summary['emotion_only_flagged']}")
    print(f"Irrelevant flagged count: {summary['irrelevant_flagged']}")
    print(f"Total errors: {summary['errors']}")


if __name__ == "__main__":
    main()
