"""
Import an exported Apify X comments dataset into the local database.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app import app, ensure_database
from data import extract_location, infer_cluster
from models import Comment, Post, PostSentiment, PreprocessedText, db
from preprocessing import save_preprocessed_text
from services.vader.sentiment_analyzer import analyze_post_with_comments
from x_matching import build_post_match_index, find_post_match


def safe_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def comment_id_for(item: dict):
    fingerprint = "||".join(
        [
            str(item.get("postId") or ""),
            item.get("postUrl") or "",
            item.get("replyUrl") or "",
            item.get("replyId") or "",
            item.get("text") or item.get("comment") or "",
        ]
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def parse_datetime(value):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def normalize_item(item: dict, post: Post | None):
    text = (item.get("text") or item.get("comment") or "").strip()
    title = (item.get("postTitle") or "").strip()
    combined = f"{title}\n{text}".strip()
    author_name = (item.get("authorName") or "").strip()
    author_username = (item.get("authorUsername") or "").strip()
    inferred_cluster, _keywords = infer_cluster(combined)

    return {
        "id": comment_id_for(item),
        "post_id": post.id if post else None,
        "source": "X",
        "page_source": post.page_source if post else (author_username or author_name or "X Source"),
        "author": author_name or author_username or "X user",
        "text": text,
        "likes": safe_int(item.get("likesCount")),
        "post_title": title,
        "post_url": item.get("postUrl") or (post.source_url if post else ""),
        "cluster_id": post.cluster_id if post else inferred_cluster["id"],
        "location": post.location if post else extract_location(combined),
        "date": parse_datetime(item.get("timestamp")),
        "raw_payload_json": json.dumps(item),
    }


def import_items(payload: list[dict]):
    inserted = updated = skipped = processed = errors = 0
    translated = translation_failed = negation_handled = lemmatized = 0
    bigrams_detected = emotion_only_flagged = irrelevant_flagged = 0

    ensure_database()
    with app.app_context():
        post_match_index = build_post_match_index(Post.query.filter_by(source="X").all())
        affected_post_ids: set[str] = set()

        for item in payload:
            text = (item.get("text") or item.get("comment") or "").strip()
            post_url = item.get("postUrl") or ""
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
                conversation_id=item.get("conversationId"),
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

        if affected_post_ids:
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

    return {
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
