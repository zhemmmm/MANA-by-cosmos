"""
Import an exported Apify Facebook dataset into the local SQLite database.

Example:
    python import_facebook_dataset.py --file "C:\\Users\\USER\\Downloads\\dataset_facebook-posts-scraper_2026-04-26_20-46-25-366.json"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app import app, ensure_database
from data import (
    CLUSTER_DEFINITIONS,
    infer_cluster,
    infer_priority,
    infer_sentiment_score,
    media_type_for,
    recommendation_payload_for,
    extract_location,
    PRIORITY_ORDER,
)
from models import Comment, Post, PostCluster, PostSentiment, PostTopic, PreprocessedText, db
from preprocessing import save_preprocessed_text
from services.corex.topic_modeler import is_model_trained, predict_topics_batch
from services.svm.cluster_classifier import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    is_model_trained as is_svm_trained,
    predict_clusters_batch,
    select_top_cluster,
)


def safe_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


REACTION_COUNT_FIELDS = (
    "reactionLikeCount",
    "reactionLoveCount",
    "reactionCareCount",
    "reactionHahaCount",
    "reactionWowCount",
    "reactionSadCount",
    "reactionAngryCount",
)


def extract_reaction_totals(item: dict) -> tuple[int, int]:
    """
    Return (total_reactions, like_reactions) from the real Apify post payload.

    In exported Facebook Posts Scraper data, `topReactionsCount` is often just the
    count of the visible top reaction summary, not the total reactions on the post.
    The accurate totals come from the per-reaction fields when present.
    """
    reaction_breakdown = {field: safe_int(item.get(field)) for field in REACTION_COUNT_FIELDS}
    breakdown_total = sum(reaction_breakdown.values())
    fallback_total = safe_int(item.get("likes")) or safe_int(item.get("topReactionsCount"))
    total_reactions = breakdown_total or fallback_total
    like_reactions = reaction_breakdown["reactionLikeCount"] or fallback_total
    return total_reactions, like_reactions


def metrics_from_item(item: dict) -> dict:
    reactions, like_reactions = extract_reaction_totals(item)
    return {
        "reactions": reactions,
        "likes": like_reactions,
        "shares": safe_int(item.get("shares")),
        "comments": safe_int(item.get("comments")),
        "views": safe_int(item.get("viewsCount")),
    }


def refresh_post_metrics_from_payload(post: Post) -> bool:
    if not post.raw_payload_json:
        return False
    try:
        item = json.loads(post.raw_payload_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    metrics = metrics_from_item(item)
    changed = False
    for field, value in metrics.items():
        if getattr(post, field) != value:
            setattr(post, field, value)
            changed = True
    return changed


def parse_iso_datetime(value: str):
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_item(item: dict):
    text = (item.get("text") or item.get("caption") or item.get("content") or "").strip()
    cluster, keywords = infer_cluster(text)
    if cluster is None:
        # Keep the import running even when a post has too little signal to classify.
        cluster = CLUSTER_DEFINITIONS[0]
        keywords = keywords or []
    metrics = metrics_from_item(item)
    reactions = metrics["reactions"]
    like_reactions = metrics["likes"]
    shares = metrics["shares"]
    comments = metrics["comments"]
    engagement = reactions + comments + shares
    priority = infer_priority(text, engagement)
    sentiment_score = infer_sentiment_score(text, engagement)
    recommendation_payload = recommendation_payload_for(
        cluster["id"],
        priority,
        sentiment_score=sentiment_score,
        reactions=reactions,
        likes=like_reactions,
        comments=comments,
        shares=shares,
        post_count=1,
    )

    post_id = str(item.get("postId") or item.get("url") or item.get("topLevelUrl"))
    return {
        "id": post_id,
        "source": "Facebook",
        "page_source": item.get("pageName") or "Facebook Source",
        "account_url": item.get("facebookUrl") or item.get("inputUrl"),
        "author": (item.get("user") or {}).get("name") or item.get("pageName"),
        "caption": text,
        "source_url": item.get("url") or item.get("topLevelUrl") or item.get("facebookUrl"),
        "external_id": str(item.get("postId") or ""),
        "reactions": reactions,
        "shares": shares,
        "likes": like_reactions,
        "reposts": 0,
        "comments": comments,
        "views": metrics["views"],
        "media_type": media_type_for(item),
        "priority": priority,
        "sentiment_score": sentiment_score,
        "recommendation": recommendation_payload["recommendation"],
        "status": "Monitoring",
        "cluster_id": cluster["id"],
        "cluster_label_source": "heuristic",
        "date": parse_iso_datetime(item.get("time")),
        "keywords_json": json.dumps(keywords),
        "location": extract_location(text),
        "severity_rank": PRIORITY_ORDER[priority],
        "raw_payload_json": json.dumps(item),
    }


def import_items(payload: list[dict]):
    inserted = updated = processed = skipped = errors = 0
    translated = translation_failed = negation_handled = lemmatized = 0
    bigrams_detected = emotion_only_flagged = irrelevant_flagged = 0

    ensure_database()
    with app.app_context():
        for item in payload:
            normalized = normalize_item(item)
            post = db.session.get(Post, normalized["id"])
            if post:
                for field, value in normalized.items():
                    setattr(post, field, value)
                updated += 1
            else:
                db.session.add(Post(**normalized))
                inserted += 1

            processed_row, processed_payload = save_preprocessed_text(
                item=item,
                raw_id=normalized["id"],
                record_type="post",
                fallback_text=normalized["caption"],
            )
            db.session.add(processed_row)
            post = db.session.get(Post, normalized["id"])
            if post:
                post.is_relevant = processed_payload["is_relevant"]
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
            errors += stats["errors"] or (1 if status == "error" else 0)
        db.session.commit()

        # If CorEx model is already trained, run topic inference on newly imported posts.
        corex_topics_assigned = 0
        if is_model_trained():
            new_post_ids = [
                item.get("postId") or item.get("url") or item.get("topLevelUrl")
                for item in payload
            ]
            rows = (
                PreprocessedText.query
                .filter(PreprocessedText.raw_id.in_([str(pid) for pid in new_post_ids if pid]))
                .filter_by(preprocessing_status="processed", is_relevant=True)
                .filter(PreprocessedText.final_tokens_json != "[]")
                .all()
            )
            if rows:
                texts = [" ".join(row.final_tokens) for row in rows]
                batch_results = predict_topics_batch(texts)
                for row, topic_list in zip(rows, batch_results):
                    for topic_item in topic_list:
                        existing = PostTopic.query.filter_by(
                            post_id=row.raw_id, topic_label=topic_item["topic"]
                        ).first()
                        if existing:
                            existing.confidence = topic_item["confidence"]
                        else:
                            db.session.add(PostTopic(
                                post_id=row.raw_id,
                                topic_label=topic_item["topic"],
                                confidence=topic_item["confidence"],
                            ))
                        corex_topics_assigned += 1
                db.session.commit()

        # If SVM model is already trained, run cluster inference on newly imported posts.
        svm_clusters_assigned = 0
        if is_svm_trained():
            new_post_ids = [
                item.get("postId") or item.get("url") or item.get("topLevelUrl")
                for item in payload
            ]
            svm_rows = (
                PreprocessedText.query
                .filter(PreprocessedText.raw_id.in_([str(pid) for pid in new_post_ids if pid]))
                .filter_by(preprocessing_status="processed", is_relevant=True, record_type="post")
                .filter(PreprocessedText.final_tokens_json != "[]")
                .all()
            )
            if svm_rows:
                svm_texts = [" ".join(row.final_tokens) for row in svm_rows]
                svm_results = predict_clusters_batch(svm_texts)
                for row, cluster_list in zip(svm_rows, svm_results):
                    if not cluster_list:
                        continue
                    for cluster_item in cluster_list:
                        existing = PostCluster.query.filter_by(
                            post_id=row.raw_id, cluster_id=cluster_item["cluster_id"]
                        ).first()
                        if existing:
                            existing.confidence = cluster_item["confidence"]
                        else:
                            db.session.add(PostCluster(
                                post_id=row.raw_id,
                                cluster_id=cluster_item["cluster_id"],
                                confidence=cluster_item["confidence"],
                            ))
                        svm_clusters_assigned += 1
                    # Only overwrite the visible cluster when the SVM is clearly confident.
                    top_cluster = select_top_cluster(
                        cluster_list,
                        min_confidence=DEFAULT_MIN_CONFIDENCE,
                        min_margin=DEFAULT_MIN_MARGIN,
                    )
                    post = db.session.get(Post, row.raw_id)
                    if post and top_cluster and post.cluster_id != top_cluster["cluster_id"]:
                        post.cluster_id = top_cluster["cluster_id"]
                        post.cluster_label_source = "svm"
                db.session.commit()

        # Run VADER sentiment analysis on newly imported posts.
        vader_sentiments_assigned = 0
        try:
            from services.vader.sentiment_analyzer import analyze_post_with_comments as _vader_analyze

            new_post_ids_set = {
                str(item.get("postId") or item.get("url") or item.get("topLevelUrl"))
                for item in payload
            }
            new_posts = Post.query.filter(Post.id.in_(new_post_ids_set)).all()
            comments_by_post: dict[str, list[str]] = {}
            for comment in Comment.query.filter(Comment.post_id.in_(new_post_ids_set)).all():
                comments_by_post.setdefault(comment.post_id, []).append(comment.text or "")

            pt_lookup: dict[str, str] = {
                row.raw_id: (row.vader_text or row.clean_text)
                for row in PreprocessedText.query
                    .filter(PreprocessedText.raw_id.in_(new_post_ids_set))
                    .filter_by(record_type="post")
                    .all()
                if (row.vader_text or row.clean_text)
            }

            for post in new_posts:
                text   = pt_lookup.get(post.id) or post.caption or ""
                result = _vader_analyze(text, post.cluster_id, comments_by_post.get(post.id, []))

                existing = PostSentiment.query.filter_by(post_id=post.id).first()
                if existing:
                    existing.compound     = result["compound"]
                    existing.positive     = result["positive"]
                    existing.negative     = result["negative"]
                    existing.neutral      = result["neutral"]
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

                post.sentiment_score    = result["sentiment_score"]
                post.sentiment_compound = result["compound"]
                vader_sentiments_assigned += 1

            db.session.commit()
        except Exception:
            db.session.rollback()
            vader_sentiments_assigned = -1

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
        "corex_topics_assigned": corex_topics_assigned,
        "svm_clusters_assigned": svm_clusters_assigned,
        "vader_sentiments_assigned": vader_sentiments_assigned,
    }
    return summary


def import_dataset(file_path: Path):
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return import_items(payload)


def main():
    parser = argparse.ArgumentParser(description="Import exported Apify Facebook dataset into MANA SQLite DB.")
    parser.add_argument("--file", required=True, help="Path to the exported JSON dataset file.")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Dataset file not found: {file_path}")

    summary = import_dataset(file_path)
    print(f"Imported dataset from {file_path}")
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
    if summary["corex_topics_assigned"]:
        print(f"CorEx topic rows assigned: {summary['corex_topics_assigned']}")
    else:
        print("CorEx: model not trained — skipped topic inference (run POST /api/admin/corex/train)")
    if summary["svm_clusters_assigned"]:
        print(f"SVM cluster rows assigned: {summary['svm_clusters_assigned']}")
    else:
        print("SVM: model not trained — skipped cluster inference (run POST /api/admin/svm/train)")
    if summary["vader_sentiments_assigned"] >= 0:
        print(f"VADER sentiment rows assigned: {summary['vader_sentiments_assigned']}")
    else:
        print("VADER: sentiment analysis failed — check logs.")


if __name__ == "__main__":
    main()
