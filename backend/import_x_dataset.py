"""
Import an exported Apify X dataset into the local database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app import app, ensure_database
from data import (
    CLUSTER_DEFINITIONS,
    PRIORITY_ORDER,
    extract_location,
    infer_cluster,
    infer_priority,
    infer_sentiment_score,
    recommendation_payload_for,
)
from models import Post, PostCluster, PostPriority, PostSentiment, PostTopic, PreprocessedText, db
from preprocessing import save_preprocessed_text
from services.corex.topic_modeler import is_model_trained, predict_topics_batch
from services.svm.cluster_classifier import (
    is_model_trained as is_svm_trained,
    predict_clusters_batch,
    select_top_cluster,
)


def safe_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


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


def infer_media_type(item: dict) -> str:
    media = item.get("media") or []
    if not media:
        return "text"
    if any((entry or {}).get("type") == "video" for entry in media if isinstance(entry, dict)):
        return "video"
    return "photo"


def normalize_item(item: dict):
    text = (item.get("text") or item.get("content") or "").strip()
    cluster, keywords = infer_cluster(text)
    if cluster is None:
        cluster = CLUSTER_DEFINITIONS[0]
        keywords = keywords or []

    likes = safe_int(item.get("likes"))
    reposts = safe_int(item.get("shares") or item.get("reposts"))
    comments = safe_int(item.get("comments"))
    views = safe_int(item.get("viewsCount"))
    engagement = likes + reposts + comments
    priority = infer_priority(text, engagement)
    sentiment_score = infer_sentiment_score(text, engagement)
    recommendation_payload = recommendation_payload_for(
        cluster["id"],
        priority,
        sentiment_score=sentiment_score,
        likes=likes,
        comments=comments,
        reposts=reposts,
        post_count=1,
    )

    post_id = str(item.get("postId") or item.get("url") or item.get("twitterUrl"))
    author_username = (item.get("authorUsername") or "").strip()
    author_name = (item.get("authorName") or "").strip()
    page_source = author_username or author_name or "X Source"

    return {
        "id": post_id,
        "source": "X",
        "page_source": page_source,
        "account_url": item.get("twitterUrl") or item.get("url"),
        "author": author_name or author_username or "X user",
        "caption": text,
        "source_url": item.get("url") or item.get("twitterUrl"),
        "external_id": str(item.get("postId") or ""),
        "reactions": 0,
        "shares": 0,
        "likes": likes,
        "reposts": reposts,
        "comments": comments,
        "views": views,
        "media_type": infer_media_type(item),
        "priority": priority,
        "sentiment_score": sentiment_score,
        "recommendation": recommendation_payload["recommendation"],
        "status": "Monitoring",
        "cluster_id": cluster["id"],
        "cluster_label_source": "heuristic",
        "date": parse_datetime(item.get("time")),
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

        corex_topics_assigned = 0
        if is_model_trained():
            new_post_ids = [item.get("postId") or item.get("url") or item.get("twitterUrl") for item in payload]
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

        svm_clusters_assigned = 0
        if is_svm_trained():
            new_post_ids = [item.get("postId") or item.get("url") or item.get("twitterUrl") for item in payload]
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
                    for item in cluster_list:
                        existing = PostCluster.query.filter_by(
                            post_id=row.raw_id, cluster_id=item["cluster_id"]
                        ).first()
                        if existing:
                            existing.confidence = item["confidence"]
                        else:
                            db.session.add(PostCluster(
                                post_id=row.raw_id,
                                cluster_id=item["cluster_id"],
                                confidence=item["confidence"],
                            ))
                        svm_clusters_assigned += 1
                    top = select_top_cluster(cluster_list)
                    post = db.session.get(Post, row.raw_id)
                    if post and top:
                        post.cluster_id = top["cluster_id"]
                        post.cluster_label_source = "svm"
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
            "corex_topics_assigned": corex_topics_assigned,
            "svm_clusters_assigned": svm_clusters_assigned,
        }
