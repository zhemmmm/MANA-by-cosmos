"""
MANA — Full ML Pipeline Orchestration Route (admin only).

Endpoints:
  POST /api/admin/pipeline/run-all   — run all ML stages in sequence
  GET  /api/admin/pipeline/status    — combined status of all three models
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt, jwt_required

from data import recommendation_payload_for
from models import Comment, Post, PostCluster, PostPriority, PostSentiment, PostTopic, PreprocessedText, db
from services.corex.topic_modeler import (
    get_model_status as corex_status,
    is_model_trained as corex_trained,
    predict_topics_batch,
    train_corex,
)
from services.svm.cluster_classifier import (
    get_model_status as svm_status,
    is_model_trained as svm_trained,
    predict_clusters_batch,
    select_top_cluster,
    train_svm,
)
from services.vader.sentiment_analyzer import analyze_post_with_comments, get_status as vader_status
from services.random_forest.priority_classifier import (
    SEVERITY_MAP,
    get_model_status as rf_status,
    is_model_trained as rf_trained,
    predict_priorities_batch,
    train_rf,
)

# Map RF labels to the priority string recommendation logic acts on
_RF_TO_REC_PRIORITY = {"High": "Critical", "Medium": "Moderate", "Low": "Monitoring"}

pipeline_bp = Blueprint("pipeline", __name__)


def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "Admin":
            return jsonify({"message": "Admin access required."}), 403
        return fn(*args, **kwargs)
    return wrapper


# ── Run full pipeline ──────────────────────────────────────────────────────────

@pipeline_bp.route("/pipeline/run-all", methods=["POST"])
@admin_required
def run_all():
    """
    Run the full ML pipeline in order:
      1. Check preprocessed posts exist
      2. CorEx train  (skipped if already trained and force_retrain is false)
      3. CorEx predict-all  → post_topics
      4. SVM train    (skipped if already trained and force_retrain is false)
      5. SVM predict-all    → post_clusters
      6. VADER analyze-all  → sentiments

    Optional JSON body: { "force_retrain": true }
    """
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force_retrain", False))

    steps: dict = {}

    # ── Step 1: Check preprocessed posts ──────────────────────────────────────
    rows = (
        PreprocessedText.query
        .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
        .filter(PreprocessedText.final_tokens_json != "[]")
        .all()
    )
    if not rows:
        return jsonify({
            "error": "No preprocessed posts found. Import and preprocess data first."
        }), 400

    post_ids = [row.raw_id for row in rows]
    texts = [" ".join(row.final_tokens) for row in rows]
    steps["preprocessing"] = {"posts_available": len(rows)}

    # ── Step 2: CorEx train ────────────────────────────────────────────────────
    if corex_trained() and not force:
        meta = corex_status()
        steps["corex_train"] = {
            "skipped": True,
            "reason": "Model already trained. Pass force_retrain=true to retrain.",
            "trained_at": meta.get("trained_at"),
            "overall_coherence": meta.get("overall_coherence"),
        }
    else:
        try:
            result = train_corex(texts)
            steps["corex_train"] = {
                "skipped": False,
                "corpus_size": result["corpus_size"],
                "trained_at": result["trained_at"],
                "overall_coherence": result["overall_coherence"],
                "low_coherence_topics": result["low_coherence_topics"],
            }
        except Exception as exc:
            return jsonify({"error": f"CorEx training failed: {exc}", "steps": steps}), 500

    # ── Step 3: CorEx predict-all ──────────────────────────────────────────────
    try:
        batch_topics = predict_topics_batch(texts)
        topic_inserted = topic_skipped = 0
        for post_id, topic_list in zip(post_ids, batch_topics):
            if not topic_list:
                topic_skipped += 1
                continue
            PostTopic.query.filter_by(post_id=post_id).delete()
            for item in topic_list:
                db.session.add(PostTopic(
                    post_id=post_id,
                    topic_label=item["topic"],
                    confidence=item["confidence"],
                ))
                topic_inserted += 1
        db.session.flush()
        steps["corex_predict"] = {
            "posts_processed": len(post_ids) - topic_skipped,
            "topic_rows_inserted": topic_inserted,
            "posts_skipped_no_topics": topic_skipped,
        }
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"CorEx prediction failed: {exc}", "steps": steps}), 500

    # ── Step 2.5: Re-score heuristic cluster labels using CorEx expanded keywords ─
    # Loads the keywords CorEx just discovered and re-assigns cluster_id for posts
    # where the label came from the heuristic (not from a human reviewer).
    # Human-reviewed labels (cluster_label_source == "reviewed") are never touched.
    # Failure is non-fatal — SVM training continues with existing labels.
    posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}
    try:
        from data import TOPIC_TO_CLUSTER, load_corex_expanded_keywords
        corex_kw = load_corex_expanded_keywords()
        corex_relabeled = 0
        if corex_kw:
            cluster_keyword_sets: dict[str, set[str]] = {
                TOPIC_TO_CLUSTER[topic]: {w.lower() for w in words}
                for topic, words in corex_kw.items()
                if topic in TOPIC_TO_CLUSTER
            }
            # Strong-signal terms — when present, force the post to that cluster
            # (these are unambiguous in the disaster-response domain).
            STRONG_CLUSTER_SIGNALS = {
                "cluster-g": {  # rescue — fire and active rescue language
                    "fire", "fire alert", "txtfire", "bfp", "firefighter", "blaze",
                    "burning", "arson", "fire truck", "fire department",
                    "structure fire", "wildfire", "rescue", "trapped", "stranded",
                    "sos", "rescue boat", "search and rescue",
                },
                "cluster-h": {  # dead/missing — fatality language
                    "fatality", "casualty", "confirmed dead", "body found",
                    "death toll", "missing person", "remains identified",
                },
                "cluster-c": {  # evacuation
                    "evacuation center", "evacuees", "displaced families",
                },
            }

            for i, pid in enumerate(post_ids):
                post = posts_map.get(pid)
                if not post or post.cluster_label_source == "reviewed":
                    continue
                tokens_set = set(rows[i].final_tokens)
                raw_text = (post.caption or "").lower()

                forced_cluster = None
                for cid, signals in STRONG_CLUSTER_SIGNALS.items():
                    for term in signals:
                        if " " in term:
                            if term in raw_text:
                                forced_cluster = cid
                                break
                        elif term in tokens_set:
                            forced_cluster = cid
                            break
                    if forced_cluster:
                        break

                if forced_cluster:
                    if post.cluster_id != forced_cluster:
                        post.cluster_id = forced_cluster
                        post.cluster_label_source = "corex_enriched"
                        corex_relabeled += 1
                    continue

                scores = {
                    cluster_id: len(tokens_set & kw_set)
                    for cluster_id, kw_set in cluster_keyword_sets.items()
                }
                if not scores:
                    continue
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_cluster_id, best_score = sorted_scores[0]
                runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                if (
                    best_score > 0
                    and (best_score - runner_up_score) >= 2
                    and post.cluster_id != best_cluster_id
                ):
                    post.cluster_id = best_cluster_id
                    post.cluster_label_source = "corex_enriched"
                    corex_relabeled += 1
        steps["corex_relabel"] = {
            "posts_relabeled": corex_relabeled,
            "topics_available": len(corex_kw),
        }
    except Exception as exc:
        steps["corex_relabel"] = {"error": str(exc), "posts_relabeled": 0}

    # ── Step 4: SVM train ──────────────────────────────────────────────────────
    training_pairs = []
    reviewed_count = heuristic_count = 0
    for i, pid in enumerate(post_ids):
        post = posts_map.get(pid)
        if not post:
            continue
        label = post.reviewed_cluster_id or post.cluster_id
        source = "reviewed" if post.reviewed_cluster_id else (post.cluster_label_source or "heuristic")
        if source == "reviewed":
            reviewed_count += 1
        else:
            heuristic_count += 1
        training_pairs.append((texts[i], [label]))
    paired_texts = [text for text, _label in training_pairs]
    labels = [label for _text, label in training_pairs]

    if svm_trained() and not force:
        meta = svm_status()
        steps["svm_train"] = {
            "skipped": True,
            "reason": "Model already trained. Pass force_retrain=true to retrain.",
            "trained_at": meta.get("trained_at"),
            "f1_macro": meta.get("f1_macro"),
        }
    else:
        try:
            result = train_svm(paired_texts, labels)
            steps["svm_train"] = {
                "skipped": False,
                "corpus_size": result["corpus_size"],
                "best_C": result["best_C"],
                "f1_macro": result["f1_macro"],
                "trained_at": result["trained_at"],
                "reviewed_labels_used": reviewed_count,
                "bootstrap_labels_used": heuristic_count,
            }
        except Exception as exc:
            return jsonify({"error": f"SVM training failed: {exc}", "steps": steps}), 500

    # ── Step 5: SVM predict-all ────────────────────────────────────────────────
    try:
        batch_clusters = predict_clusters_batch(texts)
        cluster_inserted = cluster_skipped = cluster_updates = 0
        for post_id, cluster_list in zip(post_ids, batch_clusters):
            if not cluster_list:
                cluster_skipped += 1
                continue
            PostCluster.query.filter_by(post_id=post_id).delete()
            for item in cluster_list:
                db.session.add(PostCluster(
                    post_id=post_id,
                    cluster_id=item["cluster_id"],
                    confidence=item["confidence"],
                ))
                cluster_inserted += 1
            top_cluster = select_top_cluster(cluster_list)
            post = posts_map.get(post_id)
            if post and top_cluster and post.cluster_id != top_cluster["cluster_id"]:
                # Never overwrite strong-signal CorEx decisions or human-reviewed labels.
                if post.cluster_label_source in ("reviewed", "corex_enriched"):
                    continue
                post.cluster_id = top_cluster["cluster_id"]
                post.cluster_label_source = "svm"
                cluster_updates += 1
        db.session.flush()
        steps["svm_predict"] = {
            "posts_processed": len(post_ids) - cluster_skipped,
            "cluster_rows_inserted": cluster_inserted,
            "post_cluster_id_updated": cluster_updates,
            "posts_skipped_no_clusters": cluster_skipped,
        }
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"SVM prediction failed: {exc}", "steps": steps}), 500

    # ── Step 6: VADER analyze-all ──────────────────────────────────────────────
    try:
        vader_text_map = {
            row.raw_id: (row.vader_text or row.clean_text)
            for row in PreprocessedText.query
                .filter(PreprocessedText.raw_id.in_(post_ids))
                .filter_by(record_type="post")
                .all()
            if (row.vader_text or row.clean_text)
        }
        comments_by_post: dict[str, list[str]] = {}
        for comment in Comment.query.filter(Comment.post_id.in_(post_ids)).all():
            comments_by_post.setdefault(comment.post_id, []).append(comment.text or "")
        vader_inserted = vader_updated = 0
        for post_id in post_ids:
            post = posts_map.get(post_id)
            if not post:
                continue
            text = vader_text_map.get(post_id) or post.caption or ""
            result = analyze_post_with_comments(
                text,
                post.cluster_id,
                comments_by_post.get(post_id, []),
            )
            existing = PostSentiment.query.filter_by(post_id=post_id).first()
            if existing:
                existing.compound = result["compound"]
                existing.positive = result["positive"]
                existing.negative = result["negative"]
                existing.neutral = result["neutral"]
                existing.sarcasm_flag = result["sarcasm_flag"]
                vader_updated += 1
            else:
                db.session.add(PostSentiment(
                    post_id=post_id,
                    compound=result["compound"],
                    positive=result["positive"],
                    negative=result["negative"],
                    neutral=result["neutral"],
                    sarcasm_flag=result["sarcasm_flag"],
                ))
                vader_inserted += 1
            post.sentiment_score = result["sentiment_score"]
            post.sentiment_compound = result["compound"]
        db.session.flush()
        steps["vader"] = {
            "posts_processed": vader_inserted + vader_updated,
            "sentiment_rows_inserted": vader_inserted,
            "sentiment_rows_updated": vader_updated,
        }
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"VADER analysis failed: {exc}", "steps": steps}), 500

    # ── Step 7: RF train ───────────────────────────────────────────────────────
    if rf_trained() and not force:
        meta = rf_status()
        steps["rf_train"] = {
            "skipped":    True,
            "reason":     "Model already trained. Pass force_retrain=true to retrain.",
            "trained_at": meta.get("trained_at"),
            "accuracy":   meta.get("accuracy"),
        }
    else:
        try:
            result = train_rf(post_ids)
            steps["rf_train"] = {
                "skipped":            False,
                "corpus_size":        result["corpus_size"],
                "accuracy":           result["accuracy"],
                "trained_at":         result["trained_at"],
                "class_distribution": result["class_distribution"],
            }
        except Exception as exc:
            return jsonify({"error": f"RF training failed: {exc}", "steps": steps}), 500

    # ── Step 8: RF predict-all ─────────────────────────────────────────────────
    try:
        rf_predictions = predict_priorities_batch(post_ids)
        rf_inserted = rf_updated = 0
        for pred in rf_predictions:
            pid   = pred["post_id"]
            label = pred["priority"]
            conf  = pred["confidence"]
            probs = pred["probabilities"]
            post  = posts_map.get(pid)
            if not post:
                continue
            post.priority      = label
            post.severity_rank = SEVERITY_MAP.get(label, 2)
            post.recommendation = recommendation_payload_for(
                post.cluster_id,
                _RF_TO_REC_PRIORITY.get(label, "Moderate"),
                sentiment_score=post.sentiment_score,
                reactions=post.reactions,
                likes=post.likes,
                comments=post.comments,
                shares=post.shares,
                reposts=post.reposts,
                post_count=1,
            )["recommendation"]
            existing = PostPriority.query.filter_by(post_id=pid).first()
            if existing:
                existing.priority_label     = label
                existing.confidence         = conf
                existing.high_probability   = probs.get("High",   0.0)
                existing.medium_probability = probs.get("Medium", 0.0)
                existing.low_probability    = probs.get("Low",    0.0)
                rf_updated += 1
            else:
                db.session.add(PostPriority(
                    post_id=pid,
                    priority_label=label,
                    confidence=conf,
                    high_probability=probs.get("High",   0.0),
                    medium_probability=probs.get("Medium", 0.0),
                    low_probability=probs.get("Low",    0.0),
                ))
                rf_inserted += 1
        db.session.flush()
        steps["rf_predict"] = {
            "posts_processed":        len(rf_predictions),
            "priority_rows_inserted": rf_inserted,
            "priority_rows_updated":  rf_updated,
        }
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"RF prediction failed: {exc}", "steps": steps}), 500

    # ── Commit everything at once ──────────────────────────────────────────────
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Database commit failed: {exc}", "steps": steps}), 500

    return jsonify({"message": "Pipeline complete.", "steps": steps})


# ── Train from seed dataset ────────────────────────────────────────────────────

@pipeline_bp.route("/pipeline/train-from-seed", methods=["POST"])
@admin_required
def train_from_seed():
    """
    Train CorEx + SVM from seed_dataset.json, then re-classify all real DB posts.

    The seed posts are never written to the database — training only.
    Call this once after a fresh deploy to restore accurate categorization.

    Returns: { f1_macro, corex_coherence, posts_reclassified }
    """
    import json
    from pathlib import Path
    from preprocessing import preprocess_record

    seed_path = Path(__file__).parent.parent / "seed_dataset.json"
    if not seed_path.exists():
        return jsonify({"error": "seed_dataset.json not found. Ensure it is committed to the repo."}), 404

    with seed_path.open(encoding="utf-8") as f:
        seed_posts = json.load(f)

    raw_texts = [(p.get("text") or p.get("caption") or "").strip() for p in seed_posts]
    seed_labels = [[p["_seed_cluster_id"]] for p in seed_posts]

    # Preprocess seed texts using the same pipeline as real posts.
    processed_texts = []
    for i, text in enumerate(raw_texts):
        try:
            result = preprocess_record(raw_id=f"seed_{i:04d}", item={"text": text}, record_type="post")
            final = result.get("final_tokens") or []
            processed_texts.append(" ".join(final) if final else text.lower())
        except Exception:
            processed_texts.append(text.lower())

    # Train CorEx
    try:
        corex_result = train_corex(processed_texts)
    except Exception as exc:
        return jsonify({"error": f"CorEx training failed: {exc}"}), 500

    # Train SVM
    try:
        svm_result = train_svm(processed_texts, seed_labels)
    except Exception as exc:
        return jsonify({"error": f"SVM training failed: {exc}"}), 500

    # Re-classify all real DB posts immediately.
    from services.classification.refine import refine_labels
    all_post_ids = [
        row.raw_id for row in (
            PreprocessedText.query
            .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
            .filter(PreprocessedText.final_tokens_json != "[]")
            .all()
        )
    ]
    refine_metrics = refine_labels(all_post_ids)
    db.session.commit()

    return jsonify({
        "message": "Training complete. All DB posts re-classified.",
        "f1_macro": svm_result["f1_macro"],
        "corpus_size": svm_result["corpus_size"],
        "corex_coherence": corex_result.get("overall_coherence"),
        "posts_reclassified": refine_metrics.get("posts_processed", 0),
        "corex_relabeled": refine_metrics.get("corex_relabeled", 0),
        "svm_updated": refine_metrics.get("post_cluster_id_updated", 0),
    })


# ── Combined model status ──────────────────────────────────────────────────────

@pipeline_bp.route("/pipeline/status", methods=["GET"])
@admin_required
def status():
    """Return training status for all three models in one call."""
    return jsonify({
        "corex": corex_status(),
        "svm": svm_status(),
        "vader": vader_status(),
        "rf": rf_status(),
        "preprocessing": {
            "total_posts": PreprocessedText.query.filter_by(record_type="post").count(),
            "relevant_posts": PreprocessedText.query.filter_by(
                record_type="post", preprocessing_status="processed", is_relevant=True
            ).count(),
        },
    })
