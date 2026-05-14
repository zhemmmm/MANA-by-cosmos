"""
MANA — Random Forest Priority Classification Route (admin only).

Endpoints:
  POST /api/admin/rf/train        — train from existing labeled posts
  POST /api/admin/rf/predict-all  — classify all posts, update Post.priority + PostPriority
  GET  /api/admin/rf/status       — model metadata and training state
  rff
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt, jwt_required

from data import recommendation_payload_for
from models import Post, PostPriority, PreprocessedText, db
from services.random_forest.priority_classifier import (
    SEVERITY_MAP,
    get_model_status,
    is_model_trained,
    predict_priorities_batch,
    train_rf,
)

rf_bp = Blueprint("rf", __name__)

# Map RF output back to the priority string that the recommendation logic acts on
_RF_TO_REC_PRIORITY = {"High": "Critical", "Medium": "Moderate", "Low": "Monitoring"}


def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "Admin":
            return jsonify({"message": "Admin access required."}), 403
        return fn(*args, **kwargs)
    return wrapper


@rf_bp.route("/rf/train", methods=["POST"])
@admin_required
def train():
    """
    Train the RF classifier using Post.priority as bootstrap labels.
    Accepts all preprocessed, relevant posts from the database.
    """
    try:
        result = train_rf()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"RF training failed: {exc}"}), 500

    return jsonify({
        "message":            "RF model trained successfully.",
        "trained_at":         result["trained_at"],
        "corpus_size":        result["corpus_size"],
        "accuracy":           result["accuracy"],
        "class_distribution": result["class_distribution"],
    })


@rf_bp.route("/rf/predict-all", methods=["POST"])
@admin_required
def predict_all():
    """
    Run RF priority classification on all eligible posts and persist results:
      - Post.priority        ← "High" | "Medium" | "Low"
      - Post.severity_rank   ← 3 | 2 | 1
      - Post.recommendation  ← from data.recommendation_for()
      - PostPriority row     ← label + per-class probabilities (upserted)
    """
    if not is_model_trained():
        return jsonify({"error": "RF model not trained. Run /rf/train first."}), 400

    rows = (
        PreprocessedText.query
        .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
        .filter(PreprocessedText.final_tokens_json != "[]")
        .all()
    )
    post_ids = [r.raw_id for r in rows]
    if not post_ids:
        return jsonify({"error": "No preprocessed posts available."}), 400

    try:
        predictions = predict_priorities_batch(post_ids)
    except Exception as exc:
        return jsonify({"error": f"RF prediction failed: {exc}"}), 500

    posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}

    inserted = updated = 0
    for pred in predictions:
        pid   = pred["post_id"]
        label = pred["priority"]
        conf  = pred["confidence"]
        probs = pred["probabilities"]

        post = posts_map.get(pid)
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
            updated += 1
        else:
            db.session.add(PostPriority(
                post_id=pid,
                priority_label=label,
                confidence=conf,
                high_probability=probs.get("High",   0.0),
                medium_probability=probs.get("Medium", 0.0),
                low_probability=probs.get("Low",    0.0),
            ))
            inserted += 1

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Database commit failed: {exc}"}), 500

    return jsonify({
        "message":                "RF priority classification complete.",
        "posts_processed":        inserted + updated,
        "priority_rows_inserted": inserted,
        "priority_rows_updated":  updated,
    })


@rf_bp.route("/rf/status", methods=["GET"])
@admin_required
def status():
    return jsonify(get_model_status())

