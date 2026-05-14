"""
Rule-based recommendation routes.

The recommendation engine is deterministic and uses:
topic + sentiment + engagement score + predicted priority.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from data import recommendation_payload_for
from models import Post, db
from services.rules.decision_engine import evaluate, list_rules

rules_bp = Blueprint("rules", __name__)


@rules_bp.route("/rules/list", methods=["GET"])
@jwt_required()
def get_rules():
    rules = list_rules()
    return jsonify({"rules": rules, "total": len(rules)}), 200


@rules_bp.route("/rules/evaluate", methods=["POST"])
@jwt_required()
def evaluate_single():
    body = request.get_json(silent=True) or {}
    topic = body.get("topic") or body.get("cluster_id") or ""
    sentiment = body.get("sentiment", "Neutral")
    engagement_score = body.get("engagement_score", 0)
    priority = body.get("priority", "Medium")

    if not str(topic).strip():
        return jsonify({"error": "topic is required"}), 400

    return jsonify(evaluate(topic, sentiment, engagement_score, priority)), 200


@rules_bp.route("/rules/evaluate-all", methods=["POST"])
@jwt_required()
def evaluate_all():
    posts = Post.query.all()
    if not posts:
        return jsonify({"message": "No posts found", "evaluated": 0}), 200

    evaluated = 0
    errors = 0

    for post in posts:
        try:
            result = recommendation_payload_for(
                post.cluster_id,
                post.priority or "Moderate",
                sentiment_score=post.sentiment_score,
                reactions=post.reactions,
                likes=post.likes,
                comments=post.comments,
                shares=post.shares,
                reposts=post.reposts,
                post_count=1,
            )
            post.recommendation = result["recommendation"]
            evaluated += 1
        except Exception:
            errors += 1

    db.session.commit()

    return jsonify({
        "message": "Recommendations generated successfully",
        "evaluated": evaluated,
        "errors": errors,
    }), 200
