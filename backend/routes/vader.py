"""
MANA — VADER Sentiment Analysis Routes (admin only).

Endpoints:
  GET  /api/admin/vader/status               — library availability + DB stats
  POST /api/admin/vader/analyze-all          — run VADER on all posts, upsert sentiments table
  POST /api/admin/vader/analyze/<post_id>    — re-analyze a single post on demand
  GET  /api/admin/vader/sentiment/<post_id>  — get stored PostSentiment for one post
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt, jwt_required

from models import Comment, Post, PostSentiment, PreprocessedText, db
from services.vader.sentiment_analyzer import analyze_post_with_comments, get_status

vader_bp = Blueprint("vader", __name__)


def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "Admin":
            return jsonify({"message": "Admin access required."}), 403
        return fn(*args, **kwargs)
    return wrapper


# ── Status ─────────────────────────────────────────────────────────────────────

@vader_bp.route("/vader/status", methods=["GET"])
@admin_required
def status():
    """Library availability, configuration, and DB-level sentiment stats."""
    info = get_status()
    total_posts     = Post.query.count()
    analyzed        = PostSentiment.query.count()
    sarcasm_flagged = PostSentiment.query.filter_by(sarcasm_flag=True).count()
    info.update({
        "total_posts":     total_posts,
        "analyzed":        analyzed,
        "pending":         total_posts - analyzed,
        "sarcasm_flagged": sarcasm_flagged,
    })
    return jsonify(info)


# ── Analyze all ────────────────────────────────────────────────────────────────

@vader_bp.route("/vader/analyze-all", methods=["POST"])
@admin_required
def analyze_all():
    """
    Run VADER on every post.

    Optional JSON body: { "overwrite": true }
    When overwrite is false (default) posts that already have a PostSentiment row
    are skipped, making repeated calls safe and incremental.

    For each post:
    1. Use PreprocessedText.clean_text when available (translated + lemmatized),
       otherwise fall back to Post.caption.
    2. Upsert PostSentiment row.
    3. Update Post.sentiment_compound (raw float) and Post.sentiment_score (int 0-100).
    """
    body      = request.get_json(silent=True) or {}
    overwrite = bool(body.get("overwrite", False))

    posts = Post.query.all()
    if not posts:
        return jsonify({"message": "No posts found.", "processed": 0})

    post_ids = [p.id for p in posts]

    already_done: set[str] = set()
    if not overwrite:
        already_done = {
            ps.post_id
            for ps in PostSentiment.query.filter(PostSentiment.post_id.in_(post_ids)).all()
        }

    # Build vader_text lookup — casing-preserved, translated English for VADER.
    # Falls back to clean_text for rows preprocessed before vader_text was added.
    vader_text_map: dict[str, str] = {
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

    inserted = updated = skipped = 0

    for post in posts:
        if post.id in already_done:
            skipped += 1
            continue

        text   = vader_text_map.get(post.id) or post.caption or ""
        result = analyze_post_with_comments(
            text,
            post.cluster_id,
            comments_by_post.get(post.id, []),
        )

        existing = PostSentiment.query.filter_by(post_id=post.id).first()
        if existing:
            existing.compound     = result["compound"]
            existing.positive     = result["positive"]
            existing.negative     = result["negative"]
            existing.neutral      = result["neutral"]
            existing.sarcasm_flag = result["sarcasm_flag"]
            updated += 1
        else:
            db.session.add(PostSentiment(
                post_id=post.id,
                compound=result["compound"],
                positive=result["positive"],
                negative=result["negative"],
                neutral=result["neutral"],
                sarcasm_flag=result["sarcasm_flag"],
            ))
            inserted += 1

        post.sentiment_score    = result["sentiment_score"]
        post.sentiment_compound = result["compound"]

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Database commit failed: {exc}"}), 500

    return jsonify({
        "message":                    "VADER sentiment analysis complete.",
        "posts_processed":            inserted + updated,
        "sentiment_rows_inserted":    inserted,
        "sentiment_rows_updated":     updated,
        "posts_skipped_already_done": skipped,
    })


# ── Analyze single post ────────────────────────────────────────────────────────

@vader_bp.route("/vader/analyze/<post_id>", methods=["POST"])
@admin_required
def analyze_single(post_id: str):
    """Re-analyze a single post on demand (always overwrites existing sentiment)."""
    post = db.session.get(Post, post_id)
    if not post:
        return jsonify({"error": f"Post {post_id!r} not found."}), 404

    pt      = PreprocessedText.query.filter_by(raw_id=post_id, record_type="post").first()
    vader_src = (pt.vader_text or pt.clean_text) if pt else None
    text      = vader_src or post.caption or ""

    comment_texts = [
        comment.text or ""
        for comment in Comment.query.filter_by(post_id=post_id).all()
    ]
    result   = analyze_post_with_comments(text, post.cluster_id, comment_texts)
    existing = PostSentiment.query.filter_by(post_id=post_id).first()

    if existing:
        existing.compound     = result["compound"]
        existing.positive     = result["positive"]
        existing.negative     = result["negative"]
        existing.neutral      = result["neutral"]
        existing.sarcasm_flag = result["sarcasm_flag"]
    else:
        db.session.add(PostSentiment(
            post_id=post_id,
            compound=result["compound"],
            positive=result["positive"],
            negative=result["negative"],
            neutral=result["neutral"],
            sarcasm_flag=result["sarcasm_flag"],
        ))

    post.sentiment_score    = result["sentiment_score"]
    post.sentiment_compound = result["compound"]

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Database commit failed: {exc}"}), 500

    return jsonify({
        "message":   "Post analyzed.",
        "post_id":   post_id,
        "sentiment": result,
    })


# ── Get sentiment for one post ─────────────────────────────────────────────────

@vader_bp.route("/vader/sentiment/<post_id>", methods=["GET"])
@admin_required
def get_post_sentiment(post_id: str):
    """Return the stored PostSentiment row for a single post."""
    ps = PostSentiment.query.filter_by(post_id=post_id).first()
    if not ps:
        return jsonify({"error": f"No sentiment record for post {post_id!r}."}), 404
    return jsonify({
        "post_id":   post_id,
        "sentiment": ps.to_api_dict(),
    })
