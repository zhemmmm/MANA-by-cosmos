"""
MANA — Posts, watchlist, and dashboard routes backed by SQLite.
"""

from __future__ import annotations

from datetime import timedelta
from collections import defaultdict

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from data import CLUSTER_DEFINITIONS, TOPIC_TO_CLUSTER, now_utc, parse_date_range, top_keywords_from_posts
from facebook_matching import build_post_match_index, find_post_match
from models import Comment, Post, PostTopic, Watchlist, db
from services.corex.topic_modeler import MIN_CLUSTER_CONFIDENCE

posts_bp = Blueprint("posts", __name__)

COMMENT_SIGNAL_TERMS = {
    "urgent": 10,
    "sos": 12,
    "rescue": 12,
    "help": 8,
    "init": 6,
    "heat": 6,
    "tubig": 6,
    "water": 6,
    "suspend": 7,
    "suspension": 7,
    "class": 5,
    "classes": 5,
    "newborn": 8,
    "bata": 7,
    "senior": 7,
    "hospital": 8,
}


def current_username():
    return get_jwt_identity() or "admin_mana"


def comment_rank(comment: Comment):
    text = (comment.text or "").lower()
    signal_score = sum(weight for term, weight in COMMENT_SIGNAL_TERMS.items() if term in text)
    severity = comment.post.severity_rank if comment.post else 1
    return signal_score + (comment.likes * 4) + (severity * 12) + min(len((comment.text or "").split()), 12)


def apply_post_filters(query):
    source = request.args.get("source")
    cluster_id = request.args.get("cluster_id")
    priority = request.args.get("priority")
    date_range = request.args.get("date_range")
    include_irrelevant = (request.args.get("include_irrelevant") or "").lower() in {"1", "true", "yes"}

    if not include_irrelevant:
        query = query.filter(Post.is_relevant == True)

    if source:
        query = query.filter(Post.source == source)
    if cluster_id:
        topics_for_cluster = [
            topic for topic, cid in TOPIC_TO_CLUSTER.items() if cid == cluster_id
        ]
        if topics_for_cluster:
            qualifying = (
                db.session.query(PostTopic.post_id)
                .filter(
                    PostTopic.topic_label.in_(topics_for_cluster),
                    PostTopic.confidence >= MIN_CLUSTER_CONFIDENCE,
                )
                .distinct()
                .subquery()
            )
            query = query.filter(Post.cluster_id == cluster_id, Post.id.in_(qualifying))
        else:
            query = query.filter(Post.cluster_id == cluster_id)
    if priority:
        mapped = "Moderate" if priority == "Medium" else ("Monitoring" if priority == "Low" else priority)
        query = query.filter(Post.priority == mapped)
    if date_range:
        cutoff = now_utc() - parse_date_range(date_range)
        query = query.filter(Post.date >= cutoff)
    return query


@posts_bp.route("/posts", methods=["GET"])
@jwt_required(optional=True)
def get_posts():
    posts = (
        apply_post_filters(Post.query)
        .order_by(Post.date.desc())
        .all()
    )
    post_ids = [post.id for post in posts]
    top_comments_by_post_id = defaultdict(list)

    if post_ids:
        linked_comments = (
            Comment.query
            .filter(Comment.post_id.in_(post_ids))
            .order_by(Comment.date.desc())
            .all()
        )
        orphan_facebook_comments = (
            Comment.query
            .filter(Comment.post_id.is_(None), Comment.source == "Facebook")
            .order_by(Comment.date.desc())
            .all()
        )
        post_lookup = build_post_match_index(posts)
        comments = list(linked_comments)

        for comment in orphan_facebook_comments:
            matched_post = find_post_match(post_lookup, url=comment.post_url)
            if matched_post:
                comment.post = matched_post
                comments.append(comment)

        ranked_comments = defaultdict(list)
        for comment in comments:
            target_post_id = comment.post_id or getattr(comment.post, "id", None)
            if target_post_id:
                ranked_comments[target_post_id].append(comment)

        for post_id, post_comments in ranked_comments.items():
            top_comments_by_post_id[post_id] = [
                {
                    "id": comment.id,
                    "author": comment.author,
                    "text": comment.text,
                    "likes": comment.likes,
                    "date": comment.date.isoformat() if comment.date else None,
                }
                for comment in sorted(post_comments, key=comment_rank, reverse=True)[:3]
            ]

    return jsonify([
        post.to_api_dict(top_comments=top_comments_by_post_id.get(post.id, []))
        for post in posts
    ])


@posts_bp.route("/posts/<post_id>/status", methods=["PATCH"])
@jwt_required(optional=True)
def update_post_status(post_id):
    data = request.get_json() or {}
    post = db.session.get(Post, post_id)
    if not post:
        return jsonify({"message": "Post not found"}), 404

    post.status = data.get("status", post.status)
    db.session.commit()
    return jsonify({"id": post_id, "status": post.status})


@posts_bp.route("/watchlist", methods=["GET"])
@jwt_required(optional=True)
def get_watchlist():
    pinned = (
        Watchlist.query.filter_by(username=current_username())
        .order_by(Watchlist.created_at.desc())
        .all()
    )
    return jsonify({"pinned": [item.post_id for item in pinned]})


@posts_bp.route("/watchlist/<post_id>", methods=["POST"])
@jwt_required(optional=True)
def pin_post(post_id):
    if not db.session.get(Post, post_id):
        return jsonify({"message": "Post not found"}), 404

    username = current_username()
    existing = Watchlist.query.filter_by(username=username, post_id=post_id).first()
    if not existing:
        db.session.add(Watchlist(username=username, post_id=post_id))
        db.session.commit()
    return get_watchlist()


@posts_bp.route("/watchlist/<post_id>", methods=["DELETE"])
@jwt_required(optional=True)
def unpin_post(post_id):
    username = current_username()
    Watchlist.query.filter_by(username=username, post_id=post_id).delete()
    db.session.commit()
    return get_watchlist()


@posts_bp.route("/clusters", methods=["GET"])
@jwt_required(optional=True)
def get_clusters():
    return jsonify(CLUSTER_DEFINITIONS)


@posts_bp.route("/dashboard/summary", methods=["GET"])
@jwt_required(optional=True)
def get_dashboard_summary():
    date_range = request.args.get("date_range", "7d")
    cutoff = now_utc() - parse_date_range(date_range)
    posts = Post.query.filter(Post.is_relevant == True, Post.date >= cutoff).all()
    total = len(posts)
    fb_posts = sum(1 for post in posts if post.source == "Facebook")
    x_posts = sum(1 for post in posts if post.source == "X")
    high_priority = sum(1 for post in posts if post.priority in {"Critical", "High"})
    active_clusters = len({post.cluster_id for post in posts})
    cluster_count = max(len(CLUSTER_DEFINITIONS), 1)
    total_reactions = sum(post.reactions for post in posts)
    total_likes = sum(post.likes for post in posts)
    total_shares = sum(post.shares for post in posts)
    total_comments = sum(post.comments for post in posts)
    total_reposts = sum(post.reposts for post in posts)

    def pct(count, ceiling):
        return round((count / ceiling) * 100) if ceiling else 0

    label_map = {"24h": "Last 24 hours", "7d": "Last 7 days", "14d": "Last 14 days", "30d": "Last 30 days"}
    meta = label_map.get(date_range, "Recent")
    kpis = [
        {"label": "High Priority Count", "value": f"{high_priority:,}", "meta": meta, "bar": pct(high_priority, total)},
        {"label": "Total Posts Analyzed", "value": f"{total:,}", "meta": meta, "bar": pct(total, max(total, 1))},
        {"label": "Total Facebook Posts", "value": f"{fb_posts:,}", "meta": meta, "bar": pct(fb_posts, total)},
        {"label": "Total X/Twitter Posts", "value": f"{x_posts:,}", "meta": meta, "bar": pct(x_posts, total)},
        {
            "label": "Active Clusters",
            "value": f"{active_clusters:,}",
            "meta": f"{active_clusters} of {cluster_count} clusters",
            "bar": pct(active_clusters, cluster_count),
        },
    ]
    return jsonify(
        {
            "kpis": kpis,
            "totals": {
                "highPriorityCount": high_priority,
                "totalPostsAnalyzed": total,
                "totalFacebookPosts": fb_posts,
                "totalXPosts": x_posts,
                "activeClusters": active_clusters,
                "clusterCapacity": cluster_count,
                "reactions": total_reactions,
                "likes": total_likes,
                "shares": total_shares,
                "comments": total_comments,
                "reposts": total_reposts,
            },
        }
    )


@posts_bp.route("/dashboard/keywords", methods=["GET"])
@jwt_required(optional=True)
def get_keywords():
    posts = Post.query.filter(Post.is_relevant == True).order_by(Post.date.desc()).limit(500).all()
    return jsonify({"keywords": top_keywords_from_posts(posts)})


@posts_bp.route("/dashboard/comments", methods=["GET"])
@jwt_required(optional=True)
def get_dashboard_comments():
    date_range = request.args.get("date_range", "7d")
    limit = max(1, min(int(request.args.get("limit", 6)), 24))
    cutoff = now_utc() - parse_date_range(date_range)

    comments = (
        Comment.query.filter(Comment.date >= cutoff)
        .order_by(Comment.date.desc())
        .all()
    )

    ranked = sorted(comments, key=lambda comment: (comment_rank(comment), comment.likes, comment.date), reverse=True)
    return jsonify({"comments": [comment.to_api_dict() for comment in ranked[:limit]]})


@posts_bp.route("/settings/email-alerts", methods=["PATCH"])
@jwt_required(optional=True)
def update_email_alerts():
    data = request.get_json() or {}
    enabled = bool(data.get("enabled", True))
    return jsonify({"enabled": enabled})
