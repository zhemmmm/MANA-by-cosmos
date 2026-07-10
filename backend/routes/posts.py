"""
MANA — Posts, watchlist, and dashboard routes backed by SQLite.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func
from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from data import CLUSTER_DEFINITIONS, TOPIC_TO_CLUSTER, date_range_cutoff, date_range_label, score_tone, top_keywords_from_posts
from facebook_matching import build_post_match_index, find_post_match
from models import ActivityLog, Comment, Post, PostTopic, PreprocessedText, SystemSetting, User, Watchlist, db, utc_iso
from services.corex.topic_modeler import MIN_CLUSTER_CONFIDENCE
from services.vader.sentiment_analyzer import analyze_sentiment, compound_to_score
from x_matching import build_post_match_index as build_x_post_match_index, find_post_match as find_x_post_match

posts_bp = Blueprint("posts", __name__)


def _log_post_activity(action: str, detail: str):
    username = get_jwt_identity()
    user = db.session.get(User, username) if username else None
    db.session.add(
        ActivityLog(
            actor_username=user.username if user else None,
            actor_name=(user.name or user.username) if user else "Unknown",
            action=action,
            detail=detail,
            type="edit",
        )
    )


def _log_post_activity_for_post(action: str, detail: str, post: Post):
    username = get_jwt_identity()
    user = db.session.get(User, username) if username else None
    db.session.add(
        ActivityLog(
            actor_username=user.username if user else None,
            actor_name=(user.name or user.username) if user else "Unknown",
            action=action,
            detail=detail,
            type="edit",
            target_post_id=post.id,
            target_post_title=(post.caption or post.page_source or "")[:255],
            target_post_url=post.source_url,
        )
    )

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


def current_actor_label():
    username = get_jwt_identity()
    if not username:
        return "Unknown"
    user = db.session.get(User, username)
    if not user:
        return username
    return user.name or user.username


def comment_rank(comment: Comment):
    text = (comment.text or "").lower()
    signal_score = sum(weight for term, weight in COMMENT_SIGNAL_TERMS.items() if term in text)
    severity = comment.post.severity_rank if comment.post else 1
    return signal_score + (comment.likes * 4) + (severity * 12) + min(len((comment.text or "").split()), 12)


def comment_impact_level(comment: Comment):
    score = comment_rank(comment)
    if score >= 70:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _latest_timestamp(*values):
    return max([value for value in values if value], default=None)


def _comment_relevance_map(comments: list[Comment]):
    comment_ids = [comment.id for comment in comments if comment.id]
    if not comment_ids:
        return {}
    rows = (
        PreprocessedText.query
        .filter(PreprocessedText.record_type == "comment")
        .filter(PreprocessedText.raw_id.in_(comment_ids))
        .all()
    )
    return {row.raw_id: row.is_relevant for row in rows}


def _build_comment_view(comment: Comment, post_tone: str, is_relevant: bool = True):
    if not is_relevant:
        return {
            "id": comment.id,
            "author": comment.author,
            "text": comment.text,
            "likes": comment.likes,
            "date": utc_iso(comment.date),
            "isRelevant": False,
            "relevanceLabel": "Irrelevant",
            "impactLevel": None,
            "impactLabel": None,
            "impactScore": 0,
            "sentimentTone": None,
            "sentimentLabel": None,
            "sentimentScore": None,
            "signalLabel": "Irrelevant",
            "signalClass": "tone-irrelevant",
            "matchesPostTone": False,
        }

    analysis = analyze_sentiment(comment.text or "")
    comment_tone = analysis["label"].lower()
    impact_level = comment_impact_level(comment)
    if comment_tone == "neutral":
        signal_label = "Neutral signal"
    elif comment_tone == "negative":
        signal_label = "Negative signal"
    else:
        signal_label = "Positive signal"

    return {
        "id": comment.id,
        "author": comment.author,
        "text": comment.text,
        "likes": comment.likes,
        "date": utc_iso(comment.date),
        "isRelevant": True,
        "relevanceLabel": "Relevant",
        "impactLevel": impact_level,
        "impactLabel": impact_level,
        "impactScore": comment_rank(comment),
        "sentimentTone": comment_tone,
        "sentimentLabel": analysis["label"],
        "sentimentScore": compound_to_score(analysis["compound"]),
        "signalLabel": signal_label,
        "signalClass": f"tone-{comment_tone}",
        "matchesPostTone": comment_tone == post_tone,
    }


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
        cutoff = date_range_cutoff(date_range)
        if cutoff is not None:
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
        orphan_x_comments = (
            Comment.query
            .filter(Comment.post_id.is_(None), Comment.source == "X")
            .order_by(Comment.date.desc())
            .all()
        )
        fb_posts = [post for post in posts if post.source == "Facebook"]
        x_posts = [post for post in posts if post.source == "X"]
        fb_post_lookup = build_post_match_index(fb_posts)
        x_post_lookup = build_x_post_match_index(x_posts)
        comments = list(linked_comments)

        for comment in orphan_facebook_comments:
            matched_post = find_post_match(fb_post_lookup, url=comment.post_url)
            if matched_post:
                comment.post = matched_post
                comments.append(comment)

        for comment in orphan_x_comments:
            matched_post = find_x_post_match(x_post_lookup, url=comment.post_url)
            if matched_post:
                comment.post = matched_post
                comments.append(comment)

        relevance_by_comment_id = _comment_relevance_map(comments)
        ranked_comments = defaultdict(list)
        for comment in comments:
            target_post_id = comment.post_id or getattr(comment.post, "id", None)
            if target_post_id:
                ranked_comments[target_post_id].append(comment)

        comments_by_post = {}
        for post_id, post_comments in ranked_comments.items():
            post_tone = score_tone(db.session.get(Post, post_id).sentiment_score if db.session.get(Post, post_id) else 60)
            comment_views = [
                _build_comment_view(
                    comment,
                    post_tone,
                    relevance_by_comment_id.get(comment.id, True),
                )
                for comment in sorted(
                    post_comments,
                    key=lambda item: (
                        bool(relevance_by_comment_id.get(item.id, True)),
                        comment_rank(item),
                    ),
                    reverse=True,
                )
            ]
            comments_by_post[post_id] = comment_views
            top_comments_by_post_id[post_id] = [
                comment_view
                for comment_view in comment_views[:3]
            ]

    response_posts = []
    for post in posts:
        comment_views = comments_by_post.get(post.id, [])
        tone_counts = {"negative": 0, "neutral": 0, "positive": 0}
        impact_counts = {"high": 0, "medium": 0, "low": 0}
        for comment_view in comment_views:
            if not comment_view.get("isRelevant", True):
                continue
            tone_counts[comment_view["sentimentTone"]] += 1
            impact_counts[comment_view["impactLevel"].lower()] += 1

        post_payload = post.to_api_dict(top_comments=top_comments_by_post_id.get(post.id, []))
        post_payload.update({
            "allComments": comment_views,
            "commentCount": len(comment_views),
            "commentToneSummary": tone_counts,
            "commentImpactSummary": impact_counts,
            "postTone": score_tone(post.sentiment_score),
        })
        response_posts.append(post_payload)

    return jsonify(response_posts)


@posts_bp.route("/posts/<post_id>/status", methods=["PATCH"])
@jwt_required(optional=True)
def update_post_status(post_id):
    data = request.get_json() or {}
    post = db.session.get(Post, post_id)
    if not post:
        return jsonify({"message": "Post not found"}), 404

    old_status = post.status
    post.status = data.get("status", post.status)
    _log_post_activity_for_post(
        "Post status changed",
        f"Status '{old_status}' → '{post.status}'",
        post,
    )
    db.session.commit()
    return jsonify({"id": post_id, "status": post.status})


@posts_bp.route("/posts/<post_id>/verification", methods=["PATCH"])
@jwt_required(optional=True)
def update_post_verification(post_id):
    data = request.get_json() or {}
    post = db.session.get(Post, post_id)
    if not post:
        return jsonify({"message": "Post not found"}), 404

    status = (data.get("status") or "").strip()
    note = (data.get("note") or "").strip()
    marked_by = (data.get("markedBy") or "").strip()

    valid_statuses = {"auto-verified", "auto-unverified", "manually-verified", "marked-unverified"}
    if status and status not in valid_statuses:
        return jsonify({"message": "Invalid verification status"}), 400

    if status:
        post.verification_status = status
    if note is not None:
        post.verification_note = note

    if status in {"manually-verified", "marked-unverified"}:
        post.verification_marked_by = marked_by or current_actor_label()
    elif status in {"auto-verified", "auto-unverified"}:
        post.verification_marked_by = None

    if status:
        detail = f"Post {post_id[:16]}… verification set to '{status}'"
        if note:
            detail += f" — note: '{note}'"
        _log_post_activity_for_post("Post verification updated", detail, post)

    db.session.commit()
    return jsonify({
        "id": post_id,
        "verificationStatus": post.verification_status,
        "verificationNote": post.verification_note,
        "verificationMarkedBy": post.verification_marked_by,
    })


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
        _log_post_activity("Post pinned", f"Added post {post_id[:16]}… to watchlist")
        db.session.commit()
    return get_watchlist()


@posts_bp.route("/watchlist/<post_id>", methods=["DELETE"])
@jwt_required(optional=True)
def unpin_post(post_id):
    username = current_username()
    Watchlist.query.filter_by(username=username, post_id=post_id).delete()
    _log_post_activity("Post unpinned", f"Removed post {post_id[:16]}… from watchlist")
    db.session.commit()
    return get_watchlist()


@posts_bp.route("/clusters", methods=["GET"])
@jwt_required(optional=True)
def get_clusters():
    return jsonify(CLUSTER_DEFINITIONS)


@posts_bp.route("/live/version", methods=["GET"])
@jwt_required(optional=True)
def get_live_version():
    latest_post_change = db.session.query(func.max(func.coalesce(Post.updated_at, Post.created_at))).scalar()
    latest_comment_change = db.session.query(func.max(func.coalesce(Comment.updated_at, Comment.created_at))).scalar()
    latest_activity_change = db.session.query(func.max(func.coalesce(ActivityLog.updated_at, ActivityLog.created_at))).scalar()
    latest_user_change = db.session.query(func.max(func.coalesce(User.updated_at, User.created_at))).scalar()
    latest_setting_change = db.session.query(func.max(func.coalesce(SystemSetting.updated_at, SystemSetting.created_at))).scalar()
    latest_watchlist_change = db.session.query(func.max(func.coalesce(Watchlist.updated_at, Watchlist.created_at))).scalar()

    post_count = Post.query.count()
    comment_count = Comment.query.count()
    activity_count = ActivityLog.query.count()
    user_count = User.query.count()
    setting_count = SystemSetting.query.count()
    watchlist_count = Watchlist.query.count()

    latest_change = _latest_timestamp(
        latest_post_change,
        latest_comment_change,
        latest_activity_change,
        latest_user_change,
        latest_setting_change,
        latest_watchlist_change,
    )

    return jsonify({
        "version": (
            f"{utc_iso(latest_change) or 'empty'}:"
            f"{post_count}:{comment_count}:{activity_count}:{user_count}:{setting_count}:{watchlist_count}"
        ),
        "latestChange": utc_iso(latest_change),
        "postCount": post_count,
        "commentCount": comment_count,
        "activityCount": activity_count,
        "userCount": user_count,
        "settingCount": setting_count,
        "watchlistCount": watchlist_count,
    })


@posts_bp.route("/dashboard/summary", methods=["GET"])
@jwt_required(optional=True)
def get_dashboard_summary():
    date_range = request.args.get("date_range", "7d")
    query = Post.query.filter(Post.is_relevant == True)
    cutoff = date_range_cutoff(date_range)
    if cutoff is not None:
        query = query.filter(Post.date >= cutoff)
    posts = query.all()
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

    meta = date_range_label(date_range)
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
    query = Comment.query
    cutoff = date_range_cutoff(date_range)
    if cutoff is not None:
        query = query.filter(Comment.date >= cutoff)

    comments = query.order_by(Comment.date.desc()).all()
    relevance_by_comment_id = _comment_relevance_map(comments)

    ranked = sorted(comments, key=lambda comment: (comment_rank(comment), comment.likes, comment.date), reverse=True)
    payload = []
    for comment in ranked[:limit]:
        item = comment.to_api_dict()
        is_relevant = relevance_by_comment_id.get(comment.id, True)
        item["isRelevant"] = is_relevant
        item["relevanceLabel"] = "Relevant" if is_relevant else "Irrelevant"
        payload.append(item)
    return jsonify({"comments": payload})


@posts_bp.route("/settings/email-alerts", methods=["PATCH"])
@jwt_required(optional=True)
def update_email_alerts():
    data = request.get_json() or {}
    enabled = bool(data.get("enabled", True))
    return jsonify({"enabled": enabled})
