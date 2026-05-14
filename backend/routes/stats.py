"""
MANA — Analytics / stats routes backed by imported posts.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from data import CLUSTER_MAP, now_utc, parse_date_range, priority_label, score_tone
from models import Post

stats_bp = Blueprint("stats", __name__)


def comparable_dt(value):
    if value is None:
        return None
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def filtered_posts(date_range: str):
    cutoff = now_utc() - parse_date_range(date_range)
    return Post.query.filter(Post.date >= cutoff).order_by(Post.date.asc()).all()


@stats_bp.route("/analytics/sentiment-histogram", methods=["GET"])
@stats_bp.route("/analytics/histogram", methods=["GET"])
@jwt_required(optional=True)
def sentiment_histogram():
    date_range = request.args.get("date_range", "14d")
    buckets = [
        {"label": "0-20", "value": 0, "tone": "negative"},
        {"label": "21-40", "value": 0, "tone": "negative"},
        {"label": "41-60", "value": 0, "tone": "neutral"},
        {"label": "61-80", "value": 0, "tone": "positive"},
        {"label": "81-100", "value": 0, "tone": "positive"},
    ]

    for post in filtered_posts(date_range):
        score = post.sentiment_score
        if score <= 20:
            buckets[0]["value"] += 1
        elif score <= 40:
            buckets[1]["value"] += 1
        elif score <= 60:
            buckets[2]["value"] += 1
        elif score <= 80:
            buckets[3]["value"] += 1
        else:
            buckets[4]["value"] += 1

    return jsonify({"histogram": buckets})


@stats_bp.route("/analytics/sentiment-trend", methods=["GET"])
@stats_bp.route("/analytics/trend", methods=["GET"])
@jwt_required(optional=True)
def sentiment_trend():
    date_range = request.args.get("date_range", "14d")
    days = {"7d": 7, "14d": 7, "30d": 7}.get(date_range, 7)
    window = {"7d": 1, "14d": 2, "30d": 4}.get(date_range, 2)
    start = now_utc() - timedelta(days=days * window - 1)

    labels = []
    rows = []
    for i in range(days):
        bucket_start = start + timedelta(days=i * window)
        bucket_end = bucket_start + timedelta(days=window)
        labels.append(bucket_start.strftime("%b %d") if date_range == "7d" else f"W{i + 1}")
        rows.append((bucket_start, bucket_end))

    positive, neutral, negative = [], [], []
    posts = filtered_posts(date_range)
    for bucket_start, bucket_end in rows:
        bucket_start_cmp = comparable_dt(bucket_start)
        bucket_end_cmp = comparable_dt(bucket_end)
        pos = neu = neg = 0
        for post in posts:
            post_date_cmp = comparable_dt(post.date)
            if post_date_cmp and bucket_start_cmp <= post_date_cmp < bucket_end_cmp:
                tone = score_tone(post.sentiment_score)
                if tone == "positive":
                    pos += 1
                elif tone == "neutral":
                    neu += 1
                else:
                    neg += 1
        positive.append(pos)
        neutral.append(neu)
        negative.append(neg)

    return jsonify({"labels": labels, "positive": positive, "neutral": neutral, "negative": negative})


@stats_bp.route("/analytics/cluster-activity", methods=["GET"])
@jwt_required(optional=True)
def cluster_activity():
    from services.priority.priority_scorer import compute_cluster_trends
    from models import db
    
    date_range = request.args.get("date_range", "14d")
    counts = defaultdict(int)
    for post in filtered_posts(date_range):
        counts[post.cluster_id] += 1

    # Get the advanced 4-factor trend scores to sort the options
    trends = compute_cluster_trends(db.session)

    cluster_activity_data = []
    for cluster_id, cluster in CLUSTER_MAP.items():
        label = cluster["name"].replace("WASH, Medical and Public Health, Nutrition, Mental Health and Psychosocial Support (Health)", "Health")
        trend_score = trends.get(cluster_id, {}).get("trend_score", 0)
        cluster_activity_data.append({
            "cluster_id": cluster_id,
            "label": label, 
            "value": counts.get(cluster_id, 0), 
            "color": cluster["accent"],
            "_trend_score": trend_score  # Hidden score for sorting
        })
        
    # Sort from most trending to least trending
    cluster_activity_data.sort(key=lambda x: x["_trend_score"], reverse=True)
    
    # Remove the hidden score before sending to frontend to maintain 0 blast radius
    for item in cluster_activity_data:
        del item["_trend_score"]

    return jsonify({"clusterActivity": cluster_activity_data})


@stats_bp.route("/dashboard/posts-over-time", methods=["GET"])
@jwt_required(optional=True)
def posts_over_time():
    from datetime import datetime, timezone
    date_range = request.args.get("date_range", "24h")
    cutoff = now_utc() - timedelta(hours=24)
    posts = Post.query.filter(Post.is_relevant == True, Post.date >= cutoff).order_by(Post.date.asc()).all()

    hourly_counts = defaultdict(int)
    for post in posts:
        dt = post.date
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hourly_counts[dt.strftime("%H:00")] += 1

    now = now_utc()
    result = []
    for i in range(24):
        h = (now.hour - 23 + i) % 24
        label = f"{h:02d}:00"
        result.append({"hour": label, "count": hourly_counts.get(label, 0)})

    peak_entry = max(result, key=lambda x: x["count"]) if result else {"hour": "00:00", "count": 0}
    total_volume = sum(e["count"] for e in result)

    return jsonify({
        "hourly": result,
        "peak": {"hour": peak_entry["hour"], "count": peak_entry["count"]},
        "spike": {"time": peak_entry["hour"] + " PHT", "volume": total_volume},
    })


@stats_bp.route("/analytics/priority-distribution", methods=["GET"])
@stats_bp.route("/analytics/priority", methods=["GET"])
@jwt_required(optional=True)
def priority_distribution():
    date_range = request.args.get("date_range", "14d")
    counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    colors = {"Low": "#34d399", "Medium": "#38bdf8", "High": "#f59e0b", "Critical": "#fb7185"}

    for post in filtered_posts(date_range):
        counts[priority_label(post.priority)] += 1

    return jsonify(
        {
            "priority": [
                {"label": label, "value": counts[label], "color": colors[label]}
                for label in ["Low", "Medium", "High", "Critical"]
            ]
        }
    )
