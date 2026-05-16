"""
MANA — Admin Routes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from functools import wraps
import traceback

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required

from data import now_utc, parse_date_range, priority_label, recommendation_payload_for, score_tone
from models import ActivityLog, Comment, Post, PostCluster, PostPriority, PostSentiment, PostTopic, PreprocessedText, SystemSetting, User, db
from services.apify_integration import (
    KIND_FACEBOOK,
    VALID_KINDS,
    extract_kind,
    get_task_id,
    import_dataset_items,
    resolve_dataset_id,
    start_task,
    validate_webhook_secret,
)


admin_bp = Blueprint("admin", __name__)

_RF_TO_REC_PRIORITY = {"High": "Critical", "Medium": "Moderate", "Low": "Monitoring"}


def _auto_classify_new_posts(app) -> None:
    """Run ML predictions on posts that have been imported but not yet classified."""
    try:
        from services.corex.topic_modeler import is_model_trained as corex_trained, predict_topics_batch
        from services.svm.cluster_classifier import is_model_trained as svm_trained, predict_clusters_batch, select_top_cluster
        from services.vader.sentiment_analyzer import analyze_post_with_comments
        from services.random_forest.priority_classifier import SEVERITY_MAP, is_model_trained as rf_trained, predict_priorities_batch

        if not (corex_trained() and svm_trained() and rf_trained()):
            print("[auto-classify] Models not fully trained — skipping auto-classification.")
            return

        with app.app_context():
            rows = (
                PreprocessedText.query
                .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
                .filter(PreprocessedText.final_tokens_json != "[]")
                .outerjoin(PostCluster, PreprocessedText.raw_id == PostCluster.post_id)
                .filter(PostCluster.post_id.is_(None))
                .all()
            )

            if not rows:
                print("[auto-classify] No new unclassified posts found.")
                return

            print(f"[auto-classify] Classifying {len(rows)} new post(s)…")
            post_ids = [row.raw_id for row in rows]
            texts = [" ".join(row.final_tokens) for row in rows]
            posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}

            # Topics
            for post_id, topic_list in zip(post_ids, predict_topics_batch(texts)):
                PostTopic.query.filter_by(post_id=post_id).delete()
                for item in topic_list:
                    db.session.add(PostTopic(post_id=post_id, topic_label=item["topic"], confidence=item["confidence"]))

            # Clusters
            for post_id, cluster_list in zip(post_ids, predict_clusters_batch(texts)):
                if not cluster_list:
                    continue
                for item in cluster_list:
                    db.session.add(PostCluster(post_id=post_id, cluster_id=item["cluster_id"], confidence=item["confidence"]))
                top = select_top_cluster(cluster_list)
                post = posts_map.get(post_id)
                if post and top:
                    post.cluster_id = top["cluster_id"]
                    post.cluster_label_source = "svm"

            # Sentiment
            comments_by_post: dict[str, list[str]] = {}
            for comment in Comment.query.filter(Comment.post_id.in_(post_ids)).all():
                comments_by_post.setdefault(comment.post_id, []).append(comment.text or "")
            vader_texts = {row.raw_id: (row.vader_text or row.clean_text) for row in rows}
            for post_id in post_ids:
                post = posts_map.get(post_id)
                if not post:
                    continue
                text = vader_texts.get(post_id) or post.caption or ""
                result = analyze_post_with_comments(text, post.cluster_id, comments_by_post.get(post_id, []))
                existing = PostSentiment.query.filter_by(post_id=post_id).first()
                if existing:
                    existing.compound = result["compound"]
                    existing.positive = result["positive"]
                    existing.negative = result["negative"]
                    existing.neutral = result["neutral"]
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
                post.sentiment_score = result["sentiment_score"]
                post.sentiment_compound = result["compound"]

            # Priority
            for pred in predict_priorities_batch(post_ids):
                pid, label, conf, probs = pred["post_id"], pred["priority"], pred["confidence"], pred["probabilities"]
                post = posts_map.get(pid)
                if not post:
                    continue
                post.priority = label
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
                    existing.priority_label = label
                    existing.confidence = conf
                    existing.high_probability = probs.get("High", 0.0)
                    existing.medium_probability = probs.get("Medium", 0.0)
                    existing.low_probability = probs.get("Low", 0.0)
                else:
                    db.session.add(PostPriority(
                        post_id=pid,
                        priority_label=label,
                        confidence=conf,
                        high_probability=probs.get("High", 0.0),
                        medium_probability=probs.get("Medium", 0.0),
                        low_probability=probs.get("Low", 0.0),
                    ))

            db.session.commit()
            print(f"[auto-classify] Done — {len(post_ids)} post(s) classified and committed.")

    except Exception as exc:
        print(f"[auto-classify] Error during background classification: {exc}")
        traceback.print_exc()


def _start_auto_classify() -> None:
    import threading
    from flask import current_app
    app = current_app._get_current_object()
    threading.Thread(target=_auto_classify_new_posts, args=(app,), daemon=True).start()


def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "Admin":
            return jsonify({"message": "Admin access required."}), 403
        return fn(*args, **kwargs)

    return wrapper


def current_admin():
    username = get_jwt_identity()
    return db.session.get(User, username) if username else None


def get_json():
    return request.get_json() or {}


def log_activity(action: str, detail: str, log_type: str = "admin", actor: User | None = None):
    if actor is None:
        try:
            actor = current_admin()
        except Exception:
            actor = None
    db.session.add(
        ActivityLog(
            actor_username=actor.username if actor else None,
            actor_name=(actor.name or actor.username) if actor else "System",
            action=action,
            detail=detail,
            type=log_type,
        )
    )


def ensure_unique_user(email: str, username: str, current_username: str | None = None):
    email_match = User.query.filter(User.email == email)
    username_match = User.query.filter(User.username == username)
    if current_username:
        email_match = email_match.filter(User.username != current_username)
        username_match = username_match.filter(User.username != current_username)
    if email_match.first():
        return "Email is already in use."
    if username_match.first():
        return "Username is already in use."
    return None


def username_from_email(email: str):
    base = email.split("@", 1)[0].strip().lower().replace(" ", ".")
    candidate = base
    index = 2
    while db.session.get(User, candidate):
        candidate = f"{base}{index}"
        index += 1
    return candidate


def public_webhook_url():
    return request.host_url.rstrip("/") + "/api/admin/apify/webhook"


@admin_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    search = (request.args.get("search") or "").strip().lower()
    role = request.args.get("role") or ""
    status = request.args.get("status") or ""

    query = User.query.order_by(User.created_at.asc())
    if role and role != "all":
        query = query.filter(User.role == role)
    if status and status != "all":
        query = query.filter(User.status == status)

    users = query.all()
    if search:
        users = [
            user for user in users
            if search in (user.name or user.username).lower()
            or search in user.email.lower()
            or search in user.username.lower()
        ]
    return jsonify([user.to_api_dict() for user in users])


@admin_bp.route("/users", methods=["POST"])
@admin_required
def create_user():
    data = get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    role = (data.get("role") or "LGU Analyst").strip()
    password = data.get("password") or ""
    username = (data.get("username") or "").strip().lower() or username_from_email(email)

    if not name:
        return jsonify({"message": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"message": "A valid email is required."}), 400
    if role not in {"Admin", "LGU Analyst", "Viewer"}:
        return jsonify({"message": "Invalid role."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400

    conflict = ensure_unique_user(email, username)
    if conflict:
        return jsonify({"message": conflict}), 409

    user = User(username=username, name=name, email=email, role=role, status="Active")
    user.set_password(password)
    db.session.add(user)
    log_activity("User created", f"Created {name} ({role})", "admin")
    db.session.commit()
    return jsonify(user.to_api_dict()), 201


@admin_bp.route("/users/<user_id>", methods=["PATCH"])
@admin_required
def update_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    data = get_json()
    name = (data.get("name") or user.name or user.username).strip()
    email = (data.get("email") or user.email).strip().lower()
    role = (data.get("role") or user.role).strip()

    if not name:
        return jsonify({"message": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"message": "A valid email is required."}), 400
    if role not in {"Admin", "LGU Analyst", "Viewer"}:
        return jsonify({"message": "Invalid role."}), 400

    conflict = ensure_unique_user(email, user.username, current_username=user.username)
    if conflict:
        return jsonify({"message": conflict}), 409

    user.name = name
    user.email = email
    user.role = role
    log_activity("User updated", f"Updated {name} ({role})", "admin")
    db.session.commit()
    return jsonify(user.to_api_dict())


@admin_bp.route("/users/<user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    if user.username == get_jwt_identity():
        return jsonify({"message": "You cannot delete your own admin account."}), 400

    name = user.name or user.username
    ActivityLog.query.filter_by(actor_username=user.username).delete()
    db.session.delete(user)
    log_activity("User deleted", f"Deleted {name}", "admin")
    db.session.commit()
    return jsonify({"success": True})


@admin_bp.route("/users/<user_id>/status", methods=["PATCH"])
@admin_required
def set_user_status(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    status = (get_json().get("status") or "").strip()
    if status not in {"Active", "Suspended", "Inactive"}:
        return jsonify({"message": "Invalid status."}), 400
    user.status = status
    log_activity("User status changed", f"{user.name or user.username} set to {status}", "admin")
    db.session.commit()
    return jsonify({"id": user.username, "status": user.status})


@admin_bp.route("/users/<user_id>/reset-password", methods=["POST"])
@admin_required
def reset_password(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    new_password = get_json().get("new_password") or ""
    if len(new_password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400
    user.set_password(new_password)
    log_activity("Password reset", f"Reset password for {user.name or user.username}", "admin")
    db.session.commit()
    return jsonify({"success": True})


@admin_bp.route("/logs", methods=["GET"])
@admin_required
def get_logs():
    user_id = request.args.get("user_id")
    log_type = request.args.get("type")
    limit = max(1, min(int(request.args.get("limit", 50)), 200))

    query = ActivityLog.query.order_by(ActivityLog.created_at.desc())
    if user_id:
        query = query.filter(ActivityLog.actor_username == user_id)
    if log_type and log_type != "all":
        query = query.filter(ActivityLog.type == log_type)

    logs = query.limit(limit).all()
    return jsonify([log.to_api_dict() for log in logs])


@admin_bp.route("/stats", methods=["GET"])
@admin_required
def get_stats():
    date_range = request.args.get("date_range", "7d")
    cutoff = now_utc() - parse_date_range(date_range)
    posts = Post.query.filter(Post.is_relevant == True, Post.date >= cutoff).order_by(Post.date.asc()).all()

    total_posts = len(posts)
    fb_posts = sum(1 for post in posts if post.source == "Facebook")
    x_posts = sum(1 for post in posts if post.source == "X")
    critical = sum(1 for post in posts if post.priority == "Critical")
    high = sum(1 for post in posts if post.priority == "High")
    moderate = sum(1 for post in posts if post.priority == "Moderate")
    low = sum(1 for post in posts if post.priority == "Monitoring")

    keyword_counts = Counter()
    sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
    bucket_counts = defaultdict(lambda: {"Facebook": 0, "X": 0, "critical": 0, "high": 0})

    for post in posts:
        for keyword in post.keywords:
            keyword_counts[keyword] += 1
        sentiment_counts[score_tone(post.sentiment_score)] += 1
        key = post.date.strftime("%b %d") if date_range == "7d" else post.date.strftime("%b %d")
        bucket_counts[key][post.source] += 1
        if post.priority == "Critical":
            bucket_counts[key]["critical"] += 1
        if post.priority == "High":
            bucket_counts[key]["high"] += 1

    labels = sorted(bucket_counts.keys(), key=lambda value: posts[[p.date.strftime("%b %d") for p in posts].index(value)].date if posts else now_utc())
    trend_fb = [bucket_counts[label]["Facebook"] for label in labels]
    trend_x = [bucket_counts[label]["X"] for label in labels]
    critical_trend = [bucket_counts[label]["critical"] for label in labels]
    high_trend = [bucket_counts[label]["high"] for label in labels]

    total_sentiment = max(sum(sentiment_counts.values()), 1)
    top_keywords = keyword_counts.most_common(6)
    max_keyword = top_keywords[0][1] if top_keywords else 1

    return jsonify(
        {
            "totalPosts": total_posts,
            "fbPosts": fb_posts,
            "xPosts": x_posts,
            "critical": critical,
            "high": high,
            "moderate": moderate,
            "low": low,
            "topKeywords": [
                {"word": word, "count": count, "pct": round((count / max_keyword) * 100)}
                for word, count in top_keywords
            ],
            "sentiment": {
                key: round((value / total_sentiment) * 100) for key, value in sentiment_counts.items()
            },
            "trendLabels": labels,
            "trendFb": trend_fb,
            "trendX": trend_x,
            "priorityTrend": {"critical": critical_trend, "high": high_trend},
        }
    )


@admin_bp.route("/settings", methods=["GET"])
@admin_required
def get_settings():
    rows = SystemSetting.query.order_by(SystemSetting.section.asc()).all()
    return jsonify({row.section: row.payload for row in rows})


@admin_bp.route("/settings/<section>", methods=["PATCH"])
@admin_required
def update_settings(section):
    allowed = {"general", "security", "notifications", "system"}
    if section not in allowed:
        return jsonify({"message": f"Invalid section. Must be one of: {allowed}"}), 400

    setting = db.session.get(SystemSetting, section)
    if not setting:
        setting = SystemSetting(section=section)
        setting.set_payload({})
        db.session.add(setting)

    payload = setting.payload
    payload.update(get_json())
    setting.set_payload(payload)
    log_activity("Settings updated", f"Updated {section} settings", "system")
    db.session.commit()
    return jsonify({"success": True, "section": section, "data": setting.payload})


@admin_bp.route("/apify/config", methods=["GET"])
@admin_required
def get_apify_config():
    try:
        configured = {
            KIND_FACEBOOK: {
                "taskId": get_task_id(KIND_FACEBOOK),
                "configured": True,
            }
        }
    except RuntimeError:
        configured = {
            KIND_FACEBOOK: {
                "taskId": None,
                "configured": False,
            }
        }

    return jsonify({"webhookUrl": public_webhook_url(), "tasks": configured})




@admin_bp.route("/apify/start", methods=["POST"])
@admin_required
def start_apify_task():
    data = get_json()
    kind = (data.get("kind") or "").strip().lower()
    if kind not in VALID_KINDS:
        return jsonify({"message": f"kind must be one of: {sorted(VALID_KINDS)}"}), 400

    task_input = data.get("taskInput")
    if task_input is not None and not isinstance(task_input, dict):
        return jsonify({"message": "taskInput must be a JSON object."}), 400

    try:
        result = start_task(kind, webhook_url=public_webhook_url(), task_input=task_input)
    except Exception as exc:
        return jsonify({"message": str(exc)}), 500

    log_activity(
        "Apify task started",
        f"Started {kind} task {result['task_id']} (run {result['run_id']})",
        "system",
    )
    db.session.commit()
    return jsonify(result), 202


@admin_bp.route("/apify/import-dataset", methods=["POST"])
@admin_required
def import_apify_dataset():
    data = get_json()
    kind = (data.get("kind") or "").strip().lower()
    dataset_id = (data.get("datasetId") or "").strip()
    if kind not in VALID_KINDS:
        return jsonify({"message": f"kind must be one of: {sorted(VALID_KINDS)}"}), 400
    if not dataset_id:
        return jsonify({"message": "datasetId is required."}), 400

    try:
        result = import_dataset_items(kind, dataset_id)
    except Exception as exc:
        db.session.rollback()
        return jsonify({"message": str(exc)}), 500

    log_activity(
        "Apify dataset imported",
        f"Imported {kind} dataset {dataset_id} ({result['item_count']} items)",
        "system",
    )
    db.session.commit()
    _start_auto_classify()
    return jsonify(result)


@admin_bp.route("/apify/repair-post-metrics", methods=["POST"])
@admin_required
def repair_apify_post_metrics():
    from import_facebook_dataset import refresh_post_metrics_from_payload

    posts = Post.query.filter_by(source="Facebook").all()
    scanned = repaired = skipped = 0

    for post in posts:
        scanned += 1
        if refresh_post_metrics_from_payload(post):
            repaired += 1
        elif post.raw_payload_json:
            skipped += 1

    db.session.commit()
    log_activity(
        "Apify post metrics repaired",
        f"Scanned {scanned} Facebook posts; repaired {repaired}",
        "system",
    )
    db.session.commit()
    return jsonify({
        "message": "Facebook post metrics repaired from stored Apify payloads.",
        "scanned": scanned,
        "repaired": repaired,
        "skipped": skipped,
    })


@admin_bp.route("/apify/webhook", methods=["POST"])
def apify_webhook():
    payload = get_json()
    if not validate_webhook_secret(payload.get("secret")):
        return jsonify({"message": "Invalid webhook secret."}), 403

    kind = extract_kind(payload)
    dataset_id = resolve_dataset_id(payload)
    if kind not in VALID_KINDS or not dataset_id:
        return jsonify({"message": "Webhook payload missing kind or dataset id."}), 400

    try:
        result = import_dataset_items(kind, dataset_id)
        log_activity(
            "Apify webhook import",
            f"Imported {kind} dataset {dataset_id} ({result['item_count']} items)",
            "system",
            actor=None,
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"message": str(exc), "error_type": type(exc).__name__}), 500

    _start_auto_classify()
    return jsonify({"success": True, **result})
