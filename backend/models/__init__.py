"""
MANA — SQLAlchemy Models
Shared data models used by the Flask backend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from services.rules.decision_engine import compute_engagement_score, evaluate_from_post

db = SQLAlchemy()


class TimestampMixin:
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class User(TimestampMixin, db.Model):
    __tablename__ = "users"

    username = db.Column(db.String(80), primary_key=True)
    name = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    role = db.Column(db.String(80), nullable=False, default="LGU Analyst")
    password_hash = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(32), nullable=False, default="Active")
    last_login_at = db.Column(db.DateTime, nullable=True)
    login_count = db.Column(db.Integer, nullable=False, default=0)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def to_api_dict(self):
        return {
            "id": self.username,
            "username": self.username,
            "name": self.name or self.username,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "login_count": self.login_count or 0,
        }


class ActivityLog(TimestampMixin, db.Model):
    __tablename__ = "activity_logs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    actor_username = db.Column(db.String(80), nullable=True, index=True)
    actor_name = db.Column(db.String(120), nullable=False)
    action = db.Column(db.String(120), nullable=False)
    detail = db.Column(db.Text, nullable=False, default="")
    type = db.Column(db.String(32), nullable=False, default="system", index=True)

    def to_api_dict(self):
        return {
            "id": self.id,
            "user_id": self.actor_username,
            "user_name": self.actor_name,
            "action": self.action,
            "detail": self.detail,
            "type": self.type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SystemSetting(TimestampMixin, db.Model):
    __tablename__ = "system_settings"

    section = db.Column(db.String(32), primary_key=True)
    payload_json = db.Column(db.Text, nullable=False, default="{}")

    @property
    def payload(self):
        return json.loads(self.payload_json or "{}")

    def set_payload(self, payload):
        self.payload_json = json.dumps(payload or {})


class Cluster(TimestampMixin, db.Model):
    __tablename__ = "clusters"

    id = db.Column(db.String(32), primary_key=True)
    short = db.Column(db.String(64), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    accent = db.Column(db.String(16), nullable=False)
    keywords_json = db.Column(db.Text, nullable=False, default="[]")

    @property
    def keywords(self):
        return json.loads(self.keywords_json or "[]")


class Post(TimestampMixin, db.Model):
    __tablename__ = "posts"

    id = db.Column(db.String(128), primary_key=True)
    source = db.Column(db.String(32), nullable=False)
    page_source = db.Column(db.String(255), nullable=False)
    account_url = db.Column(db.String(512), nullable=True)
    author = db.Column(db.String(255), nullable=True)
    caption = db.Column(db.Text, nullable=False, default="")
    source_url = db.Column(db.String(1024), nullable=False)
    external_id = db.Column(db.String(128), nullable=True, index=True)
    reactions = db.Column(db.Integer, nullable=False, default=0)
    shares = db.Column(db.Integer, nullable=False, default=0)
    likes = db.Column(db.Integer, nullable=False, default=0)
    reposts = db.Column(db.Integer, nullable=False, default=0)
    comments = db.Column(db.Integer, nullable=False, default=0)
    views = db.Column(db.Integer, nullable=False, default=0)
    media_type = db.Column(db.String(32), nullable=True)
    priority = db.Column(db.String(32), nullable=False, default="Moderate")
    sentiment_score    = db.Column(db.Integer, nullable=False, default=60)
    sentiment_compound = db.Column(db.Float, nullable=True)
    recommendation = db.Column(db.Text, nullable=False, default="")
    status = db.Column(db.String(32), nullable=False, default="Monitoring")
    cluster_id = db.Column(db.String(32), db.ForeignKey("clusters.id"), nullable=False)
    reviewed_cluster_id = db.Column(db.String(32), db.ForeignKey("clusters.id"), nullable=True, index=True)
    cluster_label_source = db.Column(db.String(32), nullable=False, default="heuristic", index=True)
    is_relevant = db.Column(db.Boolean, nullable=False, default=True, index=True)
    date = db.Column(db.DateTime, nullable=False, index=True)
    keywords_json = db.Column(db.Text, nullable=False, default="[]")
    location = db.Column(db.String(255), nullable=False, default="Philippines")
    severity_rank = db.Column(db.Integer, nullable=False, default=2)
    raw_payload_json = db.Column(db.Text, nullable=True)

    cluster = db.relationship("Cluster", foreign_keys=[cluster_id])
    reviewed_cluster = db.relationship("Cluster", foreign_keys=[reviewed_cluster_id])

    @property
    def keywords(self):
        return json.loads(self.keywords_json or "[]")

    def to_api_dict(self, top_comments=None):
        recommendation_details = evaluate_from_post(
            topic=self.cluster_id,
            priority=self.priority,
            sentiment_score=self.sentiment_score,
            reactions=self.reactions or self.likes,
            comments=self.comments,
            shares=self.shares or self.reposts,
            post_count=1,
        )
        return {
            "id": self.id,
            "source": self.source,
            "pageSource": self.page_source,
            "author": self.author or self.page_source,
            "caption": self.caption,
            "reactions": self.reactions,
            "shares": self.shares,
            "likes": self.likes,
            "reposts": self.reposts,
            "comments": self.comments,
            "priority": self.priority,
            "sentimentScore": self.sentiment_score,
            "sentimentCompound": self.sentiment_compound,
            "recommendation": recommendation_details["recommendation"],
            "recommendationDetails": recommendation_details,
            "engagementScore": compute_engagement_score(
                post_count=1,
                reactions=self.reactions or self.likes,
                comments=self.comments,
                shares=self.shares or self.reposts,
            ),
            "status": self.status,
            "clusterId": self.cluster_id,
            "reviewedClusterId": self.reviewed_cluster_id,
            "clusterLabelSource": self.cluster_label_source,
            "isRelevant": self.is_relevant,
            "date": self.date.isoformat(),
            "keywords": self.keywords,
            "location": self.location,
            "severityRank": self.severity_rank,
            "sourceUrl": self.source_url,
            "mediaType": self.media_type,
            "views": self.views,
            "topComments": top_comments or [],
        }


class Comment(TimestampMixin, db.Model):
    __tablename__ = "comments"

    id = db.Column(db.String(128), primary_key=True)
    post_id = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=True, index=True)
    source = db.Column(db.String(32), nullable=False, default="Facebook")
    page_source = db.Column(db.String(255), nullable=False, default="Facebook Source")
    author = db.Column(db.String(255), nullable=False, default="Facebook user")
    text = db.Column(db.Text, nullable=False, default="")
    likes = db.Column(db.Integer, nullable=False, default=0)
    post_title = db.Column(db.Text, nullable=False, default="")
    post_url = db.Column(db.String(1024), nullable=False)
    cluster_id = db.Column(db.String(32), db.ForeignKey("clusters.id"), nullable=False)
    location = db.Column(db.String(255), nullable=False, default="Philippines")
    date = db.Column(db.DateTime, nullable=False, index=True)
    raw_payload_json = db.Column(db.Text, nullable=True)

    post = db.relationship("Post")
    cluster = db.relationship("Cluster", foreign_keys=[cluster_id])

    def to_api_dict(self):
        return {
            "id": self.id,
            "postId": self.post_id,
            "source": self.source,
            "pageSource": self.page_source,
            "author": self.author,
            "text": self.text,
            "likes": self.likes,
            "postTitle": self.post_title,
            "postUrl": self.post_url,
            "clusterId": self.cluster_id,
            "location": self.location,
            "date": self.date.isoformat() if self.date else None,
        }


class PreprocessedText(TimestampMixin, db.Model):
    __tablename__ = "preprocessed_texts"
    __table_args__ = (db.UniqueConstraint("record_type", "raw_id", name="uq_preprocessed_record"),)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    record_type = db.Column(db.String(32), nullable=False, index=True)
    raw_id = db.Column(db.String(128), nullable=False, index=True)
    raw_text = db.Column(db.Text, nullable=True)
    clean_text = db.Column(db.Text, nullable=True)
    tokens_json = db.Column(db.Text, nullable=False, default="[]")
    translated_text = db.Column(db.Text, nullable=True)
    vader_text = db.Column(db.Text, nullable=True)
    translation_status = db.Column(db.String(32), nullable=False, default="skipped", index=True)
    negation_handled_tokens_json = db.Column(db.Text, nullable=False, default="[]")
    lemmatized_tokens_json = db.Column(db.Text, nullable=False, default="[]")
    bigrams_json = db.Column(db.Text, nullable=False, default="[]")
    final_tokens_json = db.Column(db.Text, nullable=False, default="[]")
    is_emotion_only = db.Column(db.Boolean, nullable=False, default=False, index=True)
    is_relevant = db.Column(db.Boolean, nullable=False, default=True, index=True)
    parent_post_id = db.Column(db.String(128), nullable=True, index=True)
    preprocessing_stage = db.Column(db.String(32), nullable=False, default="tokenized", index=True)
    preprocessing_status = db.Column(db.String(32), nullable=False, default="processed", index=True)
    error_message = db.Column(db.Text, nullable=True)

    @property
    def tokens(self):
        return json.loads(self.tokens_json or "[]")

    def set_tokens(self, tokens):
        self.tokens_json = json.dumps(tokens or [])

    @property
    def negation_handled_tokens(self):
        return json.loads(self.negation_handled_tokens_json or "[]")

    def set_negation_handled_tokens(self, tokens):
        self.negation_handled_tokens_json = json.dumps(tokens or [])

    @property
    def lemmatized_tokens(self):
        return json.loads(self.lemmatized_tokens_json or "[]")

    def set_lemmatized_tokens(self, tokens):
        self.lemmatized_tokens_json = json.dumps(tokens or [])

    @property
    def bigrams(self):
        return json.loads(self.bigrams_json or "[]")

    def set_bigrams(self, bigrams):
        self.bigrams_json = json.dumps(bigrams or [])

    @property
    def final_tokens(self):
        return json.loads(self.final_tokens_json or "[]")

    def set_final_tokens(self, tokens):
        self.final_tokens_json = json.dumps(tokens or [])

    def to_api_dict(self):
        return {
            "raw_id": self.raw_id,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "translated_text": self.translated_text,
            "tokens": self.tokens,
            "negation_handled_tokens": self.negation_handled_tokens,
            "lemmatized_tokens": self.lemmatized_tokens,
            "bigrams": self.bigrams,
            "final_tokens": self.final_tokens,
            "translation_status": self.translation_status,
            "is_emotion_only": self.is_emotion_only,
            "is_relevant": self.is_relevant,
            "parent_post_id": self.parent_post_id,
            "preprocessing_stage": self.preprocessing_stage,
            "preprocessing_status": self.preprocessing_status,
            "error_message": self.error_message,
        }


class Watchlist(TimestampMixin, db.Model):
    __tablename__ = "watchlists"
    __table_args__ = (db.UniqueConstraint("username", "post_id", name="uq_watchlist_username_post"),)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), nullable=False, index=True)
    post_id = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=False, index=True)


class PostTopic(TimestampMixin, db.Model):
    __tablename__ = "post_topics"
    __table_args__ = (db.UniqueConstraint("post_id", "topic_label", name="uq_post_topic"),)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    post_id = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=False, index=True)
    topic_label = db.Column(db.String(64), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)

    post = db.relationship("Post")

    def to_api_dict(self):
        return {
            "post_id": self.post_id,
            "topic_label": self.topic_label,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PostCluster(TimestampMixin, db.Model):
    __tablename__ = "post_clusters"
    __table_args__ = (db.UniqueConstraint("post_id", "cluster_id", name="uq_post_cluster"),)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    post_id = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=False, index=True)
    cluster_id = db.Column(db.String(32), db.ForeignKey("clusters.id"), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)

    post = db.relationship("Post")
    cluster = db.relationship("Cluster", foreign_keys=[cluster_id])

    def to_api_dict(self):
        return {
            "post_id": self.post_id,
            "cluster_id": self.cluster_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PostSentiment(TimestampMixin, db.Model):
    """
    VADER sentiment scores for posts. Thesis schema: sentiments table.
    One row per post — upserted on each VADER analysis run.
    """
    __tablename__ = "sentiments"

    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    post_id      = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=False, unique=True, index=True)
    compound     = db.Column(db.Float, nullable=False, default=0.0)
    positive     = db.Column(db.Float, nullable=False, default=0.0)
    negative     = db.Column(db.Float, nullable=False, default=0.0)
    neutral      = db.Column(db.Float, nullable=False, default=1.0)
    sarcasm_flag = db.Column(db.Boolean, nullable=False, default=False)

    post = db.relationship("Post", backref=db.backref("sentiment", uselist=False))

    def to_api_dict(self):
        return {
            "post_id":      self.post_id,
            "compound":     self.compound,
            "positive":     self.positive,
            "negative":     self.negative,
            "neutral":      self.neutral,
            "sarcasm_flag": self.sarcasm_flag,
            "created_at":   self.created_at.isoformat() if self.created_at else None,
        }


class PostPriority(TimestampMixin, db.Model):
    """
    Random Forest priority predictions. Thesis schema: post_priorities table.
    One row per post — upserted on each RF predict-all run.
    Stores label, confidence, and per-class probabilities (High / Medium / Low).
    """
    __tablename__ = "post_priorities"

    id                 = db.Column(db.Integer, primary_key=True, autoincrement=True)
    post_id            = db.Column(db.String(128), db.ForeignKey("posts.id"), nullable=False, unique=True, index=True)
    priority_label     = db.Column(db.String(32), nullable=False)
    confidence         = db.Column(db.Float, nullable=False, default=0.0)
    high_probability   = db.Column(db.Float, nullable=False, default=0.0)
    medium_probability = db.Column(db.Float, nullable=False, default=0.0)
    low_probability    = db.Column(db.Float, nullable=False, default=0.0)

    post = db.relationship("Post", backref=db.backref("rf_priority", uselist=False))

    def to_api_dict(self):
        return {
            "post_id":            self.post_id,
            "priority_label":     self.priority_label,
            "confidence":         self.confidence,
            "high_probability":   self.high_probability,
            "medium_probability": self.medium_probability,
            "low_probability":    self.low_probability,
            "created_at":         self.created_at.isoformat() if self.created_at else None,
        }
