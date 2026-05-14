"""
MANA — Flask Backend Entry Point
Run: python app.py  (dev)
     gunicorn app:app  (prod)

Install: pip install flask flask-cors flask-sqlalchemy flask-jwt-extended
"""

import os
import sqlite3

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager

load_dotenv()

from data import seed_clusters
from models import SystemSetting, User, db
from routes.auth     import auth_bp
from routes.posts    import posts_bp
from routes.stats    import stats_bp
from routes.admin    import admin_bp

OPTIONAL_BLUEPRINTS = []

try:
    from routes.corex import corex_bp
    OPTIONAL_BLUEPRINTS.append((corex_bp, "/api/admin"))
except ModuleNotFoundError:
    corex_bp = None

try:
    from routes.svm import svm_bp
    OPTIONAL_BLUEPRINTS.append((svm_bp, "/api/admin"))
except ModuleNotFoundError:
    svm_bp = None

try:
    from routes.vader import vader_bp
    OPTIONAL_BLUEPRINTS.append((vader_bp, "/api/admin"))
except ModuleNotFoundError:
    vader_bp = None

try:
    from routes.pipeline import pipeline_bp
    OPTIONAL_BLUEPRINTS.append((pipeline_bp, "/api/admin"))
except ModuleNotFoundError:
    pipeline_bp = None

try:
    from routes.random_forest import rf_bp
    OPTIONAL_BLUEPRINTS.append((rf_bp, "/api/admin"))
except ModuleNotFoundError:
    rf_bp = None

try:
    from routes.rules import rules_bp
    OPTIONAL_BLUEPRINTS.append((rules_bp, "/api/admin"))
except ModuleNotFoundError:
    rules_bp = None

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_SECRET_32B")
_db_url = os.environ.get("DATABASE_URL", "sqlite:///mana.db")
# Some platforms still provide postgres:// URLs; SQLAlchemy 2 expects postgresql://.
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ── Extensions ────────────────────────────────────────────────────────────────
_cors_origins_env = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:5500",
)
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
CORS(app, origins=_cors_origins)
JWTManager(app)
db.init_app(app)


@app.route("/", methods=["GET"])
def root():
    return {"status": "ok", "service": "mana-backend"}


@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}

# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(auth_bp,     url_prefix="/api/auth")
app.register_blueprint(posts_bp,    url_prefix="/api")
app.register_blueprint(stats_bp,    url_prefix="/api")
app.register_blueprint(admin_bp,    url_prefix="/api/admin")
for blueprint, url_prefix in OPTIONAL_BLUEPRINTS:
    app.register_blueprint(blueprint, url_prefix=url_prefix)

DEFAULT_SETTINGS = {
    "general": {
        "systemName": "MANA — Manila Advisory Network Alert",
        "systemDesc": "Disaster Response Recommendation and Decision Support System for Philippine LGUs.",
        "timezone": "Asia/Manila",
        "dateFormat": "MMM D, YYYY",
        "defaultRange": "7d",
        "maintenanceMode": False,
    },
    "security": {
        "sessionTimeout": 30,
        "maxLoginAttempts": 5,
        "require2FA": False,
        "passwordMinLength": 8,
        "logRetentionDays": 90,
    },
    "notifications": {
        "emailAlerts": True,
        "criticalAlerts": True,
        "dailyDigest": False,
        "alertEmail": "admin@mana.ph",
    },
    "system": {
        "scrapeInterval": 60,
        "maxPostsPerRun": 500,
        "retryOnFail": True,
        "debugMode": False,
        "backupEnabled": True,
        "backupFreq": "daily",
    },
}


def ensure_user_columns():
    db_path = app.instance_path + "\\mana.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    columns = {row[1] for row in cur.execute("PRAGMA table_info(users)").fetchall()}
    wanted = {
        "name": "ALTER TABLE users ADD COLUMN name VARCHAR(120)",
        "last_login_at": "ALTER TABLE users ADD COLUMN last_login_at DATETIME",
        "login_count": "ALTER TABLE users ADD COLUMN login_count INTEGER NOT NULL DEFAULT 0",
    }
    for column, statement in wanted.items():
        if column not in columns:
            cur.execute(statement)
    conn.commit()
    conn.close()


def ensure_sentiment_columns():
    """Add posts.sentiment_compound column if it doesn't exist (SQLite ALTER TABLE migration)."""
    db_path = app.instance_path + "\\mana.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    columns = {row[1] for row in cur.execute("PRAGMA table_info(posts)").fetchall()}
    if "sentiment_compound" not in columns:
        cur.execute("ALTER TABLE posts ADD COLUMN sentiment_compound REAL")
    conn.commit()
    conn.commit()
    conn.close()


def ensure_post_cluster_label_columns():
    """Add reviewed cluster tracking fields to posts for safer SVM training."""
    db_path = app.instance_path + "\\mana.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    columns = {row[1] for row in cur.execute("PRAGMA table_info(posts)").fetchall()}
    wanted = {
        "reviewed_cluster_id": "ALTER TABLE posts ADD COLUMN reviewed_cluster_id VARCHAR(32)",
        "cluster_label_source": "ALTER TABLE posts ADD COLUMN cluster_label_source VARCHAR(32) NOT NULL DEFAULT 'heuristic'",
        "is_relevant": "ALTER TABLE posts ADD COLUMN is_relevant BOOLEAN NOT NULL DEFAULT 1",
    }
    for column, statement in wanted.items():
        if column not in columns:
            cur.execute(statement)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_posts_reviewed_cluster_id ON posts(reviewed_cluster_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_posts_cluster_label_source ON posts(cluster_label_source)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_posts_is_relevant ON posts(is_relevant)"
    )
    conn.commit()
    conn.close()


def ensure_preprocessed_text_columns():
    db_path = app.instance_path + "\\mana.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS preprocessed_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_type VARCHAR(32) NOT NULL,
            raw_id VARCHAR(128) NOT NULL,
            raw_text TEXT,
            clean_text TEXT,
            tokens_json TEXT NOT NULL DEFAULT '[]',
            translated_text TEXT,
            vader_text TEXT,
            translation_status VARCHAR(32) NOT NULL DEFAULT 'skipped',
            negation_handled_tokens_json TEXT NOT NULL DEFAULT '[]',
            lemmatized_tokens_json TEXT NOT NULL DEFAULT '[]',
            bigrams_json TEXT NOT NULL DEFAULT '[]',
            final_tokens_json TEXT NOT NULL DEFAULT '[]',
            is_emotion_only BOOLEAN NOT NULL DEFAULT 0,
            is_relevant BOOLEAN NOT NULL DEFAULT 1,
            parent_post_id VARCHAR(128),
            preprocessing_stage VARCHAR(32) NOT NULL DEFAULT 'tokenized',
            preprocessing_status VARCHAR(32) NOT NULL DEFAULT 'processed',
            error_message TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {row[1] for row in cur.execute("PRAGMA table_info(preprocessed_texts)").fetchall()}
    wanted = {
        "translated_text": "ALTER TABLE preprocessed_texts ADD COLUMN translated_text TEXT",
        "vader_text": "ALTER TABLE preprocessed_texts ADD COLUMN vader_text TEXT",
        "translation_status": "ALTER TABLE preprocessed_texts ADD COLUMN translation_status VARCHAR(32) NOT NULL DEFAULT 'skipped'",
        "negation_handled_tokens_json": "ALTER TABLE preprocessed_texts ADD COLUMN negation_handled_tokens_json TEXT NOT NULL DEFAULT '[]'",
        "lemmatized_tokens_json": "ALTER TABLE preprocessed_texts ADD COLUMN lemmatized_tokens_json TEXT NOT NULL DEFAULT '[]'",
        "bigrams_json": "ALTER TABLE preprocessed_texts ADD COLUMN bigrams_json TEXT NOT NULL DEFAULT '[]'",
        "final_tokens_json": "ALTER TABLE preprocessed_texts ADD COLUMN final_tokens_json TEXT NOT NULL DEFAULT '[]'",
        "is_emotion_only": "ALTER TABLE preprocessed_texts ADD COLUMN is_emotion_only BOOLEAN NOT NULL DEFAULT 0",
        "is_relevant": "ALTER TABLE preprocessed_texts ADD COLUMN is_relevant BOOLEAN NOT NULL DEFAULT 1",
        "parent_post_id": "ALTER TABLE preprocessed_texts ADD COLUMN parent_post_id VARCHAR(128)",
        "preprocessing_stage": "ALTER TABLE preprocessed_texts ADD COLUMN preprocessing_stage VARCHAR(32) NOT NULL DEFAULT 'tokenized'",
    }
    for column, statement in wanted.items():
        if column not in columns:
            cur.execute(statement)
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_preprocessed_record ON preprocessed_texts(record_type, raw_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_preprocessed_parent_post_id ON preprocessed_texts(parent_post_id)"
    )
    conn.commit()
    conn.close()


def seed_default_users():
    if not db.session.get(User, "admin"):
        admin = User(
            username="admin",
            name="Ana Reyes",
            email="admin@mana.ph",
            role="Admin",
            status="Active",
        )
        admin.set_password("admin2026")
        db.session.add(admin)

    if not db.session.get(User, "admin_mana"):
        analyst = User(
            username="admin_mana",
            name="LGU Analyst",
            email="lgu.analyst@mana.ph",
            role="LGU Analyst",
            status="Active",
        )
        analyst.set_password("mana2026!")
        db.session.add(analyst)
    db.session.commit()


def seed_settings():
    for section, payload in DEFAULT_SETTINGS.items():
        setting = db.session.get(SystemSetting, section)
        if not setting:
            setting = SystemSetting(section=section)
            setting.set_payload(payload)
            db.session.add(setting)
    db.session.commit()


def ensure_post_priority_table():
    """Create post_priorities table for RF predictions if it does not yet exist."""
    db_path = app.instance_path + "\\mana.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS post_priorities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id VARCHAR(128) NOT NULL UNIQUE,
            priority_label VARCHAR(32) NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            high_probability REAL NOT NULL DEFAULT 0.0,
            medium_probability REAL NOT NULL DEFAULT 0.0,
            low_probability REAL NOT NULL DEFAULT 0.0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
        """
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "ix_post_priorities_post_id ON post_priorities(post_id)"
    )
    conn.commit()
    conn.close()


def ensure_database():
    with app.app_context():
        db.create_all()
        if "sqlite" in _db_url:
            # These helpers patch missing columns in an existing SQLite DB.
            # PostgreSQL uses db.create_all() above, which handles everything.
            ensure_user_columns()
            ensure_preprocessed_text_columns()
            ensure_sentiment_columns()
            ensure_post_cluster_label_columns()
            ensure_post_priority_table()
        seed_clusters()
        seed_default_users()
        seed_settings()
    _retrain_if_stale()


def _retrain_if_stale():
    """Spawn a background thread to retrain CorEx + re-predict if anchor words changed.
    No-ops immediately when no model exists or when the hash matches.
    Server starts and serves requests without waiting for this thread."""
    import threading

    def _worker():
        try:
            from services.corex.topic_modeler import (
                is_model_stale, train_corex, predict_topics_batch,
            )
            from models import PostTopic, PreprocessedText, db  # noqa: F811

            if not is_model_stale():
                return

            print("[MANA] Anchor words changed — retraining CorEx in background...", flush=True)

            with app.app_context():
                rows = (
                    PreprocessedText.query
                    .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
                    .filter(PreprocessedText.final_tokens_json != "[]")
                    .all()
                )
                texts = [" ".join(r.final_tokens) for r in rows if r.final_tokens]
                post_ids = [r.raw_id for r in rows if r.final_tokens]

                if not texts:
                    print("[MANA] No preprocessed posts found — skipping auto-retrain.", flush=True)
                    return

                train_corex(texts)
                print("[MANA] CorEx retrained. Running predictions...", flush=True)

                batch_results = predict_topics_batch(texts)
                inserted = skipped = 0
                for post_id, topic_list in zip(post_ids, batch_results):
                    PostTopic.query.filter_by(post_id=post_id).delete()
                    for item in topic_list:
                        db.session.add(PostTopic(
                            post_id=post_id,
                            topic_label=item["topic"],
                            confidence=item["confidence"],
                        ))
                        inserted += 1
                    if not topic_list:
                        skipped += 1
                db.session.commit()

                # Correct Post.cluster_id for posts whose top CorEx topic changed.
                from data import TOPIC_TO_CLUSTER
                from models import Post
                cluster_updates = 0
                for post_id, topic_list in zip(post_ids, batch_results):
                    if not topic_list:
                        continue
                    top_topic = max(topic_list, key=lambda x: x["confidence"])
                    new_cluster_id = TOPIC_TO_CLUSTER.get(top_topic["topic"])
                    if not new_cluster_id:
                        continue
                    post = db.session.get(Post, post_id)
                    if post and post.cluster_label_source != "reviewed" and post.cluster_id != new_cluster_id:
                        post.cluster_id = new_cluster_id
                        post.cluster_label_source = "corex_enriched"
                        cluster_updates += 1
                if cluster_updates:
                    db.session.commit()

                print(
                    f"[MANA] Auto-retrain complete. "
                    f"{inserted} topic rows inserted, {skipped} posts unclassified, "
                    f"{cluster_updates} cluster_id corrections applied.",
                    flush=True,
                )
        except Exception as exc:
            import sys
            print(f"[MANA] Auto-retrain failed: {exc}", file=sys.stderr, flush=True)

    threading.Thread(target=_worker, daemon=True).start()


# Run at module load so gunicorn also initialises the DB on startup.
try:
    ensure_database()
except Exception as _db_err:
    import sys
    print(f"[MANA] WARNING: ensure_database() failed: {_db_err}", file=sys.stderr)
    print("[MANA] App will start, but the database may not be initialised.", file=sys.stderr)
    print("[MANA] Check that DATABASE_URL is set correctly in your environment.", file=sys.stderr)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
