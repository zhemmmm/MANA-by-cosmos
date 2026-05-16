"""
Random Forest Priority Classifier — Stage 6 of the MANA ML pipeline.

Classifies preprocessed posts into: High, Medium, or Low priority.
Consumes VADER sentiment outputs, CorEx topic labels, and engagement metrics.

Responsibilities:
- _fetch_post_records: join Post, PostSentiment, PostTopic, PreprocessedText rows
- _build_feature_matrix: assemble (n_samples, n_features) numpy array
- train_rf: fit RandomForestClassifier on labeled posts, persist artifacts
- predict_priorities_batch: predict priority + probabilities for a list of post IDs
- is_model_trained / get_model_status: introspection helpers used by routes

Persistence:
  models/rf_classifier.pkl       — trained RandomForestClassifier
  models/rf_feature_columns.json — ordered feature column list (train/predict parity)
  models/rf_meta.json            — training metadata (accuracy, class distribution, etc.)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
# services/random_forest/ → services/ → backend/
_BASE = Path(__file__).parent.parent.parent
MODEL_DIR = _BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

RF_MODEL_PATH   = MODEL_DIR / "rf_classifier.pkl"
RF_COLUMNS_PATH = MODEL_DIR / "rf_feature_columns.json"
RF_META_PATH    = MODEL_DIR / "rf_meta.json"

# ── Label constants ────────────────────────────────────────────────────────────
PRIORITY_LABELS = ["High", "Medium", "Low"]

# Map heuristic Post.priority values → RF training labels
LABEL_MAP = {
    "Critical":   "High",
    "High":       "High",
    "Moderate":   "Medium",
    "Monitoring": "Low",
}

# Map RF labels → Post.severity_rank
SEVERITY_MAP = {"High": 3, "Medium": 2, "Low": 1}

# ── Feature schema ─────────────────────────────────────────────────────────────
# Order must be stable — this list is saved to rf_feature_columns.json so that
# training and prediction always use identical column positions.
DEFAULT_FEATURE_COLUMNS = [
    "vader_compound",
    "vader_positive",
    "vader_negative",
    "vader_neutral",
    "sentiment_encoded",
    "reactions",
    "comments",
    "shares",
    "reposts",
    "engagement_score",
    "engagement_level",
    "keyword_intensity",
    "recurrence_frequency",
    "topic_education",
    "topic_evacuation",
    "topic_rescue",
    "topic_logistics",
    "topic_relief",
    "topic_telecom_power",
    "topic_health_medical",
    "topic_dead_missing",
]

# CorEx topic names in feature-column order — must match TOPIC_LABELS in
# services/corex/topic_modeler.py so that PostTopic rows produce non-zero features.
_TOPIC_NAMES = [
    "education",
    "evacuation",
    "rescue",
    "logistics",
    "relief",
    "telecom_power",
    "health_medical",
    "dead_missing",
]

SENTIMENT_ENCODE = {"Negative": 0, "Neutral": 1, "Positive": 2}

# Disaster-related keywords for keyword_intensity computation.
# Covers English disaster terms + common Filipino/Tagalog equivalents.
DISASTER_TERMS = frozenset({
    "urgent", "alert", "critical", "danger", "stranded", "rescue", "sos",
    "evacuate", "evacuation", "warning", "lagnat", "hospital", "trapped",
    "ashfall", "volcano", "flood", "baha", "tubig", "bagyo", "typhoon",
    "landslide", "casualty", "missing", "fatality", "injured", "injury",
    "damage", "destroyed", "likas", "sagipin", "saklolo", "putik",
    "pagguho", "collapsed", "blocked", "fire", "sunog", "nasunog",
    "submerged", "overflow", "flash", "relief", "ayuda", "dead", "killed",
    "medical", "rescue_team", "rescue_boat", "missing_person", "rooftop",
    "trapped_family", "inundation", "debris", "shelter", "displaced",
})


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_post_records(post_ids: list[str]) -> list[dict]:
    """
    Join Post, PostSentiment, PostTopic, and PreprocessedText for each post ID.
    Returns a flat list of dicts ready for feature engineering.
    Posts with no matching Post row are silently skipped.
    """
    from models import Post, PostSentiment, PostTopic, PreprocessedText

    posts = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}

    sentiments = {
        s.post_id: s
        for s in PostSentiment.query.filter(PostSentiment.post_id.in_(post_ids)).all()
    }

    topics_map: dict[str, list[str]] = {}
    for row in PostTopic.query.filter(PostTopic.post_id.in_(post_ids)).all():
        topics_map.setdefault(row.post_id, []).append(row.topic_label)

    clean_texts = {
        row.raw_id: row.clean_text
        for row in (
            PreprocessedText.query
            .filter(PreprocessedText.raw_id.in_(post_ids))
            .filter_by(record_type="post")
            .all()
        )
        if row.clean_text
    }

    records = []
    for pid in post_ids:
        post = posts.get(pid)
        if not post:
            continue

        sent = sentiments.get(pid)
        compound = sent.compound if sent else 0.0
        positive = sent.positive if sent else 0.0
        negative = sent.negative if sent else 0.0
        neutral  = sent.neutral  if sent else 1.0
        sent_label = (
            "Positive" if compound >= 0.05
            else "Negative" if compound <= -0.05
            else "Neutral"
        )

        records.append({
            "post_id":        pid,
            "priority_label": LABEL_MAP.get(post.priority, "Medium"),
            "compound":       compound,
            "positive":       positive,
            "negative":       negative,
            "neutral":        neutral,
            "sentiment_label": sent_label,
            "reactions":      post.reactions or 0,
            "comments":       post.comments  or 0,
            "shares":         post.shares    or 0,
            "reposts":        post.reposts   or 0,
            "clean_text":     clean_texts.get(pid) or post.caption or "",
            "topic_labels":   topics_map.get(pid, []),
        })
    return records


# ── Feature engineering ────────────────────────────────────────────────────────

def _build_feature_matrix(
    records: list[dict],
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a list of post record dicts into a (n_samples, n_features) float array.

    feature_columns:
      - None (train mode): use DEFAULT_FEATURE_COLUMNS and return it
      - list  (predict mode): align output columns to the saved training order,
        filling any missing columns with 0

    Engineered features:
      engagement_score  = reactions + comments + shares + reposts
      engagement_level  = 0 (bottom 25%) | 1 (middle 50%) | 2 (top 25%)
      keyword_intensity = count of DISASTER_TERMS in clean_text
      recurrence_frequency = mean topic frequency across the current batch
    """
    if not records:
        cols = feature_columns or DEFAULT_FEATURE_COLUMNS
        return np.zeros((0, len(cols)), dtype=float), cols

    # Compute engagement scores for all records at once (needed for batch quantiles)
    eng = np.array(
        [r["reactions"] + r["comments"] + r["shares"] + r["reposts"] for r in records],
        dtype=float,
    )
    q25 = float(np.quantile(eng, 0.25))
    q75 = float(np.quantile(eng, 0.75))
    if q25 == q75:
        # No spread — assign all as medium to avoid boundary artifacts
        eng_levels = np.ones(len(records))
    else:
        eng_levels = np.where(eng > q75, 2.0, np.where(eng <= q25, 0.0, 1.0))

    # Batch-level topic frequencies for recurrence_frequency
    topic_counter: Counter = Counter()
    for r in records:
        topic_counter.update(r["topic_labels"])
    batch_total = max(len(records), 1)

    rows = []
    for i, rec in enumerate(records):
        text_lower = (rec.get("clean_text") or "").lower()
        keyword_intensity = sum(1 for term in DISASTER_TERMS if term in text_lower)

        tl = rec["topic_labels"]
        recurrence_frequency = (
            sum(topic_counter[t] / batch_total for t in tl) / len(tl)
            if tl else 0.0
        )

        topic_set = set(tl)
        row = [
            float(rec.get("compound",  0.0)),
            float(rec.get("positive",  0.0)),
            float(rec.get("negative",  0.0)),
            float(rec.get("neutral",   0.0)),
            float(SENTIMENT_ENCODE.get(rec.get("sentiment_label", "Neutral"), 1)),
            float(rec.get("reactions", 0)),
            float(rec.get("comments",  0)),
            float(rec.get("shares",    0)),
            float(rec.get("reposts",   0)),
            float(eng[i]),
            float(eng_levels[i]),
            float(keyword_intensity),
            float(recurrence_frequency),
        ] + [float(name in topic_set) for name in _TOPIC_NAMES]
        rows.append(row)

    X = np.array(rows, dtype=float)

    if feature_columns is None:
        return X, DEFAULT_FEATURE_COLUMNS

    # Predict mode — reorder/pad columns to match the saved training schema
    col_idx = {col: i for i, col in enumerate(DEFAULT_FEATURE_COLUMNS)}
    aligned = np.zeros((len(rows), len(feature_columns)), dtype=float)
    for j, col in enumerate(feature_columns):
        if col in col_idx:
            aligned[:, j] = X[:, col_idx[col]]
    return aligned, feature_columns


# ── Public API ─────────────────────────────────────────────────────────────────

def is_model_trained() -> bool:
    return RF_MODEL_PATH.exists() and RF_COLUMNS_PATH.exists()


def get_model_status() -> dict:
    if not is_model_trained():
        return {"trained": False}
    meta: dict = {}
    if RF_META_PATH.exists():
        try:
            meta = json.loads(RF_META_PATH.read_text())
        except Exception:
            pass
    return {"trained": True, **meta}


def train_rf(post_ids: list[str] | None = None) -> dict:
    """
    Train the Random Forest classifier from labeled posts in the database.

    Labels come from Post.priority (heuristic or reviewed), mapped to
    High / Medium / Low via LABEL_MAP.  This mirrors how SVM bootstraps
    from heuristic cluster labels.

    post_ids: specific IDs to train on. When None, all relevant preprocessed
              posts are used.

    Returns a metadata dict (also persisted to rf_meta.json).
    """
    from models import PreprocessedText

    if post_ids is None:
        rows = (
            PreprocessedText.query
            .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
            .filter(PreprocessedText.final_tokens_json != "[]")
            .all()
        )
        post_ids = [r.raw_id for r in rows]

    if not post_ids:
        raise ValueError("No eligible posts found for RF training.")

    records = _fetch_post_records(post_ids)
    if len(records) < 3:
        raise ValueError(
            f"Need at least 3 posts for RF training, got {len(records)}."
        )

    y = np.array([rec["priority_label"] for rec in records])
    X, feature_columns = _build_feature_matrix(records)

    # Stratified split when every class has at least 2 samples
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) > 1 and all(c >= 2 for c in counts)
    if len(records) < 5:
        # Too small to split meaningfully — train and evaluate on full set
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=(y if can_stratify else None),
            )
        except ValueError:
            X_train, X_test, y_train, y_test = X, X, y, y

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    joblib.dump(clf, RF_MODEL_PATH)
    RF_COLUMNS_PATH.write_text(json.dumps(feature_columns))

    class_dist = {lbl: int(np.sum(y == lbl)) for lbl in PRIORITY_LABELS}
    meta = {
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "corpus_size":        len(records),
        "n_estimators":       100,
        "accuracy":           round(accuracy, 4),
        "class_distribution": class_dist,
        "feature_columns":    feature_columns,
    }
    RF_META_PATH.write_text(json.dumps(meta, indent=2))

    return {**meta, "report": report}


def predict_priorities_batch(post_ids: list[str]) -> list[dict]:
    """
    Predict priority labels for a list of post IDs.

    Returns a list of dicts:
      { post_id, priority, confidence, probabilities: {High, Medium, Low} }
    """
    if not is_model_trained():
        raise RuntimeError("RF model is not trained. Call train_rf() first.")

    clf: RandomForestClassifier = joblib.load(RF_MODEL_PATH)
    feature_columns: list[str] = json.loads(RF_COLUMNS_PATH.read_text())

    records = _fetch_post_records(post_ids)
    if not records:
        return []

    X, _ = _build_feature_matrix(records, feature_columns=feature_columns)
    proba = clf.predict_proba(X)
    preds = clf.predict(X)
    classes = list(clf.classes_)

    results = []
    for i, rec in enumerate(records):
        label = preds[i]
        proba_dict = {cls: round(float(proba[i][j]), 4) for j, cls in enumerate(classes)}
        results.append({
            "post_id":       rec["post_id"],
            "priority":      label,
            "confidence":    round(proba_dict.get(label, 0.0), 4),
            "probabilities": proba_dict,
        })
    return results
