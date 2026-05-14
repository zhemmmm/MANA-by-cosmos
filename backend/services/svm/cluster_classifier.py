"""
Linear SVM (One-vs-Rest) Cluster Classifier — Stage 4 of the MANA ML pipeline.

Classifies preprocessed posts into the 8 NDRRMC response clusters (A–H).
Uses TF-IDF features + LinearSVC with OvR strategy and GridSearch for C tuning.

Responsibilities:
- train_svm: trains on (texts, cluster_labels), saves model files to backend/models/
- predict_clusters: returns cluster assignments + confidence scores for a single text
- predict_clusters_batch: same for many texts at once
- is_model_trained / get_model_status: introspection helpers for routes
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

# ── Constants ─────────────────────────────────────────────────────────────────

CLUSTER_LABELS = [
    "cluster-a",
    "cluster-b",
    "cluster-c",
    "cluster-d",
    "cluster-e",
    "cluster-f",
    "cluster-g",
    "cluster-h",
]

CLUSTER_NAMES = {
    "cluster-a": "Food and Non-food Items (NFIs)",
    "cluster-b": "WASH, Medical and Public Health, Nutrition, Mental Health",
    "cluster-c": "Camp Coordination, Management and Protection (CCCM)",
    "cluster-d": "Logistics",
    "cluster-e": "Emergency Telecommunications (ETC)",
    "cluster-f": "Education",
    "cluster-g": "Search, Rescue and Retrieval (SRR)",
    "cluster-h": "Management of Dead and Missing (MDM)",
}

MIN_CORPUS_SIZE = 20
MAX_FEATURES = 5000
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_MIN_MARGIN = 0.08

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_SVM_PATH = os.path.join(_MODEL_DIR, "svm_classifier.pkl")
_TFIDF_PATH = os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl")
_BINARIZER_PATH = os.path.join(_MODEL_DIR, "label_binarizer.pkl")
_META_PATH = os.path.join(_MODEL_DIR, "svm_meta.json")
_REPORT_PATH = os.path.join(_MODEL_DIR, "svm_report.json")


# ── Training ──────────────────────────────────────────────────────────────────

def train_svm(texts: list[str], labels: list[list[str]]) -> dict:
    """
    Train a LinearSVC One-vs-Rest classifier on preprocessed text corpus.

    Args:
        texts:  Space-joined final_tokens strings, one per post.
        labels: Multi-label list per post, e.g. [["cluster-g"], ["cluster-a", "cluster-b"]].
                Single-label training (one cluster per post) is the typical starting point.

    Returns a summary dict with corpus_size, best_C, f1_macro, per_class_report, trained_at.
    Saves five files to backend/models/:
        svm_classifier.pkl, tfidf_vectorizer.pkl, label_binarizer.pkl,
        svm_meta.json, svm_report.json

    Raises ValueError if corpus is too small.
    """
    paired = [(t, l) for t, l in zip(texts, labels) if t and t.strip() and l]
    if len(paired) < MIN_CORPUS_SIZE:
        raise ValueError(
            f"Corpus too small: {len(paired)} labeled posts. "
            f"Need at least {MIN_CORPUS_SIZE} preprocessed posts with cluster labels to train SVM."
        )

    texts_clean, labels_clean = zip(*paired)
    texts_clean = list(texts_clean)
    labels_clean = list(labels_clean)

    mlb = MultiLabelBinarizer(classes=CLUSTER_LABELS)
    y = mlb.fit_transform(labels_clean)

    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    X = tfidf.fit_transform(texts_clean)

    # Stratified split on the primary (first) label for reproducibility.
    # test_size must be at least n_classes so every class has >= 1 test sample.
    primary_labels = [l[0] for l in labels_clean]
    n_classes = len(set(primary_labels))
    test_size = max(0.2, n_classes / len(texts_clean))
    
    # Stratify fails if any class has only 1 sample
    from collections import Counter
    class_counts = Counter(primary_labels)
    can_stratify = min(class_counts.values()) > 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=primary_labels if can_stratify else None
    )

    base_clf = OneVsRestClassifier(LinearSVC(max_iter=10000, class_weight="balanced"))
    param_grid = {"estimator__C": [0.1, 0.5, 1.0, 3.0]}

    # cv=min(5, smallest_class_count) to avoid folds with no positive samples
    min_class_count = int(np.min(y_train.sum(axis=0))) if y_train.shape[0] > 0 else 1
    cv_folds = max(2, min(5, min_class_count))

    grid = GridSearchCV(base_clf, param_grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    best_C = grid.best_params_["estimator__C"]

    y_pred = best_clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=CLUSTER_LABELS,
        output_dict=True,
        zero_division=0,
    )
    f1_macro = report.get("macro avg", {}).get("f1-score", 0.0)

    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(best_clf, _SVM_PATH)
    joblib.dump(tfidf, _TFIDF_PATH)
    joblib.dump(mlb, _BINARIZER_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(texts_clean),
        "best_C": best_C,
        "cv_folds": cv_folds,
        "f1_macro": round(f1_macro, 4),
        "max_features": MAX_FEATURES,
        "target": "f1_macro >= 0.75",
        "cluster_labels": CLUSTER_LABELS,
    }
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "corpus_size": len(texts_clean),
        "best_C": best_C,
        "f1_macro": round(f1_macro, 4),
        "per_class_report": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in CLUSTER_LABELS
            if label in report
        },
        "trained_at": meta["trained_at"],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def _load_model():
    if not is_model_trained():
        raise RuntimeError(
            "SVM model is not trained yet. "
            "Call POST /api/admin/svm/train first."
        )
    clf = joblib.load(_SVM_PATH)
    tfidf = joblib.load(_TFIDF_PATH)
    mlb = joblib.load(_BINARIZER_PATH)
    return clf, tfidf, mlb


def _decision_to_confidence(decision_scores: np.ndarray) -> np.ndarray:
    """
    Convert raw LinearSVC decision_function scores to [0, 1] confidence values
    using per-class sigmoid normalisation. This is a calibration approximation —
    scores > 0 mean the classifier leans "yes" for that class.
    """
    return 1.0 / (1.0 + np.exp(-decision_scores))


def predict_clusters(text: str) -> list[dict]:
    """
    Return cluster assignments for a single preprocessed text.

    Returns a list of dicts for clusters the model predicts as active
    (decision score > 0), sorted by confidence descending:
      [{"cluster_id": "cluster-g", "confidence": 0.84}, ...]

    Returns an empty list if no clusters are predicted.
    """
    clf, tfidf, mlb = _load_model()
    X = tfidf.transform([text or ""])

    # decision_function returns shape (1, n_classes) for OvR
    decision = clf.decision_function(X)[0]  # shape: (n_classes,)
    confidences = _decision_to_confidence(decision)

    results = []
    for cluster_id, score, conf in zip(CLUSTER_LABELS, decision, confidences):
        if conf > 0.60:
            results.append({"cluster_id": cluster_id, "confidence": round(float(conf), 4)})

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def predict_clusters_batch(texts: list[str]) -> list[list[dict]]:
    """
    Return cluster assignments for a list of preprocessed texts.

    Returns a list of lists — one inner list per input text.
    """
    if not texts:
        return []

    clf, tfidf, mlb = _load_model()
    cleaned = [t or "" for t in texts]
    X = tfidf.transform(cleaned)

    decisions = clf.decision_function(X)  # shape: (n_docs, n_classes)
    confidences = _decision_to_confidence(decisions)

    all_results = []
    for doc_decisions, doc_confidences in zip(decisions, confidences):
        doc_clusters = []
        for cluster_id, score, conf in zip(CLUSTER_LABELS, doc_decisions, doc_confidences):
            if conf > 0.60:
                doc_clusters.append({"cluster_id": cluster_id, "confidence": round(float(conf), 4)})
        doc_clusters.sort(key=lambda x: x["confidence"], reverse=True)
        all_results.append(doc_clusters)

    return all_results


def _scaled_thresholds() -> tuple[float, float]:
    """Return (min_confidence, min_margin) scaled to the trained corpus size.

    Smaller corpora produce less-calibrated decision scores, so we relax the
    thresholds to avoid discarding all SVM predictions on early datasets.
    Reads corpus_size from svm_meta.json; falls back to strict defaults if the
    file is missing (keeps existing behaviour when model isn't trained yet).
    """
    corpus_size = 0
    if os.path.exists(_META_PATH):
        try:
            with open(_META_PATH, encoding="utf-8") as f:
                corpus_size = json.load(f).get("corpus_size", 0)
        except Exception:
            pass
    if corpus_size < 50:
        return 0.55, 0.05
    if corpus_size < 200:
        return 0.60, 0.07
    return DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_MARGIN


def select_top_cluster(
    cluster_list: list[dict],
    min_confidence: float | None = None,
    min_margin: float | None = None,
) -> dict | None:
    """
    Return the top cluster only when the prediction is strong enough to trust.

    Thresholds scale automatically with the trained corpus size when not
    provided explicitly — smaller corpora use relaxed thresholds so SVM
    predictions are not discarded on early datasets.
    """
    if not cluster_list:
        return None
    if min_confidence is None or min_margin is None:
        _mc, _mm = _scaled_thresholds()
        min_confidence = min_confidence if min_confidence is not None else _mc
        min_margin = min_margin if min_margin is not None else _mm
    ranked = sorted(cluster_list, key=lambda item: item["confidence"], reverse=True)
    top = ranked[0]
    if top["confidence"] < min_confidence:
        return None
    second_conf = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    if (top["confidence"] - second_conf) < min_margin:
        return None
    return top


# ── Status helpers ────────────────────────────────────────────────────────────

def is_model_trained() -> bool:
    return (
        os.path.exists(_SVM_PATH)
        and os.path.exists(_TFIDF_PATH)
        and os.path.exists(_BINARIZER_PATH)
    )


def get_model_status() -> dict:
    if not is_model_trained():
        return {"trained": False}

    meta: dict = {}
    if os.path.exists(_META_PATH):
        with open(_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    report: dict = {}
    if os.path.exists(_REPORT_PATH):
        with open(_REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)

    per_class = {}
    for label in CLUSTER_LABELS:
        if label in report:
            per_class[label] = {
                "name": CLUSTER_NAMES.get(label, label),
                "precision": round(report[label].get("precision", 0.0), 4),
                "recall": round(report[label].get("recall", 0.0), 4),
                "f1": round(report[label].get("f1-score", 0.0), 4),
                "support": int(report[label].get("support", 0)),
            }

    return {
        "trained": True,
        "trained_at": meta.get("trained_at"),
        "corpus_size": meta.get("corpus_size"),
        "best_C": meta.get("best_C"),
        "f1_macro": meta.get("f1_macro"),
        "target": meta.get("target"),
        "cluster_labels": CLUSTER_LABELS,
        "per_class_report": per_class,
    }
