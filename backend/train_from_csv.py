"""
Standalone training script — Random Forest (priority) + SVM (topic/cluster).

Usage:
    cd backend
    python train_from_csv.py --csv "C:/path/to/posts_rows_manual_topic_labels.csv"

Reads manual ground-truth labels from the CSV, preprocesses captions with the
existing preprocessing pipeline, then trains and saves:
  - SVM (topic categorisation):  svm_classifier.pkl, tfidf_vectorizer.pkl,
                                  label_binarizer.pkl, svm_meta.json, svm_report.json
  - RF  (priority):              rf_classifier.pkl, rf_feature_columns.json,
                                  rf_meta.json

Does NOT require a running database.  CorEx is untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Path setup ─────────────────────────────────────────────────────────────────
# Script lives in backend/; add it to sys.path so imports resolve.
_BACKEND = Path(__file__).parent
sys.path.insert(0, str(_BACKEND))

from preprocessing import preprocess_record
from services.svm.cluster_classifier import train_svm
from services.random_forest.priority_classifier import (
    DEFAULT_FEATURE_COLUMNS,
    DISASTER_TERMS,
    PRIORITY_LABELS,
    RF_COLUMNS_PATH,
    RF_META_PATH,
    RF_MODEL_PATH,
    SENTIMENT_ENCODE,
    _TOPIC_NAMES,
)

# ── Category → cluster mapping ─────────────────────────────────────────────────
# Maps the 12 human-annotated manual_topic_category values to the 8 NDRRMC
# response cluster IDs used by the SVM.  None = skip this row for SVM training.
CATEGORY_TO_CLUSTER: dict[str, str | None] = {
    "Air Quality Advisory":                     "cluster-b",
    "Fire Incident":                            "cluster-g",
    "Flood Control / Drainage Infrastructure":  "cluster-d",
    "Flood Prevention / Sanitation Operations": "cluster-b",
    "Heat Advisory / Heat Mitigation":          "cluster-b",
    "Medical Emergency Response":               "cluster-b",
    "Power Advisory / Grid Alert":              "cluster-e",
    "Preparedness / Coordination Activity":     "cluster-f",
    "Volcanic Activity / Ashfall":              "cluster-c",
    "Water Service Interruption / Repair Advisory": "cluster-b",
    "Weather Forecast":                         None,
    "Non-Disaster / General Information":       None,
}

# Maps the same categories to CorEx topic label names used as RF one-hot features.
# Must match _TOPIC_NAMES in priority_classifier.py exactly.
CATEGORY_TO_TOPIC: dict[str, str | None] = {
    "Air Quality Advisory":                     "health_medical",
    "Fire Incident":                            "rescue",
    "Flood Control / Drainage Infrastructure":  "logistics",
    "Flood Prevention / Sanitation Operations": "health_medical",
    "Heat Advisory / Heat Mitigation":          "health_medical",
    "Medical Emergency Response":               "health_medical",
    "Power Advisory / Grid Alert":              "telecom_power",
    "Preparedness / Coordination Activity":     "education",
    "Volcanic Activity / Ashfall":              "evacuation",
    "Water Service Interruption / Repair Advisory": "telecom_power",
    "Weather Forecast":                         None,
    "Non-Disaster / General Information":       None,
}

PRIORITY_LABEL_MAP = {"High": "High", "Medium": "Medium", "Low": "Low"}


# ── CSV loader ─────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"[CSV] Loaded {len(rows)} rows from {path}")
    return rows


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_rows(rows: list[dict]) -> list[dict]:
    """Run the full MANA preprocessing pipeline on each row's caption.

    Reuses preprocess_record() so tokens are identical to what the live
    pipeline stores in PreprocessedText.final_tokens_json.
    """
    # One shared translator to avoid recreating the GoogleTranslator per row.
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="en")
    except Exception:
        translator = None
        print("[WARN] deep_translator not available — translation skipped.")

    results = []
    skipped = 0
    for row in rows:
        caption = (row.get("caption") or "").strip()
        result = preprocess_record(
            raw_id=row.get("id", ""),
            item={"caption": caption, "text": caption},
            record_type="post",
            translator=translator,
        )
        # Store preprocessing result alongside the original row.
        row["_preprocess"] = result
        if result["preprocessing_status"] != "processed" or not result["final_tokens"]:
            skipped += 1
        results.append(row)

    processed = len(rows) - skipped
    print(f"[Preprocess] {processed} processed, {skipped} skipped (empty/irrelevant)")
    return results


# ── VADER scoring ──────────────────────────────────────────────────────────────

def compute_vader_scores(rows: list[dict]) -> list[dict]:
    """Add VADER scores to each row using the clean_text from preprocessing."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    except ImportError:
        print("[WARN] vaderSentiment not installed — using sentiment_compound from CSV.")
        vader = None

    for row in rows:
        pre = row["_preprocess"]
        text = pre.get("vader_text") or pre.get("clean_text") or row.get("caption") or ""

        if vader:
            s = vader.polarity_scores(text)
            row["_vader"] = {
                "compound": s["compound"],
                "positive": s["pos"],
                "negative": s["neg"],
                "neutral":  s["neu"],
            }
        else:
            # Fall back to CSV compound; approximate pos/neg/neu.
            compound = float(row.get("sentiment_compound") or 0.0)
            abs_c = abs(compound)
            if compound >= 0.05:
                row["_vader"] = {"compound": compound, "positive": abs_c, "negative": 0.0, "neutral": 1.0 - abs_c}
            elif compound <= -0.05:
                row["_vader"] = {"compound": compound, "positive": 0.0, "negative": abs_c, "neutral": 1.0 - abs_c}
            else:
                row["_vader"] = {"compound": compound, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    return rows


# ── SVM training ───────────────────────────────────────────────────────────────

def run_svm_training(rows: list[dict]) -> dict:
    """Build (texts, labels) pairs from manual_topic_category and train SVM."""
    svm_texts: list[str] = []
    svm_labels: list[list[str]] = []
    skipped_no_cluster = skipped_no_tokens = 0

    for row in rows:
        cat = row.get("manual_topic_category", "").strip()
        cluster = CATEGORY_TO_CLUSTER.get(cat)
        if cluster is None:
            skipped_no_cluster += 1
            continue

        pre = row["_preprocess"]
        final_tokens = pre.get("final_tokens") or []
        text = " ".join(final_tokens)
        if not text.strip():
            skipped_no_tokens += 1
            continue

        svm_texts.append(text)
        svm_labels.append([cluster])

    print(f"\n[SVM] Training corpus: {len(svm_texts)} posts")
    print(f"      Skipped (no cluster mapping): {skipped_no_cluster}")
    print(f"      Skipped (empty tokens):       {skipped_no_tokens}")

    label_dist = Counter(l[0] for l in svm_labels)
    print("      Cluster distribution:")
    for k, v in sorted(label_dist.items()):
        print(f"        {k}: {v}")

    if len(svm_texts) < 20:
        raise ValueError(
            f"Too few SVM training samples ({len(svm_texts)}). Need at least 20."
        )

    result = train_svm(svm_texts, svm_labels)
    print(f"\n[SVM] Done — f1_macro={result['f1_macro']:.4f}, best_C={result['best_C']}")
    print("      Per-cluster F1:")
    for label, metrics in result.get("per_class_report", {}).items():
        print(f"        {label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} (n={metrics['support']})")
    return result


# ── RF feature matrix ──────────────────────────────────────────────────────────

def _build_rf_features(rows: list[dict]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build (X, feature_columns, y) for RF training.

    All 105 rows are used (including Non-Disaster) since manual_priority is
    available for every row.
    """
    valid_rows = [
        r for r in rows
        if PRIORITY_LABEL_MAP.get(r.get("manual_priority", "").strip())
    ]
    if not valid_rows:
        raise ValueError("No rows with valid manual_priority found.")

    # Batch-level engagement for quartile bucketing.
    eng_raw = np.array([
        float(r.get("reactions") or 0)
        + float(r.get("comments") or 0)
        + float(r.get("shares") or 0)
        + float(r.get("reposts") or 0)
        for r in valid_rows
    ], dtype=float)

    q25 = float(np.quantile(eng_raw, 0.25))
    q75 = float(np.quantile(eng_raw, 0.75))
    if q25 == q75:
        eng_levels = np.ones(len(valid_rows))
    else:
        eng_levels = np.where(eng_raw > q75, 2.0, np.where(eng_raw <= q25, 0.0, 1.0))

    # Batch-level topic frequency for recurrence_frequency feature.
    topic_counter: Counter = Counter()
    for r in valid_rows:
        cat = r.get("manual_topic_category", "").strip()
        topic = CATEGORY_TO_TOPIC.get(cat)
        if topic:
            topic_counter[topic] += 1
    batch_total = max(len(valid_rows), 1)

    X_rows = []
    y_labels = []

    for i, row in enumerate(valid_rows):
        vader = row["_vader"]
        compound  = float(vader["compound"])
        positive  = float(vader["positive"])
        negative  = float(vader["negative"])
        neutral   = float(vader["neutral"])

        sent_label = (
            "Positive" if compound >= 0.05
            else "Negative" if compound <= -0.05
            else "Neutral"
        )
        sent_enc = float(SENTIMENT_ENCODE.get(sent_label, 1))

        reactions = float(row.get("reactions") or 0)
        comments  = float(row.get("comments")  or 0)
        shares    = float(row.get("shares")    or 0)
        reposts   = float(row.get("reposts")   or 0)
        eng       = eng_raw[i]
        eng_lvl   = eng_levels[i]

        pre = row["_preprocess"]
        clean = (pre.get("clean_text") or row.get("caption") or "").lower()
        keyword_intensity = float(sum(1 for term in DISASTER_TERMS if term in clean))

        cat = row.get("manual_topic_category", "").strip()
        topic_label = CATEGORY_TO_TOPIC.get(cat)
        topic_list  = [topic_label] if topic_label else []

        recurrence_frequency = (
            sum(topic_counter[t] / batch_total for t in topic_list) / len(topic_list)
            if topic_list else 0.0
        )

        topic_set = set(topic_list)
        topic_one_hot = [float(name in topic_set) for name in _TOPIC_NAMES]

        feature_row = [
            compound, positive, negative, neutral, sent_enc,
            reactions, comments, shares, reposts,
            eng, eng_lvl,
            keyword_intensity,
            recurrence_frequency,
        ] + topic_one_hot

        X_rows.append(feature_row)
        y_labels.append(PRIORITY_LABEL_MAP[row["manual_priority"].strip()])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_labels)
    return X, DEFAULT_FEATURE_COLUMNS, y


def run_rf_training(rows: list[dict]) -> dict:
    """Train RandomForestClassifier from CSV features and save model files."""
    X, feature_columns, y = _build_rf_features(rows)

    priority_dist = Counter(y)
    print(f"\n[RF] Training corpus: {len(y)} posts")
    print("     Priority distribution:")
    for k, v in sorted(priority_dist.items()):
        print(f"       {k}: {v}")

    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) > 1 and all(c >= 2 for c in counts)
    if len(y) < 5:
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
        "corpus_size":        int(len(y)),
        "n_estimators":       100,
        "accuracy":           round(accuracy, 4),
        "class_distribution": class_dist,
        "feature_columns":    feature_columns,
        "source":             "train_from_csv.py",
    }
    RF_META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\n[RF] Done — accuracy={accuracy:.4f} on {len(y_test)}-post holdout")
    print("     Per-class report:")
    for lbl in PRIORITY_LABELS:
        m = report.get(lbl, {})
        print(
            f"       {lbl}: P={m.get('precision', 0):.3f} "
            f"R={m.get('recall', 0):.3f} "
            f"F1={m.get('f1-score', 0):.3f} "
            f"(n={int(m.get('support', 0))})"
        )
    return {**meta, "report": report}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SVM + RF from labeled CSV")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to posts_rows_manual_topic_labels.csv",
    )
    parser.add_argument(
        "--skip-svm", action="store_true", help="Skip SVM training"
    )
    parser.add_argument(
        "--skip-rf",  action="store_true", help="Skip RF training"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MANA — CSV-based model training")
    print("=" * 60)

    rows = load_csv(args.csv)
    rows = preprocess_rows(rows)
    rows = compute_vader_scores(rows)

    svm_result = rf_result = None

    if not args.skip_svm:
        svm_result = run_svm_training(rows)

    if not args.skip_rf:
        rf_result = run_rf_training(rows)

    print("\n" + "=" * 60)
    print("Training complete.  Model files saved to backend/models/")
    if svm_result:
        print(f"  SVM f1_macro : {svm_result['f1_macro']:.4f}")
    if rf_result:
        print(f"  RF  accuracy : {rf_result['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
