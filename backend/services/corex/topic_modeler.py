"""
CorEx (Anchored Correlation Explanation) Topic Modeler — Stage 3 of the MANA ML pipeline.

Responsibilities:
- train_corex: trains on preprocessed text corpus, saves model files to backend/models/
- predict_topics: returns topic assignments + confidence scores for a single text
- predict_topics_batch: same for many texts at once
- is_model_trained / get_model_status: introspection helpers for routes
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import corextopic.corextopic as ct

# ── Constants ─────────────────────────────────────────────────────────────────

# Topics are aligned 1-to-1 with the 8 NDRRMC disaster response clusters (A–H).
# Old "flood" and "communication" topics were removed — flood is an event type (not a
# response cluster) and communication overlapped with power_outage (both = cluster-e).
# "education" (cluster-f) and "dead_missing" (cluster-h) fill the two gaps that had
# no CorEx coverage, so all 8 NDRRMC clusters now have a dedicated topic.
TOPIC_LABELS = [
    "education",        # cluster-f
    "evacuation",       # cluster-c
    "rescue",           # cluster-g
    "logistics",        # cluster-d
    "relief",           # cluster-a
    "telecom_power",    # cluster-e
    "health_medical",   # cluster-b
    "dead_missing",     # cluster-h
]

# Anchor words aligned with NDRRMC disaster response clusters.
# Each list seeds the corresponding CorEx topic — CorEx will then expand
# these based on co-occurrence patterns in the actual corpus.
# Multi-word phrases are included; CountVectorizer uses ngram_range=(1,2) so bigrams work.
ANCHOR_WORDS: dict[str, list[str]] = {
    "education": [
        # core terms
        "school", "class", "suspension", "deped", "students", "learners",
        "class suspended", "school closure", "temporary classroom",
        "online class", "modular", "teacher", "academic",
        # class disruption
        "no classes", "class cancellation", "school cancelled",
        "school flooded", "school damaged", "school reopening",
        "classes resume", "class resumption",
        # alternative learning
        "self learning", "printed module", "gadget", "internet access",
        "virtual learning", "learning materials", "distance learning",
        "blended learning", "learning continuity",
        "learning continuity plan",
        # school infrastructure
        "school building", "school supplies", "classroom shortage",
        "enrollment", "academic calendar",
        "school as evacuation center", "school used as shelter",
        # expanded
        "no school", "classes cancelled", "remote learning",
        "face to face", "school damage", "learner device",
        "make up class", "school resumption", "school suspension",
        "class disruption", "deped advisory", "learning disruption",
    ],
    "evacuation": [
        # core terms
        "evacuation", "evacuate", "shelter", "evacuation center",
        "displaced", "evacuees", "camp", "evacuation site", "overcrowded",
        "safe space", "tent", "covered court",
        # facility types
        "gymnasium", "chapel", "multipurpose", "barangay hall",
        "astrodome", "sports complex", "welfare desk",
        # camp management
        "headcount", "curfew", "segregation", "overflow", "capacity",
        "displaced families", "privacy", "overcapacity", "relocated",
        # specific needs
        "sleeping mat", "portable toilet", "cooking area",
        "child friendly", "breastfeeding area", "preemptive evacuation",
        "mandatory evacuation", "return home", "decampment",
        "forced evacuation", "warning level", "danger zone",
        "high risk area", "low lying area", "flood prone",
        # expanded
        "rescue center", "evacuation notice", "mass evacuation",
        "night shelter", "community shelter", "safe haven",
        "stranded families", "relief center", "welfare center",
        "evacuation order", "temporary shelter", "displaced residents",
    ],
    "rescue": [
        # core terms
        "rescue", "trapped", "stranded", "search and rescue",
        "sos", "roof", "rescue boat", "save us",
        "help us", "helicopter", "coast guard",
        # fire rescue
        "fire", "arson", "blaze", "burning", "firefighter", "fire alert",
        "fire truck", "bureau of fire", "bfp", "put out fire",
        "engulfed in fire", "wildfire", "structure fire", "fire alarm",
        "two alarm", "three alarm", "four alarm", "five alarm",
        # operations
        "swift water", "rooftop", "second floor", "submerged",
        "pinned", "emergency", "distress", "search party",
        "usar", "extraction", "thermal", "call for help",
        # water rescue specifics
        "rising water", "chest deep", "neck deep", "waist deep",
        "inflatable boat", "kayak", "life vest", "rubber boat",
        # structural rescue
        "collapsed structure", "building collapse", "pinned under",
        "debris flow", "mudslide", "flash flood",
        "swept away", "washed away",
        # expanded
        "people trapped", "need rescue", "rescue needed",
        "rescue operation", "fire rescue", "flood rescue",
        "water rescue", "emergency rescue", "rescue team deployed",
        "rescue personnel", "requesting rescue", "please rescue",
        # fire alert specifics — ensures TXTFIRE-style posts land in rescue
        "fire out", "fire update", "fire incident", "fire response",
        "fire department", "active fire", "fire site", "fire reported",
        "fire call", "txtfire",
    ],
    "logistics": [
        # core terms
        "road", "bridge", "blocked", "landslide", "truck", "convoy",
        "delivery", "reroute", "passable", "road clearing", "warehouse",
        "transport", "debris", "collapsed", "infrastructure",
        "traffic", "car crash", "vehicular accident", "collision",
        # road conditions
        "road damage", "impassable", "alternate", "checkpoint", "detour",
        "sinkhole", "dpwh", "obstructed", "alternate route", "blocked road",
        "road network", "access road", "diversion",
        "road closed", "not passable", "road subsidence",
        "fallen tree", "road cut off",
        # vehicles and equipment
        "barge", "aerial", "fuel", "backhoe", "crane",
        "heavy equipment", "clearing operation",
        # supply chain
        "staging area", "distribution point", "supply chain",
        "chokepoint", "bottleneck",
        # expanded
        "road flooded", "supply route", "road condition",
        "passable road", "road blocked", "delivery route",
        "supply delivery", "logistics team", "cargo",
        "access blocked", "route blocked", "road update",
    ],
    "relief": [
        # core terms — use multi-word phrases to avoid bleeding into fire/rescue posts
        "relief goods", "relief donation", "food donation", "food pack",
        "relief distribution", "relief operations", "relief convoy",
        "packed relief", "relief goods for", "relief pack", "relief items",
        "relief package", "relief drive",
        # specific food and non-food items
        "rice", "noodles", "sardines", "canned goods", "water sachet",
        "family pack", "drinking water", "tarpaulin", "sleeping mat",
        "mosquito net", "jerry can", "water container", "blanket",
        "hygiene kit", "water refill",
        # personal care
        "diaper", "baby food", "formula milk", "soap", "toothbrush",
        "feminine hygiene", "sanitary napkin",
        # distribution operations — specific phrases only
        "repacking", "food distribution", "hot meal", "community kitchen",
        "supply drop", "aid package", "food assistance",
        "non food items", "nfi", "ready to eat",
        # organizations involved in relief
        "dswd", "red cross",
    ],
    "telecom_power": [
        # core terms
        "blackout", "power outage", "electricity",
        "brownout", "no power", "signal", "network",
        "no signal", "internet", "cell site", "radio", "connectivity", "telecom",
        # providers
        "pldt", "globe", "smart", "meralco",
        # equipment
        "generator", "solar", "inverter", "power bank", "transformer",
        "utility pole", "antenna", "satellite", "charging station",
        "flashlight", "emergency light", "backup power",
        # outage specifics
        "tower", "repair", "restoration", "dead zone",
        "no electricity", "network outage",
        "power restoration", "signal restored",
        "communication cut", "no connectivity",
        "fiber optic", "cable", "wire",
        # expanded
        "no internet", "signal loss", "power cut",
        "grid down", "communication blackout", "telecoms down",
        "network down", "power failure", "electricity cut",
        "phone dead", "no signal area", "mobile data down",
    ],
    "health_medical": [
        # core terms
        "hospital", "injured", "medical", "health", "sick",
        "medicine", "doctor", "patient", "nurse",
        "clinic", "fever", "dehydration", "sanitation", "wound",
        "water outage", "no water", "maynilad", "manila water",
        # post-flood diseases
        "leptospirosis", "diarrhea", "cholera", "dengue",
        "infection", "outbreak", "illness", "respiratory",
        "pneumonia", "measles", "hepatitis", "tetanus",
        # emergency medical
        "trauma", "ambulance", "first aid", "treatment",
        "drowning", "heat stroke", "snake bite",
        "field hospital", "medical volunteers", "triage",
        "water-borne disease", "contaminated water",
        # mental health and support
        "mental health", "counseling", "psychosocial",
        # nutrition and maternal
        "malnutrition", "vaccination", "midwife", "prenatal",
        "pharmacy", "medical team", "medical mission",
        # expanded
        "medical supply", "medical shortage", "health risk",
        "water safety", "disease outbreak", "health emergency",
        "medication", "medicine shortage", "health team",
        "medical aid", "sick evacuee", "health advisory",
    ],
    "dead_missing": [
        # core terms — deliberately free of "fire", "swept away", "retrieval"
        # to avoid pulling in fire-alert or active-rescue posts
        "missing", "missing person", "fatality", "casualty", "dead",
        "body", "unaccounted", "family tracing",
        "coordination desk", "remains",
        "found dead", "body found", "death", "confirmed dead",
        "declared dead",
        # confirmed/official death & missing language
        "confirmed fatality", "death toll confirmed", "bodies recovered",
        "remains identified", "victim identified", "declared missing",
        "officially missing", "confirmed missing", "reported dead",
        "death certificate issued", "buried", "interment",
        "post mortem", "cadaver",
        # identification procedures
        "ante mortem", "death toll", "death certificate",
        "morgue", "autopsy", "burial", "cremation",
        "dental record", "dna", "fingerprint", "next of kin",
        # missing person specifics
        "victim", "deceased", "disappeared", "fatalities",
        "missing child", "lost contact",
        "last seen", "wearing", "description", "photograph",
        "death notice",
        # expanded
        "person missing", "still missing", "reported missing",
        "death report", "missing family", "casualty report",
        "whereabouts unknown", "last contact", "has not returned",
        "feared dead", "search for missing", "missing after flood",
    ],
}




N_TOPICS = len(TOPIC_LABELS)
MIN_CORPUS_SIZE = 10
ANCHOR_STRENGTH = 3
MAX_FEATURES = 5000

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_COREX_MODEL_PATH = os.path.join(_MODEL_DIR, "corex_model.pkl")
_VECTORIZER_PATH = os.path.join(_MODEL_DIR, "corex_vectorizer.pkl")
_KEYWORDS_PATH = os.path.join(_MODEL_DIR, "corex_keywords.json")
_META_PATH = os.path.join(_MODEL_DIR, "corex_meta.json")


def _anchor_fingerprint() -> str:
    """SHA-256 hash of ANCHOR_WORDS + TOPIC_CONFIDENCE_THRESHOLDS.
    Saved to corex_meta.json at training time so the server can detect when
    source-code anchor changes make the on-disk model stale."""
    import hashlib
    import json as _json
    payload = {"anchors": ANCHOR_WORDS, "thresholds": TOPIC_CONFIDENCE_THRESHOLDS}
    raw = _json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def is_model_stale() -> bool:
    """Return True if ANCHOR_WORDS/thresholds changed since last training run.
    Returns False (not stale) if no model exists — let the admin train manually.
    Used by app.py startup to decide whether to auto-retrain."""
    if not is_model_trained():
        return False
    if not os.path.exists(_META_PATH):
        return True
    try:
        with open(_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("anchor_fingerprint") != _anchor_fingerprint()
    except Exception:
        return True


# Per-topic minimum confidence for a topic assignment to be stored.
# Higher thresholds for high-consequence or easily-confused topics.
TOPIC_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "education":      0.55,
    "evacuation":     0.55,
    "rescue":         0.60,  # fire posts must land here; slightly raised above others
    "logistics":      0.55,
    "relief":         0.55,
    "telecom_power":  0.55,
    "health_medical": 0.55,
    "dead_missing":   0.68,  # strictest — false positives here are very visible and misleading
}

# ── Iterative training constants ───────────────────────────────────────────────
MIN_CLUSTER_CONFIDENCE: float = 0.58   # exported; used by routes/posts.py
LOW_COHERENCE_THRESHOLD: float = 2.5  # per-topic coherence below this = boost anchors next pass
MAX_ENRICHED_KEYWORDS: int = 150       # cap on keyword list per topic
N_WORDS_PER_TOPIC: int = 30            # words pulled from CorEx per pass for expansion


# ── Internal helpers ───────────────────────────────────────────────────────────

def _train_single_pass(
    relevant: list[str],
    anchors_dict: dict[str, list[str]],
    anchor_strength: int = ANCHOR_STRENGTH,
) -> tuple:
    """Returns (model, vectorizer, coherence_dict, discovered_keywords_dict). No disk I/O."""
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, binary=True, ngram_range=(1, 2))
    doc_term_matrix = vectorizer.fit_transform(relevant)
    vocab = list(vectorizer.get_feature_names_out())
    vocab_set = set(vocab)

    anchor_indices = [
        [vocab.index(w) for w in anchors_dict.get(label, []) if w in vocab_set]
        for label in TOPIC_LABELS
    ]
    model = ct.Corex(n_hidden=N_TOPICS, seed=42)
    model.fit(doc_term_matrix, words=vocab, anchors=anchor_indices, anchor_strength=anchor_strength)

    coherence_dict = {label: float(score) for label, score in zip(TOPIC_LABELS, model.tcs)}

    discovered: dict[str, list[str]] = {}
    if len(relevant) >= 200:
        for i, label in enumerate(TOPIC_LABELS):
            top_words = model.get_topics(topic=i, n_words=N_WORDS_PER_TOPIC, print_words=True)
            discovered[label] = [w for w, mi, sign in top_words if sign == 1 and mi > 0]
    else:
        discovered = {label: [] for label in TOPIC_LABELS}

    return model, vectorizer, coherence_dict, discovered


def _build_enriched_anchors(
    current_anchors: dict[str, list[str]],
    discovered: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge discovered positive-MI words into current anchors, deduped, capped per topic."""
    enriched: dict[str, list[str]] = {}
    for label in TOPIC_LABELS:
        seen: set[str] = set()
        merged: list[str] = []
        for w in current_anchors.get(label, []) + discovered.get(label, []):
            w_clean = w.strip().lower()
            if w_clean and w_clean not in seen:
                seen.add(w_clean)
                merged.append(w_clean)
            if len(merged) >= MAX_ENRICHED_KEYWORDS:
                break
        enriched[label] = merged
    return enriched


# ── Training ───────────────────────────────────────────────────────────────────

def train_iteratively(
    texts: list[str],
    max_iterations: int = 5,
    target_coherence: float = 3.0,
) -> dict:
    """
    Train CorEx repeatedly in memory, enriching anchors after each pass.
    Only the best model (highest mean coherence) is written to disk at the end.
    Returns per-iteration diagnostics + the saved keyword set.

    Termination (first satisfied wins):
      1. overall_coherence >= target_coherence
      2. max_iterations exhausted
      3. Coherence improvement < 0.01 vs previous best (plateau guard)
    """
    relevant = [t for t in texts if t and t.strip()]
    if len(relevant) < MIN_CORPUS_SIZE:
        raise ValueError(
            f"Corpus too small: {len(relevant)} documents. "
            f"Need at least {MIN_CORPUS_SIZE} preprocessed posts to train CorEx."
        )

    current_anchors = {label: list(words) for label, words in ANCHOR_WORDS.items()}
    iteration_results = []
    best_model = best_vectorizer = None
    best_coherence_val = -1.0
    best_iteration = 0
    best_keywords: dict[str, list[str]] = {}
    prev_coherence = -1.0

    for iteration in range(1, max_iterations + 1):
        strength = min(ANCHOR_STRENGTH + (iteration - 1), 6)

        model, vectorizer, coherence_dict, discovered = _train_single_pass(
            relevant, current_anchors, anchor_strength=strength
        )
        overall = float(np.mean(list(coherence_dict.values())))
        low_topics = [lbl for lbl, sc in coherence_dict.items() if sc < LOW_COHERENCE_THRESHOLD]

        iteration_results.append({
            "iteration": iteration,
            "overall_coherence": round(overall, 6),
            "coherence_scores": {k: round(v, 6) for k, v in coherence_dict.items()},
            "low_coherence_topics": low_topics,
            "anchor_strength_used": strength,
            "anchor_count": {lbl: len(current_anchors.get(lbl, [])) for lbl in TOPIC_LABELS},
        })

        if overall > best_coherence_val:
            best_model, best_vectorizer = model, vectorizer
            best_coherence_val = overall
            best_iteration = iteration
            best_keywords = _build_enriched_anchors(current_anchors, discovered)

        if overall >= target_coherence:
            break
        if iteration > 1 and (overall - prev_coherence) < 0.01:
            break

        prev_coherence = overall
        enriched = _build_enriched_anchors(current_anchors, discovered)
        for label in low_topics:
            current_anchors[label] = enriched.get(label, current_anchors[label])

    # Save only the winning model
    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, _COREX_MODEL_PATH)
    joblib.dump(best_vectorizer, _VECTORIZER_PATH)
    with open(_KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_keywords, f, indent=2)

    trained_at = datetime.now(timezone.utc).isoformat()
    best_scores = iteration_results[best_iteration - 1]["coherence_scores"]
    meta = {
        "trained_at": trained_at,
        "corpus_size": len(relevant),
        "n_topics": N_TOPICS,
        "anchor_strength": ANCHOR_STRENGTH,
        "max_features": MAX_FEATURES,
        "coherence_scores": best_scores,
        "overall_coherence": best_coherence_val,
        "low_coherence_topics": iteration_results[best_iteration - 1]["low_coherence_topics"],
        "iterative_training": True,
        "best_iteration": best_iteration,
        "total_iterations": len(iteration_results),
        "anchor_fingerprint": _anchor_fingerprint(),
    }
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "iterations": iteration_results,
        "best_iteration": best_iteration,
        "best_overall_coherence": round(best_coherence_val, 6),
        "corpus_size": len(relevant),
        "final_keywords": best_keywords,
        "trained_at": trained_at,
    }


def train_corex(texts: list[str]) -> dict:
    """
    Train Anchored CorEx on a corpus of preprocessed texts.

    Returns a summary dict with expanded keywords, coherence scores, and
    training metadata. Saves four files to backend/models/:
      corex_model.pkl, corex_vectorizer.pkl, corex_keywords.json, corex_meta.json

    Raises ValueError if the corpus is too small to train on.
    """
    relevant = [t for t in texts if t and t.strip()]
    if len(relevant) < MIN_CORPUS_SIZE:
        raise ValueError(
            f"Corpus too small: {len(relevant)} documents. "
            f"Need at least {MIN_CORPUS_SIZE} preprocessed posts to train CorEx."
        )

    model, vectorizer, coherence_scores, discovered = _train_single_pass(
        relevant, ANCHOR_WORDS, anchor_strength=ANCHOR_STRENGTH
    )

    # Always include core anchor words; only expand from corpus if large enough
    expanded_keywords: dict[str, list[str]] = {
        label: list(ANCHOR_WORDS.get(label, [])) for label in TOPIC_LABELS
    }
    if len(relevant) >= 200:
        for label in TOPIC_LABELS:
            for w in discovered.get(label, []):
                if w not in expanded_keywords[label]:
                    expanded_keywords[label].append(w)

    low_coherence = [
        label for label, score in coherence_scores.items() if score < LOW_COHERENCE_THRESHOLD
    ]

    os.makedirs(_MODEL_DIR, exist_ok=True)
    joblib.dump(model, _COREX_MODEL_PATH)
    joblib.dump(vectorizer, _VECTORIZER_PATH)
    with open(_KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(expanded_keywords, f, indent=2)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(relevant),
        "n_topics": N_TOPICS,
        "anchor_strength": ANCHOR_STRENGTH,
        "max_features": MAX_FEATURES,
        "coherence_scores": coherence_scores,
        "overall_coherence": float(np.mean(list(coherence_scores.values()))),
        "low_coherence_topics": low_coherence,
        "anchor_fingerprint": _anchor_fingerprint(),
    }
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "corpus_size": len(relevant),
        "expanded_keywords": expanded_keywords,
        "coherence_scores": coherence_scores,
        "overall_coherence": meta["overall_coherence"],
        "low_coherence_topics": low_coherence,
        "trained_at": meta["trained_at"],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def _load_model():
    if not is_model_trained():
        raise RuntimeError(
            "CorEx model is not trained yet. "
            "Call POST /api/admin/corex/train first."
        )
    model = joblib.load(_COREX_MODEL_PATH)
    vectorizer = joblib.load(_VECTORIZER_PATH)
    return model, vectorizer


def predict_topics(text: str) -> list[dict]:
    """
    Return topic assignments for a single preprocessed text.

    Returns a list of dicts for topics the model considers active (p > 0.5):
      [{"topic": "flood", "confidence": 0.87}, ...]

    Returns an empty list if no topics are detected.
    """
    model, vectorizer = _load_model()
    doc_term = vectorizer.transform([text or ""])

    # corextopic 1.1: use transform() which returns p_y_given_x (posterior probs, not log-probs)
    probs = model.transform(doc_term)[0]  # shape: (n_topics,), values in [0, 1]

    results = []
    for label, prob in zip(TOPIC_LABELS, probs):
        threshold = TOPIC_CONFIDENCE_THRESHOLDS.get(label, 0.65)
        if prob > threshold:
            results.append({"topic": label, "confidence": round(float(prob), 4)})

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def predict_topics_batch(texts: list[str]) -> list[list[dict]]:
    """
    Return topic assignments for a list of preprocessed texts.

    Returns a list of lists — one inner list per input text.
    """
    if not texts:
        return []

    model, vectorizer = _load_model()
    cleaned = [t or "" for t in texts]
    doc_term = vectorizer.transform(cleaned)

    # corextopic 1.1: transform() returns p_y_given_x, shape (n_docs, n_topics)
    probs = model.transform(doc_term)  # shape: (n_docs, n_topics)

    all_results = []
    for doc_probs in probs:
        doc_topics = []
        for label, prob in zip(TOPIC_LABELS, doc_probs):
            threshold = TOPIC_CONFIDENCE_THRESHOLDS.get(label, 0.65)
            if prob > threshold:
                doc_topics.append({"topic": label, "confidence": round(float(prob), 4)})
        doc_topics.sort(key=lambda x: x["confidence"], reverse=True)
        all_results.append(doc_topics)

    return all_results


# ── Status helpers ────────────────────────────────────────────────────────────

def is_model_trained() -> bool:
    return os.path.exists(_COREX_MODEL_PATH) and os.path.exists(_VECTORIZER_PATH)


def get_model_status() -> dict:
    if not is_model_trained():
        return {"trained": False}

    meta: dict = {}
    if os.path.exists(_META_PATH):
        with open(_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    keywords: dict = {}
    if os.path.exists(_KEYWORDS_PATH):
        with open(_KEYWORDS_PATH, encoding="utf-8") as f:
            keywords = json.load(f)

    return {
        "trained": True,
        "trained_at": meta.get("trained_at"),
        "corpus_size": meta.get("corpus_size"),
        "n_topics": meta.get("n_topics", N_TOPICS),
        "overall_coherence": meta.get("overall_coherence"),
        "coherence_scores": meta.get("coherence_scores", {}),
        "low_coherence_topics": meta.get("low_coherence_topics", []),
        "topic_labels": TOPIC_LABELS,
        "expanded_keywords": keywords,
    }
