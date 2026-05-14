"""
MANA Topic Categorization Sandbox — In-Memory Training
Trains CorEx on synthetic Taglish posts WITHOUT touching the database or production models.
Outputs: coherence comparison, accuracy report, and recommended anchor expansions.

Usage: python tests/sandbox_topic_training.py
"""
from __future__ import annotations
import sys, os, json, tempfile
from collections import Counter
from pathlib import Path

# Add backend to path so we can import preprocessing + corex modules
_BACKEND = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import corextopic.corextopic as ct

# Import preprocessing functions (no DB needed for these)
from preprocessing import (
    clean_text, tokenize_text, normalize_informal_tokens,
    should_translate, translate_text, apply_negation_handling,
    lemmatize_tokens, detect_bigrams, remove_stop_words,
    tokenize_preserving_apostrophes, clean_text_for_vader,
)
from services.corex.topic_modeler import (
    TOPIC_LABELS, ANCHOR_WORDS, ANCHOR_STRENGTH, MAX_FEATURES, N_TOPICS,
)
from sandbox_corpus import SANDBOX_POSTS


# ── Proposed expanded anchors ────────────────────────────────────────────────
EXPANDED_ANCHOR_WORDS: dict[str, list[str]] = {
    "education": [
        "school", "class", "suspension", "deped", "students", "learners",
        "walang pasok", "class suspended", "school closure", "temporary classroom",
        "school cancelled", "online class", "modular",
        # NEW
        "self learning", "printed module", "gadget", "internet access",
        "virtual learning", "teacher", "academic", "learning materials",
        "class cancellation", "school flooded", "no classes", "estudyante",
    ],
    "evacuation": [
        "evacuation", "evacuate", "shelter", "likas", "evacuation center",
        "displaced", "evacuees", "camp", "evacuation site", "overcrowded",
        "registration", "safe space", "tent", "covered court",
        # NEW
        "gymnasium", "chapel", "multipurpose", "headcount", "curfew",
        "segregation", "overflow", "capacity", "barangay hall",
        "displaced families", "privacy", "overcapacity", "relocated",
    ],
    "rescue": [
        "rescue", "trapped", "stranded", "sagipin", "search and rescue",
        "sos", "roof", "retrieval", "rescue boat", "save us",
        "tabang", "saklolo", "helicopter", "coast guard",
        # NEW
        "swift water", "rooftop", "second floor", "submerged",
        "pinned", "debris", "emergency", "distress", "zodiac",
        "search party", "usar", "rappel", "extraction", "thermal",
    ],
    "logistics": [
        "road", "bridge", "blocked", "landslide", "truck", "convoy",
        "delivery", "reroute", "passable", "road clearing", "warehouse",
        "transport", "debris", "collapsed", "guho", "infrastructure",
        # NEW
        "road damage", "impassable", "alternate", "checkpoint", "detour",
        "barge", "bangka", "aerial", "sinkhole", "dpwh",
        "fuel", "obstructed", "alternate route", "blocked road",
    ],
    "relief": [
        "relief", "ayuda", "goods", "donation", "food pack",
        "supply", "distribution", "relief goods", "rice", "bigas",
        "pagkain", "blanket", "hygiene kit", "water refill", "canned goods",
        # NEW
        "noodles", "sardines", "water sachet", "family pack",
        "repacking", "loading", "queue", "food", "aid",
        "nfi", "kit", "assistance", "relief pack", "drinking water",
    ],
    "telecom_power": [
        "blackout", "kuryente", "power outage", "electricity",
        "walang kuryente", "brownout", "no power", "signal", "network",
        "no signal", "internet", "cell site", "radio", "connectivity", "telecom",
        # NEW
        "pldt", "globe", "smart", "generator", "solar", "inverter",
        "tower", "repair", "restoration", "power bank", "dead zone",
        "no electricity", "network outage", "meralco",
    ],
    "health_medical": [
        "hospital", "injured", "medical", "health", "sick",
        "ospital", "medicine", "doctor", "patient", "nurse",
        "gamot", "clinic", "fever", "dehydration", "sanitation", "wound",
        # NEW
        "leptospirosis", "diarrhea", "cholera", "dengue",
        "trauma", "mental health", "counseling", "ambulance",
        "first aid", "infection", "outbreak", "illness", "treatment",
    ],
    "dead_missing": [
        "missing", "missing person", "fatality", "casualty", "dead",
        "body", "retrieval", "identified", "unaccounted", "family tracing",
        "body identified", "hospital list", "coordination desk", "remains",
        # NEW
        "ante mortem", "death toll", "death certificate", "registration",
        "victim", "deceased", "disappeared", "fatalities",
        "missing child", "lost contact", "nawawala", "patay",
    ],
}


# ── Preprocessing (in-memory, no DB) ─────────────────────────────────────────

def preprocess_text_standalone(raw_text: str) -> str:
    """Run the full MANA preprocessing pipeline on a single text, returning
    space-joined final tokens. No DB, no translator (skip translation to
    avoid quota burn — use normalized tokens directly)."""
    cleaned = clean_text(raw_text)
    tokens = tokenize_text(cleaned)
    normalized = normalize_informal_tokens(tokens)
    normalized_clean = " ".join(normalized)

    # For sandbox: skip Google Translate to avoid quota.
    # Use normalized tokens as the ML input (already mostly English after
    # informal normalization maps Taglish → English equivalents).
    translation_tokens = tokenize_preserving_apostrophes(normalized_clean)
    negation_tokens, _ = apply_negation_handling(translation_tokens)
    lemmatized, _ = lemmatize_tokens(negation_tokens)
    bigrams = detect_bigrams(lemmatized)
    final_tokens = remove_stop_words(lemmatized)
    for b in bigrams:
        if b not in final_tokens:
            final_tokens.append(b)
    return " ".join(final_tokens)


# ── CorEx training (in-memory) ───────────────────────────────────────────────

def train_corex_sandbox(
    texts: list[str],
    anchor_words: dict[str, list[str]],
    label: str = "model",
) -> tuple:
    """Train CorEx in memory. Returns (model, vectorizer, coherence_dict, expanded_kw)."""
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, binary=True, ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(texts)
    vocab = list(vectorizer.get_feature_names_out())
    vocab_set = set(vocab)

    anchor_indices = []
    for topic in TOPIC_LABELS:
        words = anchor_words.get(topic, [])
        indices = [vocab.index(w) for w in words if w in vocab_set]
        anchor_indices.append(indices)

    model = ct.Corex(n_hidden=N_TOPICS, seed=42)
    model.fit(dtm, words=vocab, anchors=anchor_indices, anchor_strength=ANCHOR_STRENGTH)

    coherence = {lbl: float(s) for lbl, s in zip(TOPIC_LABELS, model.tcs)}
    expanded = {}
    for i, lbl in enumerate(TOPIC_LABELS):
        top = model.get_topics(topic=i, n_words=20)
        expanded[lbl] = [w for w, _, sign in top if sign == 1]

    print(f"\n{'='*60}")
    print(f"  {label} — Coherence Scores")
    print(f"{'='*60}")
    for lbl in TOPIC_LABELS:
        bar = "#" * int(coherence[lbl])
        print(f"  {lbl:22s} {coherence[lbl]:7.3f}  {bar}")
    avg = np.mean(list(coherence.values()))
    print(f"  {'AVERAGE':22s} {avg:7.3f}")

    return model, vectorizer, coherence, expanded


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, vectorizer, texts, ground_truths):
    """Evaluate CorEx predictions against ground truth labels."""
    dtm = vectorizer.transform(texts)
    probs = model.transform(dtm)

    tp = Counter()
    fp = Counter()
    fn = Counter()
    correct = 0
    total = len(texts)

    for i, (doc_probs, gt_topics) in enumerate(zip(probs, ground_truths)):
        predicted = set()
        for j, (lbl, prob) in enumerate(zip(TOPIC_LABELS, doc_probs)):
            if prob > 0.5:
                predicted.add(lbl)
        gt_set = set(gt_topics)

        if predicted & gt_set:
            correct += 1

        for lbl in TOPIC_LABELS:
            if lbl in predicted and lbl in gt_set:
                tp[lbl] += 1
            elif lbl in predicted and lbl not in gt_set:
                fp[lbl] += 1
            elif lbl not in predicted and lbl in gt_set:
                fn[lbl] += 1

    print(f"\n{'='*60}")
    print(f"  Classification Report")
    print(f"{'='*60}")
    print(f"  {'Topic':22s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print(f"  {'-'*55}")
    f1_scores = []
    for lbl in TOPIC_LABELS:
        p = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0
        r = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
        f1_scores.append(f1)
        print(f"  {lbl:22s} {p:6.2f} {r:7.2f} {f1:6.2f} {tp[lbl]:4d} {fp[lbl]:4d} {fn[lbl]:4d}")
    print(f"  {'-'*55}")
    print(f"  {'MACRO AVG':22s} {np.mean(f1_scores):6.2f}")
    print(f"  Hit rate (>=1 correct): {correct}/{total} ({100*correct/total:.1f}%)")

    return {"tp": tp, "fp": fp, "fn": fn, "hit_rate": correct/total}


# ── Keyword diff ─────────────────────────────────────────────────────────────

def show_keyword_diff(old_expanded, new_expanded):
    """Show new keywords discovered by expanded anchors."""
    print(f"\n{'='*60}")
    print(f"  New Keywords Discovered (expanded - baseline)")
    print(f"{'='*60}")
    for lbl in TOPIC_LABELS:
        old_set = set(old_expanded.get(lbl, []))
        new_set = set(new_expanded.get(lbl, []))
        new_words = new_set - old_set
        if new_words:
            print(f"\n  {lbl}:")
            for w in sorted(new_words):
                print(f"    + {w}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  MANA Topic Categorization — Sandbox Training")
    print("  In-memory only. No DB. No production model changes.")
    print("="*60)

    # Step 1: Preprocess all sandbox posts
    print(f"\nPreprocessing {len(SANDBOX_POSTS)} synthetic posts...")
    texts = []
    ground_truths = []
    for post in SANDBOX_POSTS:
        processed = preprocess_text_standalone(post["text"])
        if processed.strip():
            texts.append(processed)
            ground_truths.append(post["topics"])

    print(f"  {len(texts)} posts preprocessed successfully")

    # Corpus stats
    cluster_counts = Counter(p["cluster"] for p in SANDBOX_POSTS)
    multi = sum(1 for p in SANDBOX_POSTS if len(p["topics"]) > 1)
    print(f"  Multi-label posts: {multi} ({100*multi/len(SANDBOX_POSTS):.0f}%)")
    for c in sorted(cluster_counts):
        print(f"    {c}: {cluster_counts[c]} posts")

    # Step 2: Train baseline (current anchors)
    print("\n" + "-"*60)
    print("  BASELINE: Training with current ANCHOR_WORDS")
    print("-"*60)
    base_model, base_vec, base_coherence, base_kw = train_corex_sandbox(
        texts, ANCHOR_WORDS, "BASELINE (current anchors)"
    )
    base_eval = evaluate_model(base_model, base_vec, texts, ground_truths)

    # Step 3: Train expanded (proposed anchors)
    print("\n" + "-"*60)
    print("  EXPANDED: Training with EXPANDED_ANCHOR_WORDS")
    print("-"*60)
    exp_model, exp_vec, exp_coherence, exp_kw = train_corex_sandbox(
        texts, EXPANDED_ANCHOR_WORDS, "EXPANDED (proposed anchors)"
    )
    exp_eval = evaluate_model(exp_model, exp_vec, texts, ground_truths)

    # Step 4: Comparison
    print(f"\n{'='*60}")
    print(f"  Coherence Comparison: BASELINE vs EXPANDED")
    print(f"{'='*60}")
    print(f"  {'Topic':22s} {'Base':>8s} {'Expanded':>9s} {'Delta':>7s}")
    print(f"  {'-'*50}")
    for lbl in TOPIC_LABELS:
        b = base_coherence[lbl]
        e = exp_coherence[lbl]
        d = e - b
        arrow = "+" if d > 0 else ("-" if d < 0 else "=")
        print(f"  {lbl:22s} {b:8.3f} {e:9.3f} {arrow}{abs(d):6.3f}")
    b_avg = np.mean(list(base_coherence.values()))
    e_avg = np.mean(list(exp_coherence.values()))
    d_avg = e_avg - b_avg
    arrow = "+" if d_avg > 0 else "-"
    print(f"  {'-'*50}")
    print(f"  {'AVERAGE':22s} {b_avg:8.3f} {e_avg:9.3f} {arrow}{abs(d_avg):6.3f}")

    # Step 5: Show new discovered keywords
    show_keyword_diff(base_kw, exp_kw)

    # Step 6: Output the expanded keywords JSON (for review, not auto-applied)
    print(f"\n{'='*60}")
    print(f"  Expanded Keywords JSON (for corex_keywords.json)")
    print(f"{'='*60}")
    print(json.dumps(exp_kw, indent=2))

    print(f"\n{'='*60}")
    print(f"  DONE — Review results above.")
    print(f"  No production files were modified.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
