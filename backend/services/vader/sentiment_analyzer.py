"""
VADER Sentiment Analyzer — Stage 5 of the MANA ML pipeline.

Classifies preprocessed post text into sentiment scores and labels.
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — no training required.

Responsibilities:
- analyze_sentiment: raw VADER compound/pos/neg/neu + label for a text
- compound_to_score: maps VADER compound (-1..+1) → legacy distress int (0-100)
- check_sarcasm_incongruence: positive compound + inherently negative cluster → flag
- check_thread_deviation: comment deviates from thread mean by > 1.5 std → flag
- analyze_post: high-level helper combining sentiment + sarcasm, used by routes and import
- get_status: introspection helper for /vader/status endpoint
"""

from __future__ import annotations

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Singleton (VADER lexicon loaded once at import time) ──────────────────────
_analyzer = SentimentIntensityAnalyzer()

# Domain-specific lexicon injection: Teach VADER that these words are heavy
# distress signals in a disaster context, even if normally neutral.
DISASTER_LEXICON = {
    "stranded": -3.2,
    "trapped": -3.5,
    "sos": -4.0,
    "rescue": -1.5,   # Inherently implies distress
    "missing": -3.0,
    "fire": -2.8,
    "blackout": -2.5,
    "outage": -2.2,
    "submerged": -3.0,
    "flash flood": -3.0,
    "sos calls": -3.5,
    "help us": -3.0,
    "please help": -2.5,
}
_analyzer.lexicon.update(DISASTER_LEXICON)

# MANA cluster IDs that represent inherently negative disaster contexts.
NEGATIVE_CLUSTERS = {
    "cluster-d",   # Logistics — blocked roads/bridges are objectively negative
    "cluster-e",   # Telecom/Power — outages are objectively negative
    "cluster-h",   # MDM — no genuinely positive news about death/missing
    "cluster-g",   # SRR — rescue calls are almost always negative (distress)
}

# ── Sarcasm detection phrases ──────────────────────────────────────────────────
_SARCASM_PHRASES = frozenset({
    "oh great", "oh wow", "how wonderful", "how amazing", "how nice",
    "how fantastic", "how lovely", "how convenient", "how helpful",
    "yeah right", "of course it is", "oh perfect", "just perfect",
    "absolutely perfect", "so great", "so wonderful", "so amazing",
    "how great", "how perfect", "how nice of",
    "congrats", "well done", "keep it up", "nice one", "good job",
    "thank you so much", "thanks for everything", "god bless you",
    "salamat sa lahat", "napaka galing",
})

def _check_sarcasm_phrases(text: str) -> bool:
    """Thesis sarcasm rule 3: exclamatory positive phrases in disaster context."""
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _SARCASM_PHRASES)

# ── Core functions ─────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Run VADER on text. Returns compound, positive, negative, neutral, and label.
    Thresholds (per thesis spec):
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        else              → Neutral
    """
    s = _analyzer.polarity_scores(text or "")
    c = s["compound"]
    return {
        "compound": round(c, 4),
        "positive": round(s["pos"], 4),
        "negative": round(s["neg"], 4),
        "neutral":  round(s["neu"], 4),
        "label":    "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"),
    }


def compound_to_score(compound: float) -> int:
    """
    Map VADER compound (-1..+1) to the legacy distress scale (int 0-100) used by
    Post.sentiment_score and score_tone() in data.py.

    Formula: max(20, min(round(60 - compound * 40), 97))

    Alignment with score_tone() thresholds:
        compound = +1.0  →  score = 20  →  "positive"  (score < 60)
        compound =  0.0  →  score = 60  →  "neutral"   (60 <= score < 80)
        compound = -0.5  →  score = 80  →  "negative"  (score >= 80)
        compound = -1.0  →  score = 97  →  "negative"  (clamped at 97)
    """
    return max(20, min(round(60 - compound * 40), 97))


def check_sarcasm_incongruence(compound: float, cluster_id: str | None) -> bool:
    """
    Thesis sarcasm rule 1: positive sentiment in an inherently negative disaster cluster.
    Returns True when compound >= 0.05 AND cluster_id is one of the negative clusters.
    """
    return compound >= 0.05 and cluster_id in NEGATIVE_CLUSTERS


def check_thread_deviation(
    comment_compound: float,
    thread_compounds: list[float],
    threshold: float = 1.5,
) -> bool:
    """
    Thesis sarcasm rule 2: comment deviates from the thread's mean compound by
    more than `threshold` standard deviations.

    Returns False when fewer than 3 thread compounds are available (insufficient data).
    """
    if len(thread_compounds) < 3:
        return False
    std = float(np.std(thread_compounds))
    if std == 0:
        return False
    return abs(comment_compound - float(np.mean(thread_compounds))) / std > threshold


def analyze_post(
    text: str,
    cluster_id: str | None,
    thread_compounds: list[float] | None = None,
) -> dict:
    """
    High-level helper combining analyze_sentiment + all three sarcasm rules.

    Returns a dict ready to be stored in PostSentiment and used to update
    Post.sentiment_score and Post.sentiment_compound.

    Keys: compound, positive, negative, neutral, label,
          sentiment_score (int 0-100), sarcasm_flag (bool)

    thread_compounds: sibling comment compounds for Rule 2 (thread deviation).
    Omit or pass None for posts (Rule 2 fires only with ≥ 3 data points).
    """
    result = analyze_sentiment(text)
    result["sentiment_score"] = compound_to_score(result["compound"])
    result["sarcasm_flag"] = (
        check_sarcasm_incongruence(result["compound"], cluster_id)
        or check_thread_deviation(result["compound"], thread_compounds or [])
        or _check_sarcasm_phrases(text)
    )
    return result


def analyze_post_with_comments(
    text: str,
    cluster_id: str | None,
    comment_texts: list[str] | None = None,
) -> dict:
    """
    Analyze a post plus its comments as one thread-level sentiment signal.

    No comment sentiment rows are stored. Each usable comment contributes one
    VADER observation to the post's existing sentiment row and sentiment_score.
    """
    comments = [value for value in (comment_texts or []) if value and value.strip()]
    post_result = analyze_sentiment(text)
    comment_results = [analyze_sentiment(value) for value in comments]
    observations = [post_result, *comment_results]

    compound = sum(item["compound"] for item in observations) / len(observations)
    positive = sum(item["positive"] for item in observations) / len(observations)
    negative = sum(item["negative"] for item in observations) / len(observations)
    neutral = sum(item["neutral"] for item in observations) / len(observations)
    comment_compounds = [item["compound"] for item in comment_results]

    result = {
        "compound": round(compound, 4),
        "positive": round(positive, 4),
        "negative": round(negative, 4),
        "neutral": round(neutral, 4),
        "label": "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral"),
    }
    result["sentiment_score"] = compound_to_score(result["compound"])
    result["sarcasm_flag"] = (
        check_sarcasm_incongruence(result["compound"], cluster_id)
        or check_thread_deviation(post_result["compound"], comment_compounds)
        or _check_sarcasm_phrases(text)
        or any(_check_sarcasm_phrases(value) for value in comments)
    )
    result["comments_analyzed"] = len(comments)
    return result


def get_status() -> dict:
    """Introspection helper for the /vader/status endpoint."""
    try:
        _analyzer.polarity_scores("test")
        available = True
    except Exception:
        available = False
    return {
        "available": available,
        "negative_clusters": sorted(NEGATIVE_CLUSTERS),
        "compound_thresholds": {"positive": 0.05, "negative": -0.05},
        "thread_deviation_threshold": 1.5,
        "score_formula": "max(20, min(round(60 - compound * 40), 97))",
    }
