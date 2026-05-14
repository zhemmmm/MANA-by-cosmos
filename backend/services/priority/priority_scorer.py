"""
Hybrid Priority Scorer — blends Random Forest ML probabilities with
real-time heuristic signals (cluster urgency, volume, location, recency).

This module does NOT read or write any database column directly.
All DB writes are handled by the pipeline route that calls these functions.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# ── Configuration constants ────────────────────────────────────────────────────

CLUSTER_URGENCY_WEIGHTS: dict[str, float] = {
    "cluster-g": 1.00,  # Search, Rescue & Retrieval
    "cluster-h": 0.95,  # Dead & Missing
    "cluster-b": 0.85,  # Health / Medical
    "cluster-d": 0.70,  # Logistics
    "cluster-a": 0.65,  # Food / NFIs
    "cluster-c": 0.60,  # CCCM / Evacuation
    "cluster-e": 0.50,  # Telecom / Power
    "cluster-f": 0.35,  # Education
}

HIGH_SEVERITY_KEYWORDS = [
    "sos", "trapped", "dead", "blood", "critical", "help us", "rescue needed", 
    "drowning", "body found", "fatal", "emergency", "save us", "submerged",
    "pinned", "casualty", "missing person", "heart attack", "unconscious"
]

MANILA_DISTRICTS = [
    "tondo", "binondo", "quiapo", "sampaloc", "malate", "ermita", "pandacan", 
    "sta. mesa", "santa mesa", "san nicolas", "san miguel", "port area", 
    "intramuros", "paco", "santa ana", "sta. ana"
]

PRIORITY_THRESHOLDS = {"high": 65.0, "medium": 40.0}

MAX_VOLUME_PER_HOUR = 20
MAX_LOCATION_POSTS = 10
LOCATION_WINDOW_HOURS = 1  # Updated to 1 hour
RECENCY_HALF_LIFE_HOURS = 6
TREND_SPIKE_RATIO = 3.0

# Blending weights: how much influence RF vs formula has on final score
RF_WEIGHT = 0.40
FORMULA_WEIGHT = 0.60


def as_utc_naive(value):
    if value is None:
        return None
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


# ── Core scoring ───────────────────────────────────────────────────────────────

def compute_priority_score(
    post,
    db_session,
    rf_probabilities: dict[str, float] | None = None,
    topic_count: int = 0,
    sarcasm_flag: bool = False,
    exaggeration_score: float = 0.0,
) -> float:
    """
    Return a priority score in [0.0, 100.0].

    Blends Random Forest ML output (40%) with real-time formula signals (60%).
    Does NOT read or write any DB column itself.
    """
    from models import Post  # deferred import to avoid circular dependency

    post_caption = (post.caption or "").lower()
    post_loc = (post.location or "").lower()

    # ── Component 1: Cluster urgency & Severity (30% of formula) ───────────
    cluster_w = CLUSTER_URGENCY_WEIGHTS.get(post.cluster_id, 0.50)
    
    # Severity Keyword Bonus
    severity_hits = sum(1 for kw in HIGH_SEVERITY_KEYWORDS if kw in post_caption)
    severity_bonus = min(severity_hits * 5.0, 15.0)  # up to 15 bonus points

    # ── Component 2: Sentiment intensity (25% of formula) ──────────────────
    compound = post.sentiment_compound or 0.0
    # Penalize negative sentiment more heavily. Positive sentiment implies safety.
    sentiment = abs(compound) if compound < 0 else abs(compound) * 0.5
    
    if sarcasm_flag and compound > 0:
        sentiment = max(sentiment, 0.7)

    # ── Component 3: Volume last 1h same cluster (20% of formula) ──────────
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    volume_count = (
        db_session.query(Post)
        .filter(
            Post.cluster_id == post.cluster_id,
            Post.date >= one_hour_ago,
            Post.is_relevant == True,  # noqa: E712
        )
        .count()
    )
    volume = min(volume_count / MAX_VOLUME_PER_HOUR, 1.0)

    # ── Component 4: Location Hotspot Manila 1h (15% of formula) ───────────
    location_score = 0.0
    hotspot_bonus = 0.0
    
    if post_loc and post_loc != "philippines":
        # Fetch recent posts to do in-memory fuzzy matching
        recent_posts = (
            db_session.query(Post.location, Post.caption)
            .filter(
                Post.date >= one_hour_ago,
                Post.is_relevant == True
            ).all()
        )
        
        matched_district = None
        for district in MANILA_DISTRICTS:
            if district in post_loc or district in post_caption:
                matched_district = district
                break
                
        loc_count = 0
        if matched_district:
            for r_loc, r_cap in recent_posts:
                r_loc_str = (r_loc or "").lower()
                r_cap_str = (r_cap or "").lower()
                if matched_district in r_loc_str or matched_district in r_cap_str:
                    loc_count += 1
        else:
            for r_loc, _ in recent_posts:
                if r_loc == post.location:
                    loc_count += 1
                    
        location_score = min(loc_count / MAX_LOCATION_POSTS, 1.0)
        
        # Manila Hotspot detection: >= 2 posts in the same area within 1 hour
        if loc_count >= 2:
            hotspot_bonus = 10.0

    # ── Component 5: Recency exponential decay (10% of formula) ────────────
    if post.date:
        age_h = max((now - post.date.replace(tzinfo=timezone.utc)).total_seconds() / 3600, 0)
    else:
        age_h = 24.0
    recency = math.exp(-age_h * math.log(2) / RECENCY_HALF_LIFE_HOURS)

    # ── Formula base score ─────────────────────────────────────────────────
    formula_base = (
        cluster_w * 0.30
        + sentiment * 0.25
        + volume * 0.20
        + location_score * 0.15
        + recency * 0.10
    ) * 100

    # ── RF base score ──────────────────────────────────────────────────────
    if rf_probabilities:
        rf_base = (
            rf_probabilities.get("High", 0.0) * 100
            + rf_probabilities.get("Medium", 0.0) * 50
            + rf_probabilities.get("Low", 0.0) * 0
        )
        base = (rf_base * RF_WEIGHT) + (formula_base * FORMULA_WEIGHT)
    else:
        base = formula_base

    # ── Additive bonuses ───────────────────────────────────────────────────
    base += exaggeration_score * 5.0
    base += severity_bonus
    base += hotspot_bonus

    # ── Topic multiplier ───────────────────────────────────────────────────
    multiplier = 1.0 + min(topic_count, 3) * 0.03

    return round(min(base * multiplier, 100.0), 1)


def assign_priority_label(score: float) -> str:
    """Map a numeric score to a human-readable priority label."""
    if score >= PRIORITY_THRESHOLDS["high"]:
        return "High"
    if score >= PRIORITY_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


# ── Cluster trend analysis ─────────────────────────────────────────────────────

def compute_cluster_trends(db_session) -> dict[str, dict]:
    """
    Advanced 4-factor Trend Meter (Velocity, Acceleration, Severity Density, Localization).
    Returns a dict keyed by cluster_id with trend metadata.
    """
    from models import Post

    now = as_utc_naive(datetime.now(timezone.utc))
    hour_1_ago = now - timedelta(hours=1)
    hour_2_ago = now - timedelta(hours=2)
    twenty_four_hours_ago = now - timedelta(hours=24)

    trends: dict[str, dict] = {}
    
    # Pre-fetch 24h baseline counts per cluster
    baseline_query = (
        db_session.query(Post.cluster_id)
        .filter(Post.date >= twenty_four_hours_ago, Post.is_relevant == True)
    ).all()
    
    baseline_counts = {k: 0 for k in CLUSTER_URGENCY_WEIGHTS}
    for row in baseline_query:
        if row[0] in baseline_counts:
            baseline_counts[row[0]] += 1

    for cluster_id in CLUSTER_URGENCY_WEIGHTS:
        # Fetch posts from the last 2 hours for this cluster
        recent_posts = (
            db_session.query(Post)
            .filter(
                Post.cluster_id == cluster_id,
                Post.date >= hour_2_ago,
                Post.is_relevant == True,
            ).all()
        )
        
        posts_h1 = [p for p in recent_posts if as_utc_naive(p.date) and as_utc_naive(p.date) >= hour_1_ago]
        posts_h2 = [p for p in recent_posts if as_utc_naive(p.date) and as_utc_naive(p.date) < hour_1_ago]
        
        vol_h1 = len(posts_h1)
        vol_h2 = len(posts_h2)
        
        # 1. Velocity (Current 1h vs 24h baseline)
        baseline_per_hour = max(baseline_counts[cluster_id] / 24.0, 0.1)
        velocity_ratio = vol_h1 / baseline_per_hour
        
        # 2. Acceleration (H1 vs H2)
        acceleration = vol_h1 - vol_h2
        
        # 3. Severity Density (Average priority of recent posts)
        avg_priority = 0.0
        if vol_h1 > 0:
            avg_priority = sum(
                (100 if p.priority == "High" else (50 if p.priority == "Medium" else 0)) 
                for p in posts_h1
            ) / vol_h1
            
        # 4. Localization (Are posts clustered in a specific Manila district?)
        max_district_concentration = 0.0
        if vol_h1 >= 2:
            district_counts = {d: 0 for d in MANILA_DISTRICTS}
            for p in posts_h1:
                loc = (p.location or "").lower()
                cap = (p.caption or "").lower()
                for d in MANILA_DISTRICTS:
                    if d in loc or d in cap:
                        district_counts[d] += 1
            
            top_district_count = max(district_counts.values()) if district_counts else 0
            max_district_concentration = top_district_count / vol_h1  # 0.0 to 1.0

        # Compute Final Trend Score (0 - 100 Meter)
        # Base from velocity (up to 40 pts)
        trend_score = min(velocity_ratio * 10, 40.0)
        # Add acceleration bonus (up to 20 pts)
        if acceleration > 0:
            trend_score += min(acceleration * 5, 20.0)
        # Add severity density (up to 20 pts)
        trend_score += (avg_priority / 100.0) * 20.0
        # Add localization bonus (up to 20 pts)
        trend_score += max_district_concentration * 20.0
        
        trend_score = round(min(trend_score, 100.0), 1)
        is_trending = trend_score >= 60.0

        trends[cluster_id] = {
            "cluster_id": cluster_id,
            "current_1h": vol_h1,
            "total_24h": baseline_counts[cluster_id],
            "baseline_per_hour": round(baseline_per_hour, 2),
            "ratio": round(velocity_ratio, 2),
            "is_trending": is_trending,
            "trend_score": trend_score,
            "acceleration": acceleration,
            "severity_density": round(avg_priority, 1),
            "localization_ratio": round(max_district_concentration, 2)
        }

    return trends
