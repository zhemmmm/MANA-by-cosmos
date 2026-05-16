"""
Add _seed_priority labels to all seed posts in seed_dataset.json.

Priority rules (disaster response context):
  High   — immediate threat to life, active rescue, fatalities, people trapped
  Medium — urgent but not life-threatening, needs response within hours
  Low    — informational, monitoring, no immediate action needed

Run from backend/: python scripts/label_seed_priorities.py
"""
import json
import re
from pathlib import Path

SEED_FILE = Path(__file__).parent.parent / "seed_dataset.json"

# ── Signal word sets ──────────────────────────────────────────────────────────

HIGH_SIGNALS = {
    # Active rescue / life threat
    "sos", "trapped", "stranded", "rooftop", "rescue", "rescue boat", "rescue team",
    "search and rescue", "extract", "extraction", "helicopter",
    # Fire / active danger
    "fire", "blaze", "burning", "arson", "nasunog", "sunog", "bfp", "txtfire",
    "firefighter", "wildfire", "structure fire",
    # Fatalities / missing
    "dead", "died", "death", "fatality", "fatalities", "casualty", "casualties",
    "confirmed dead", "body found", "death toll", "killed", "lifeless",
    "missing person", "missing persons", "nawala", "patay", "namatay",
    # Medical emergency
    "critical condition", "cardiac", "unconscious", "drowning", "drowned",
    "severe injury", "nalunod",
    # Immediate shelter / no food no water (urgent)
    "no food", "walang pagkain", "no water", "walang tubig", "starving",
    "gutom", "dehydrated", "naubusan",
    # Contaminated / dangerous now
    "contaminated water", "toxic", "hazardous",
    # Volcano / immediate
    "lava flow", "pyroclastic", "eruption", "ashfall alert", "alert level 4",
    "alert level 5", "permanent danger zone",
}

MEDIUM_SIGNALS = {
    # Evacuation / shelter status
    "evacuation center", "evacuation site", "evacu", "likas", "evacuees",
    "displaced", "evacuated", "naglikas", "nailikas",
    "overcrowded", "overcapacity", "overflow shelter",
    # Supply / relief en route or delayed
    "relief goods", "relief operations", "relief distribution", "ayuda",
    "food packs", "relief convoy", "supply convoy", "delayed",
    "supply shortage", "shortage", "kulang",
    # Water advisory (happening now or very soon)
    "water interruption", "water supply cut", "boil water", "no water supply",
    "water tanker", "tanker truck", "water rationing",
    # Road / logistics blockage
    "road blocked", "road closed", "impassable", "blocked highway",
    "reroute", "alternate route", "convoy delay",
    # Medical supply shortage (not emergency)
    "medicine shortage", "medical supply", "insulin", "bp meds", "gamot",
    "medical team needed", "kailangan ng doktor",
    # Comms degraded
    "signal loss", "no signal", "reduced signal", "backup comms",
    "communication disruption",
    # Flood / typhoon active (not immediate life threat)
    "flood warning", "storm surge warning", "typhoon signal",
    "baha", "umaapaw", "flash flood",
    # School closure
    "walang pasok", "class suspension", "school closure", "no classes",
}

LOW_SIGNALS = {
    # Informational updates / advisories
    "advisory", "update", "status update", "situation report", "sitrep",
    "monitoring", "under monitoring", "being monitored",
    # Scheduled / future
    "scheduled", "will be", "papunta", "darating", "inaasahan",
    "pre-positioned", "pre positioned", "standby", "on standby",
    # Learning continuity / education plan
    "learning continuity", "learning plan", "modular", "distance learning",
    "deped", "alternative learning",
    # Weather forecast / outlook
    "weather forecast", "weather outlook", "pagasa", "low pressure area",
    "weather update",
    # Inventory / goods ready
    "goods ready", "stocked", "inventory", "in stock", "available",
    "nakahanda", "handa na",
    # Air quality non-emergency
    "air quality index", "aqi", "good aqi", "moderate aqi",
    # Post-event / recovery
    "clearing operations", "road clearing", "debris clearing",
    "restoration", "being restored", "repaired", "assessment",
    "estimated restoration",
}


def score_priority(text: str, cluster: str, likes: int, shares: int, comments: int) -> str:
    lower = text.lower()
    engagement = likes + shares + comments

    # Count signal hits
    high_hits = sum(1 for s in HIGH_SIGNALS if s in lower)
    medium_hits = sum(1 for s in MEDIUM_SIGNALS if s in lower)
    low_hits = sum(1 for s in LOW_SIGNALS if s in lower)

    # Cluster-based bias (some clusters lean a certain direction)
    cluster_bias = {
        "cluster-g": ("High", 3),    # SRR — rescue posts lean High
        "cluster-h": ("High", 2),    # MDM — fatality/missing lean High
        "cluster-f": ("Low", 1),     # Education — lean Low/Medium
    }

    # Apply cluster bias as extra signal hits
    if cluster in cluster_bias:
        direction, weight = cluster_bias[cluster]
        if direction == "High":
            high_hits += weight
        elif direction == "Low":
            low_hits += weight

    # High engagement (>300 total) boosts toward High
    if engagement > 300:
        high_hits += 1
    elif engagement < 80:
        low_hits += 1

    # Decision: pick highest score, break ties toward Medium
    scores = {"High": high_hits, "Medium": medium_hits, "Low": low_hits}
    top = max(scores, key=lambda k: scores[k])

    # If all tied at 0, use cluster default
    if scores[top] == 0:
        cluster_defaults = {
            "cluster-g": "High",
            "cluster-h": "High",
            "cluster-a": "Medium",
            "cluster-b": "Medium",
            "cluster-c": "Medium",
            "cluster-d": "Medium",
            "cluster-e": "Medium",
            "cluster-f": "Low",
        }
        return cluster_defaults.get(cluster, "Medium")

    return top


def main():
    with SEED_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    from collections import Counter
    dist: Counter = Counter()

    for post in data:
        text = post.get("text") or ""
        cluster = post.get("_seed_cluster_id", "")
        likes = post.get("likes", 0) or 0
        shares = post.get("shares", 0) or 0
        comments = post.get("comments", 0) or 0
        priority = score_priority(text, cluster, likes, shares, comments)
        post["_seed_priority"] = priority
        dist[priority] += 1

    print("Priority distribution:")
    for p in ["High", "Medium", "Low"]:
        print(f"  {p}: {dist[p]}")
    print(f"  Total: {sum(dist.values())}")

    # Show a few examples per priority
    for target in ["High", "Medium", "Low"]:
        examples = [p for p in data if p["_seed_priority"] == target][:3]
        print(f"\n{target} examples:")
        for p in examples:
            print(f"  [{p['_seed_cluster_id']}] {p['text'][:100]}")

    with SEED_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(data)} posts with _seed_priority labels.")


if __name__ == "__main__":
    main()
