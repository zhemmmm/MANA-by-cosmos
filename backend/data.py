"""
MANA — Shared data helpers
Cluster definitions plus lightweight heuristics for imported social posts.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from models import Cluster, db
from services.rules.decision_engine import evaluate_from_post

CLUSTER_DEFINITIONS = [
    {
        "id": "cluster-a",
        "short": "Cluster A",
        "name": "Food and Non-food Items (NFIs)",
        "description": "Tracks posts about food packs, water, hygiene kits, blankets, and other basic relief needs.",
        "keywords": ["relief goods", "rice", "water refill", "hygiene kit", "blanket", "food pack", "relief pack", "drinking water", "canned goods", "noodles", "water sachet", "nfi", "non-food items", "donation", "grocery"],
        "accent": "#f59e0b",
        "recommendation": "Dispatch rapid food and NFI validation support to the affected area within the next response cycle.",
    },
    {
        "id": "cluster-b",
        "short": "Cluster B",
        "name": "WASH, Medical and Public Health, Nutrition, Mental Health and Psychosocial Support (Health)",
        "description": "Tracks posts about health, medicine, clean water, nutrition, and mental health support.",
        "keywords": ["fever", "insulin", "washing area", "dehydration", "doctor", "medical team", "medicine", "health center", "clean water", "clinic", "nurse", "hospital", "wound", "injured", "leptospirosis", "diarrhea", "cholera", "first aid", "water outage", "no water"],
        "accent": "#3b82f6",
        "recommendation": "Coordinate a health sweep, water safety check, and medicine support with the nearest response unit.",
    },
    {
        "id": "cluster-c",
        "short": "Cluster C",
        "name": "Camp Coordination, Management and Protection (CCCM)",
        "description": "Tracks evacuation center crowding, camp services, registration, and protection issues.",
        "keywords": ["evacuation center", "overcapacity", "privacy", "registration", "safe space", "toilet line", "evacuation site", "evacuees", "shelter", "camp", "crowded", "displaced families", "housing", "gym", "covered court", "barangay hall", "tent", "evacuation area"],
        "accent": "#8b5cf6",
        "recommendation": "Coordinate immediate shelter protection adjustments, sanitation checks, and overflow site support.",
    },
    {
        "id": "cluster-d",
        "short": "Cluster D",
        "name": "Logistics",
        "description": "Tracks blocked routes, delivery delays, convoy movement, and supply transport issues.",
        "keywords": ["blocked road", "convoy", "truck", "warehouse", "delivery", "reroute", "road clearing", "bridge damage", "passable", "supply route", "transport", "blocked bridge", "impassable", "mudslide", "alternate route", "aerial delivery", "boat", "barge", "traffic", "car crash", "vehicular accident", "collision"],
        "accent": "#f97316",
        "recommendation": "Activate alternate routing and issue a field logistics advisory before dispatch resumes.",
    },
    {
        "id": "cluster-e",
        "short": "Cluster E",
        "name": "Emergency Telecommunications (ETC)",
        "description": "Tracks signal loss, network problems, and urgent communication needs.",
        "keywords": ["signal down", "no network", "power bank", "cell site", "radio", "connectivity", "no signal", "communication line", "internet down", "telecom", "network outage", "dead zone", "no wifi", "brownout", "blackout", "no power", "generator", "drrm radio"],
        "accent": "#06b6d4",
        "recommendation": "Escalate emergency telecommunications support and deploy backup communications where needed.",
    },
    {
        "id": "cluster-f",
        "short": "Cluster F",
        "name": "Education",
        "description": "Tracks school closures, displaced learners, and temporary learning needs.",
        "keywords": ["school closure", "class suspension", "learning materials", "temporary classroom", "deped", "students", "no classes", "school suspension", "learners", "class cancelled", "online class", "virtual learning", "school flooded", "school used as evacuation"],
        "accent": "#10b981",
        "recommendation": "Coordinate temporary learning support and school recovery planning with education partners.",
    },
    {
        "id": "cluster-g",
        "short": "Cluster G",
        "name": "Search, Rescue and Retrieval (SRR)",
        "description": "Tracks stranded people, rescue calls, rooftop signals, and retrieval updates.",
        "keywords": ["stranded", "roof", "rescue boat", "trapped family", "sos", "retrieval", "rescue team", "help needed", "trapped", "save us", "helicopter", "coast guard", "search party", "swift water rescue", "fire", "blaze", "burning", "firefighter", "fire alert"],
        "accent": "#ef4444",
        "recommendation": "Push rescue coordinates to the nearest SRR team and validate extraction access immediately.",
    },
    {
        "id": "cluster-h",
        "short": "Cluster H",
        "name": "Management of Dead and Missing (MDM)",
        "description": "Tracks missing persons, identification concerns, and related coordination updates.",
        "keywords": ["missing", "identified", "hospital list", "family tracing", "coordination desk", "missing person", "body identified", "casualty", "fatality", "unaccounted", "dead", "missing child", "lost contact", "remains", "deceased"],
        "accent": "#64748b",
        "recommendation": "Cross-check tracing, registry, and hospital intake data with missing-person coordination desks.",
    },
]

CLUSTER_MAP = {cluster["id"]: cluster for cluster in CLUSTER_DEFINITIONS}
SCENARIO_RECOMMENDATIONS = [
    {
        "cluster_id": "cluster-a",
        "priorities": {"HIGH"},
        "match_terms": ["food", "evacuation center", "no food", "hungry", "starving"],
        "recommendation": (
            "Coordinate with DSWD for immediate food pack deployment. Alert LGU and "
            "partner NGOs for supplementary relief distribution. Prioritize families "
            "with infants and elderly."
        ),
    },
    {
        "cluster_id": "cluster-a",
        "priorities": {"MEDIUM"},
        "match_terms": ["damit", "kumot", "blanket", "clothing", "hygiene"],
        "recommendation": (
            "Coordinate NFI distribution with DSWD and Red Cross. Pre-position "
            "blankets, clothing, and hygiene kits to affected evacuation center."
        ),
    },
    {
        "cluster_id": "cluster-a",
        "priorities": {"LOW"},
        "match_terms": ["relief goods", "dumating", "sapat", "salamat"],
        "recommendation": (
            "Log successful relief distribution. Continue monitoring for additional "
            "NFI needs. Assess remaining stock for next distribution cycle."
        ),
    },
    {
        "cluster_id": "cluster-b",
        "priorities": {"CRITICAL"},
        "match_terms": ["nasusuka", "nagtatae", "outbreak", "sakit na kumakalat", "diarrhea", "vomit"],
        "recommendation": (
            "Immediately dispatch DOH rapid response team. Isolate symptomatic "
            "individuals. Coordinate water quality testing and sanitation intervention. "
            "Alert RESU for disease outbreak investigation."
        ),
    },
    {
        "cluster_id": "cluster-b",
        "priorities": {"HIGH"},
        "match_terms": ["malinis na tubig", "cr", "toilet", "water", "sanitation"],
        "recommendation": (
            "Deploy water purification units and potable water trucks. Coordinate "
            "with LWUA for emergency water supply. Mobilize sanitation team for "
            "toilet facility augmentation."
        ),
    },
    {
        "cluster_id": "cluster-b",
        "priorities": {"MEDIUM"},
        "match_terms": ["bakuna", "vaccin", "immunization", "nahalkunahan"],
        "recommendation": (
            "Coordinate with DOH for mobile vaccination team deployment. Alert RHU "
            "for supplementary immunization activity. Conduct health status profiling "
            "of evacuees."
        ),
    },
    {
        "cluster_id": "cluster-c",
        "priorities": {"HIGH"},
        "match_terms": ["full capacity", "capacity", "cannot enter", "overflow", "new arrivals"],
        "recommendation": (
            "Activate secondary evacuation sites. Coordinate transport of overflow "
            "evacuees to alternate centers. Update evacuation center capacity "
            "registry with MDRRMO."
        ),
    },
    {
        "cluster_id": "cluster-c",
        "priorities": {"MEDIUM"},
        "match_terms": ["fighting", "crowd", "peacekeeping", "conflict"],
        "recommendation": (
            "Deploy barangay peacekeeping team and social workers to evacuation site. "
            "Coordinate with PNP for crowd management support. Establish camp "
            "management protocol."
        ),
    },
    {
        "cluster_id": "cluster-c",
        "priorities": {"HIGH"},
        "match_terms": ["child", "parent", "alone", "minor", "family tracing", "lost child"],
        "recommendation": (
            "Immediately coordinate with DSWD for child protection intervention. "
            "Deploy social workers for family tracing. Alert BCPC for unaccompanied "
            "minor documentation and safeguarding."
        ),
    },
    {
        "cluster_id": "cluster-d",
        "priorities": {"HIGH"},
        "match_terms": ["relief truck", "street", "road", "route", "flood", "blocked"],
        "recommendation": (
            "Coordinate with DPWH for alternate route assessment. Deploy amphibious "
            "or boat transport for relief delivery. Alert MDRRMO logistics team for "
            "rerouting plan."
        ),
    },
    {
        "cluster_id": "cluster-d",
        "priorities": {"CRITICAL"},
        "match_terms": ["not enough vehicles", "evacuation", "transport assets", "stranded group", "cannot leave"],
        "recommendation": (
            "Immediately mobilize all available LGU vehicles and coordinate with DOTC "
            "for additional transport assets. Request mutual aid from adjacent LGUs. "
            "Prioritize evacuation of vulnerable groups."
        ),
    },
    {
        "cluster_id": "cluster-d",
        "priorities": {"HIGH"},
        "match_terms": ["fuel", "rescue boat", "vessel", "operation"],
        "recommendation": (
            "Dispatch fuel supply team to reported coordinates. Coordinate with Coast "
            "Guard for backup vessel deployment. Alert rescue operations commander of "
            "operational delay."
        ),
    },
    {
        "cluster_id": "cluster-e",
        "priorities": {"HIGH"},
        "match_terms": ["no signal", "signal", "cannot contact", "network", "dead zone"],
        "recommendation": (
            "Coordinate with DICT for deployment of emergency satellite communication "
            "unit. Alert TELCO providers for priority signal restoration. Activate "
            "amateur radio network for backup communication."
        ),
    },
    {
        "cluster_id": "cluster-e",
        "priorities": {"MEDIUM"},
        "match_terms": ["hotline", "calling", "answering", "contact numbers"],
        "recommendation": (
            "Investigate MDRRMO hotline system status. Activate secondary communication "
            "channels. Post alternate contact numbers on official LGU social media "
            "pages immediately."
        ),
    },
    {
        "cluster_id": "cluster-e",
        "priorities": {"LOW"},
        "match_terms": ["restored", "restoration", "signal", "contacted"],
        "recommendation": (
            "Log restoration of communication in affected area. Continue monitoring "
            "for remaining communication gaps in adjacent barangays."
        ),
    },
    {
        "cluster_id": "cluster-f",
        "priorities": {"MEDIUM"},
        "match_terms": ["evacuation center", "resume class", "school facility", "learning spaces"],
        "recommendation": (
            "Coordinate with DepEd for school facility assessment and timeline for "
            "resumption. Identify alternative learning spaces. Alert DepEd Division "
            "Office for temporary learning modality activation."
        ),
    },
    {
        "cluster_id": "cluster-f",
        "priorities": {"MEDIUM"},
        "match_terms": ["books", "supplies", "materials", "damaged", "flood"],
        "recommendation": (
            "Coordinate with DepEd and partner NGOs for emergency learning materials "
            "replacement. Submit rapid school damage assessment to MDRRMO and DepEd "
            "Division Office."
        ),
    },
    {
        "cluster_id": "cluster-f",
        "priorities": {"MEDIUM"},
        "match_terms": ["scared", "panic", "earthquake", "psychosocial", "mhpss"],
        "recommendation": (
            "Coordinate with DepEd and DSWD for deployment of psychosocial support "
            "team. Conduct school-based MHPSS sessions before class resumption."
        ),
    },
    {
        "cluster_id": "cluster-g",
        "priorities": {"HIGH"},
        "match_terms": ["flood", "trapped", "water rescue", "cannot get out"],
        "recommendation": (
            "Deploy Search and Rescue units. Coordinate with MDRRMO and Coast Guard "
            "for water rescue operations. Report exact location to rescue operations "
            "commander."
        ),
    },
    {
        "cluster_id": "cluster-g",
        "priorities": {"CRITICAL"},
        "match_terms": ["collapse", "rescue now", "pinned", "usar"],
        "recommendation": (
            "Immediately dispatch Urban Search and Rescue (USAR) team. Coordinate "
            "with BFP and MDRRMO for structural collapse rescue protocol. Alert "
            "nearest hospital for trauma care standby."
        ),
    },
    {
        "cluster_id": "cluster-g",
        "priorities": {"LOW"},
        "match_terms": ["rescued", "no casualty", "completed rescue"],
        "recommendation": (
            "Log completed rescue operation. Document number of rescued individuals. "
            "Continue area monitoring for additional rescue requests."
        ),
    },
    {
        "cluster_id": "cluster-h",
        "priorities": {"HIGH"},
        "match_terms": ["corpse", "body", "retrieval", "identification"],
        "recommendation": (
            "Coordinate with PNP and NBI for body retrieval and identification. "
            "Notify MDRRMO for incident documentation. Alert barangay officials for "
            "missing persons cross-referencing."
        ),
    },
    {
        "cluster_id": "cluster-h",
        "priorities": {"MEDIUM"},
        "match_terms": ["still no news", "missing", "hospital admission", "evacuation center registries"],
        "recommendation": (
            "Register reported missing person with MDRRMO and PNP. Coordinate with "
            "barangay officials for local search. Cross-reference with evacuation "
            "center registries and hospital admission records."
        ),
    },
    {
        "cluster_id": "cluster-h",
        "priorities": {"CRITICAL"},
        "match_terms": ["maraming patay", "mass casualty", "landslide", "nawawala"],
        "recommendation": (
            "Immediately activate MDM protocol. Coordinate with PNP-SOCO, NBI, and "
            "MDRRMO for mass casualty management. Establish ante-mortem data "
            "collection. Alert NDRRMC for incident escalation report."
        ),
    },
]
EXCEL_RECOMMENDATIONS = {
    "cluster-a": {
        "HIGH": (
            "Coordinate with DSWD for immediate food pack deployment. Alert LGU and "
            "partner NGOs for supplementary relief distribution. Prioritize families "
            "with infants and elderly."
        ),
        "MEDIUM": (
            "Coordinate NFI distribution with DSWD and Red Cross. Pre-position "
            "blankets, clothing, and hygiene kits to affected evacuation center."
        ),
        "LOW": (
            "Log successful relief distribution. Continue monitoring for additional "
            "NFI needs. Assess remaining stock for next distribution cycle."
        ),
    },
    "cluster-b": {
        "CRITICAL": (
            "Immediately dispatch DOH rapid response team. Isolate symptomatic "
            "individuals. Coordinate water quality testing and sanitation intervention. "
            "Alert RESU for disease outbreak investigation."
        ),
        "HIGH": (
            "Deploy water purification units and potable water trucks. Coordinate "
            "with LWUA for emergency water supply. Mobilize sanitation team for "
            "toilet facility augmentation."
        ),
        "MEDIUM": (
            "Coordinate with DOH for mobile vaccination team deployment. Alert RHU "
            "for supplementary immunization activity. Conduct health status profiling "
            "of evacuees."
        ),
    },
    "cluster-c": {
        "HIGH": (
            "Activate secondary evacuation sites. Coordinate transport of overflow "
            "evacuees to alternate centers. Update evacuation center capacity "
            "registry with MDRRMO."
        ),
        "MEDIUM": (
            "Deploy barangay peacekeeping team and social workers to evacuation site. "
            "Coordinate with PNP for crowd management support. Establish camp "
            "management protocol."
        ),
    },
    "cluster-d": {
        "CRITICAL": (
            "Immediately mobilize all available LGU vehicles and coordinate with DOTC "
            "for additional transport assets. Request mutual aid from adjacent LGUs. "
            "Prioritize evacuation of vulnerable groups."
        ),
        "HIGH": (
            "Coordinate with DPWH for alternate route assessment. Deploy amphibious "
            "or boat transport for relief delivery. Alert MDRRMO logistics team for "
            "rerouting plan."
        ),
    },
    "cluster-e": {
        "HIGH": (
            "Coordinate with DICT for deployment of emergency satellite communication "
            "unit. Alert TELCO providers for priority signal restoration. Activate "
            "amateur radio network for backup communication."
        ),
        "MEDIUM": (
            "Investigate MDRRMO hotline system status. Activate secondary communication "
            "channels. Post alternate contact numbers on official LGU social media "
            "pages immediately."
        ),
        "LOW": (
            "Log restoration of communication in affected area. Continue monitoring "
            "for remaining communication gaps in adjacent barangays."
        ),
    },
    "cluster-f": {
        "MEDIUM": (
            "Coordinate with DepEd for school facility assessment and timeline for "
            "resumption. Identify alternative learning spaces. Alert DepEd Division "
            "Office for temporary learning modality activation."
        ),
    },
    "cluster-g": {
        "CRITICAL": (
            "Immediately dispatch Urban Search and Rescue (USAR) team. Coordinate "
            "with BFP and MDRRMO for structural collapse rescue protocol. Alert "
            "nearest hospital for trauma care standby."
        ),
        "HIGH": (
            "Deploy Search and Rescue units. Coordinate with MDRRMO and Coast Guard "
            "for water rescue operations. Report exact location to rescue operations "
            "commander."
        ),
        "LOW": (
            "Log completed rescue operation. Document number of rescued individuals. "
            "Continue area monitoring for additional rescue requests."
        ),
    },
    "cluster-h": {
        "CRITICAL": (
            "Immediately activate MDM protocol. Coordinate with PNP-SOCO, NBI, and "
            "MDRRMO for mass casualty management. Establish ante-mortem data "
            "collection. Alert NDRRMC for incident escalation report."
        ),
        "HIGH": (
            "Coordinate with PNP and NBI for body retrieval and identification. "
            "Notify MDRRMO for incident documentation. Alert barangay officials for "
            "missing persons cross-referencing."
        ),
        "MEDIUM": (
            "Register reported missing person with MDRRMO and PNP. Coordinate with "
            "barangay officials for local search. Cross-reference with evacuation "
            "center registries and hospital admission records."
        ),
    },
}
CLUSTER_SIGNAL_TERMS = {
    "cluster-a": {
        "relief goods", "food pack", "rice", "bigas", "pagkain", "tubig", "water", "drinking water", "blanket",
        "hygiene kit", "relief pack", "inumin", "canned goods", "sardinas", "noodles", "ayuda", "relief donation", "food donation",
        "NFI", "distribution", "supply", "water sachet", "DSWD", "relief operations", "relief convoy",
        "packed relief", "family pack", "hygiene pack",
    },
    "cluster-b": {
        "doctor", "nurse", "hospital", "clinic", "medical", "medicine", "gamot", "health", "dehydration",
        "fever", "clean water", "sanitation", "ospital", "sugat", "wound", "injured", "sick",
        "leptospirosis", "diarrhea", "cholera", "first aid", "health center", "tubig linis",
        "DOH", "medical team", "ambulance", "triage", "medical mission", "contaminant",
    },
    "cluster-c": {
        "evacuation center", "evacuation site", "evacuees", "shelter", "camp", "registration", "safe space",
        "overcapacity", "crowded", "likas", "displaced", "pabahay", "gym", "covered court",
        "barangay hall", "tent", "displaced families", "evacuation area", "preemptive evacuation",
        "mandatory evacuation", "danger zone", "evacuated",
    },
    "cluster-d": {
        "blocked road", "bridge", "road", "truck", "convoy", "delivery", "reroute", "warehouse", "transport",
        "passable", "impassable", "landslide", "debris", "alternate route", "road clearing",
        "mudslide", "blocked bridge", "guho", "collapsed road", "DPWH", "road damage", "road closed",
        "fallen tree", "not passable", "road cut off",
    },
    "cluster-e": {
        "signal", "no signal", "no network", "network", "connectivity", "radio", "cell site", "internet",
        "telecom", "communication", "power bank", "walang signal", "dead zone", "no wifi", "brownout",
        "blackout", "walang kuryente", "generator", "power outage", "Meralco", "blackout", "no electricity",
        "signal restored", "communication cut", "fiber optic",
    },
    "cluster-f": {
        "school", "class suspension", "school closure", "walang pasok", "deped", "students", "learners",
        "class cancelled", "temporary classroom", "walang klase", "school flooded", "online class",
        "virtual learning", "no classes", "classes suspended", "academic calendar", "CHED",
    },
    "cluster-g": {
        "rescue", "rescue team", "rescue boat", "stranded", "trapped", "saklolo", "sos", "roof", "retrieval",
        "save us", "tabang", "nastranded", "naipit", "helicopter", "coast guard", "naiipit",
        "swift water", "search party", "BFP", "Bureau of Fire", "fire alert", "fire truck", "blaze",
        "burning", "structure fire", "five alarm", "call for help", "trapped", "SOS",
    },
    "cluster-h": {
        "missing", "missing person", "identified", "family tracing", "casualty", "fatality", "body identified",
        "hospital list", "unaccounted", "namatay", "patay", "nawawala", "nawala",
        "missing child", "lost contact", "death toll", "confirmed dead", "found dead", "morgue",
        "next of kin", "remains",
    },
}
# Maps CorEx topic labels → NDRRMC cluster IDs.
# Must stay in sync with TOPIC_LABELS in services/corex/topic_modeler.py.
TOPIC_TO_CLUSTER: dict[str, str] = {
    "education":      "cluster-f",
    "evacuation":     "cluster-c",
    "rescue":         "cluster-g",
    "logistics":      "cluster-d",
    "relief":         "cluster-a",
    "telecom_power":  "cluster-e",
    "health_medical": "cluster-b",
    "dead_missing":   "cluster-h",
}

_COREX_KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), "models", "corex_keywords.json")


def load_corex_expanded_keywords() -> dict[str, list[str]]:
    """Load CorEx-discovered keywords from the saved model file.

    Returns {topic_label: [word, ...]} or {} if CorEx has not been trained yet.
    Called by the pipeline to enrich SVM training labels after each CorEx retrain.
    """
    if not os.path.exists(_COREX_KEYWORDS_PATH):
        return {}
    try:
        with open(_COREX_KEYWORDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


PRIORITY_ORDER = {"Monitoring": 1, "Moderate": 2, "High": 3, "Critical": 4}
DISTRESS_TERMS = {
    "urgent": 8,
    "alert": 8,
    "critical": 12,
    "danger": 10,
    "stranded": 14,
    "rescue": 14,
    "sos": 18,
    "evacuate": 10,
    "warning": 8,
    "lagnat": 8,
    "hospital": 8,
    "trapped": 14,
    "ashfall": 10,
    "volcano": 6,
    "flood": 8,
}
LOCATION_PATTERNS = [
    re.compile(r"#([A-Z][A-Za-z]+)"),
    re.compile(r"\b(?:sa|ng|of)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})"),
]


def now_utc():
    return datetime.now(timezone.utc)


def parse_date_range(date_range: str) -> timedelta:
    mapping = {"24h": timedelta(days=1), "3d": timedelta(days=3), "7d": timedelta(days=7), "14d": timedelta(days=14), "30d": timedelta(days=30)}
    return mapping.get(date_range, timedelta(days=7))


def seed_clusters():
    for cluster in CLUSTER_DEFINITIONS:
        existing = db.session.get(Cluster, cluster["id"])
        if existing:
            existing.short = cluster["short"]
            existing.name = cluster["name"]
            existing.description = cluster["description"]
            existing.accent = cluster["accent"]
            existing.keywords_json = json.dumps(cluster["keywords"])
        else:
            db.session.add(
                Cluster(
                    id=cluster["id"],
                    short=cluster["short"],
                    name=cluster["name"],
                    description=cluster["description"],
                    accent=cluster["accent"],
                    keywords_json=json.dumps(cluster["keywords"]),
                )
            )
    db.session.commit()


def infer_cluster(text: str):
    lower = (text or "").lower()

    # Soft Irrelevance Penalty: civic/cultural terms raise the minimum threshold.
    # Posts about fashion shows or festivals won't classify unless 3+ disaster
    # keywords are present — a genuine disaster at a public event would still pass.
    IRRELEVANCE_TERMS = [
        "fashion", "fiesta", "pageant", "sagala", "flores de mayo",
        "festival", "concert", "parade", "beauty queen", "gala", "exhibit",
        "inauguration", "ceremony", "anniversary", "awarding", "pride month",
        "christmas", "new year", "valentines", "holy week",
        "impeachment", "senate", "congress", "ordinance", "proclamation",
        "promo", "discount", "sale", "voucher", "raffle", "giveaway", "free trial",
        "launching", "ribbon cutting", "groundbreaking",
    ]
    is_civic_post = any(
        re.search(r'\b' + re.escape(term) + r'\b', lower)
        for term in IRRELEVANCE_TERMS
    )
    # Normal posts need score >= 3; civic posts need >= 9 (3 strong keyword matches)
    min_score = 9 if is_civic_post else 3

    best_cluster = CLUSTER_DEFINITIONS[0]
    best_score = -1
    matched_keywords = []

    for cluster in CLUSTER_DEFINITIONS:
        exact_matches = [keyword for keyword in cluster["keywords"] if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', lower)]
        signal_matches = [term for term in CLUSTER_SIGNAL_TERMS.get(cluster["id"], set()) if re.search(r'\b' + re.escape(term) + r'\b', lower)]
        weighted_score = (len(exact_matches) * 3) + len(signal_matches)

        if any(re.search(r'\b' + re.escape(term) + r'\b', lower) for term in ["sos", "rescue", "trapped", "stranded", "fire"]) and cluster["id"] == "cluster-g":
            weighted_score += 4
        if any(re.search(r'\b' + re.escape(term) + r'\b', lower) for term in ["evacuation center", "evacuees", "shelter"]) and cluster["id"] == "cluster-c":
            weighted_score += 3
        if any(re.search(r'\b' + re.escape(term) + r'\b', lower) for term in ["missing", "fatality", "body identified"]) and cluster["id"] == "cluster-h":
            weighted_score += 4

        score = weighted_score
        if score > best_score:
            best_cluster = cluster
            best_score = score
            matched_keywords = exact_matches or signal_matches

    # Apply threshold (higher for civic posts)
    if best_score < min_score:
        # Only check fallback if it's not a civic post (civic posts require real keyword density)
        if not is_civic_post:
            fallback_map = [
                ("weather", "cluster-e"),
                ("heat index", "cluster-b"),
                ("air quality", "cluster-b"),
                ("volcano", "cluster-g"),
                ("ash", "cluster-g"),
                ("evacuation", "cluster-c"),
                ("relief", "cluster-a"),
                ("medicine", "cluster-b"),
                ("school", "cluster-f"),
                ("signal", "cluster-e"),
                ("fire", "cluster-g"),
            ]
            for trigger, cluster_id in fallback_map:
                if re.search(r'\b' + re.escape(trigger) + r'\b', lower):
                    best_cluster = CLUSTER_MAP[cluster_id]
                    matched_keywords = [trigger]
                    break
            else:
                return None, []
        else:
            return None, []

    hashtags = [token.strip("#") for token in re.findall(r"#([A-Za-z][A-Za-z0-9]+)", text or "")]
    keywords = matched_keywords or hashtags[:3]
    return best_cluster, keywords[:6]


def infer_sentiment_score(text: str, engagement: int) -> int:
    """
    Return a distress score in range [20, 97].
    Primary: VADER compound mapped via compound_to_score().
    Fallback: keyword heuristic if vaderSentiment is unavailable.
    """
    try:
        from services.vader.sentiment_analyzer import analyze_sentiment, compound_to_score
        result = analyze_sentiment(text or "")
        return compound_to_score(result["compound"])
    except Exception:
        lower = (text or "").lower()
        score = 58
        for term, weight in DISTRESS_TERMS.items():
            if term in lower:
                score += weight
        if engagement >= 100:
            score += 8
        if engagement >= 250:
            score += 8
        return max(20, min(score, 97))


def infer_priority(text: str, engagement: int):
    lower = (text or "").lower()
    if any(term in lower for term in ["sos", "rescue", "stranded", "trapped", "critical", "danger zone"]):
        return "Critical"
    if any(re.search(r'\b' + re.escape(term) + r'\b', lower) for term in ["alert", "warning", "volcano", "flood", "medical", "heat index"]) or re.search(r'\bevacuat', lower) or re.search(r'\bash\b', lower):
        return "High"
    if engagement >= 100:
        return "High"
    return "Moderate"


def extract_location(text: str):
    source = text or ""
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(source)
        if match:
            return match.group(1).replace("#", "").strip()
    return "Philippines"


def normalize_recommendation_priority(priority: str) -> str:
    return {
        "Critical": "CRITICAL",
        "High": "HIGH",
        "Moderate": "MEDIUM",
        "Medium": "MEDIUM",
        "Monitoring": "LOW",
        "Low": "LOW",
    }.get((priority or "").strip(), "MEDIUM")


def recommendation_payload_for(
    cluster_id: str | None,
    priority: str,
    *,
    sentiment: str | None = None,
    sentiment_score: int | float | None = None,
    reactions: int = 0,
    likes: int = 0,
    comments: int = 0,
    shares: int = 0,
    reposts: int = 0,
    post_count: int = 1,
):
    topic = cluster_id if cluster_id in CLUSTER_MAP else None
    reaction_total = int(reactions or 0) or int(likes or 0)
    share_total = int(shares or 0) or int(reposts or 0)
    return evaluate_from_post(
        topic=topic or "general",
        priority=normalize_recommendation_priority(priority),
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        reactions=reaction_total,
        comments=int(comments or 0),
        shares=share_total,
        post_count=post_count,
    )


def recommendation_for(
    cluster_id: str | None,
    priority: str,
    text: str | None = None,
    *,
    sentiment: str | None = None,
    sentiment_score: int | float | None = None,
    reactions: int = 0,
    likes: int = 0,
    comments: int = 0,
    shares: int = 0,
    reposts: int = 0,
    post_count: int = 1,
):
    del text
    payload = recommendation_payload_for(
        cluster_id,
        priority,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        reactions=reactions,
        likes=likes,
        comments=comments,
        shares=shares,
        reposts=reposts,
        post_count=post_count,
    )
    return payload["recommendation"]


def priority_label(priority: str):
    return {"Monitoring": "Low", "Moderate": "Medium"}.get(priority, priority)


def score_tone(score: int):
    if score >= 80:
        return "negative"
    if score >= 60:
        return "neutral"
    return "positive"


def media_type_for(item: dict):
    if item.get("isVideo"):
        return "video"
    if item.get("media"):
        return "photo"
    return "text"


def top_keywords_from_posts(posts, limit=6):
    counts = Counter()
    notes = {}
    for post in posts:
        for keyword in post.keywords:
            counts[keyword] += 1
            notes.setdefault(keyword, f"{CLUSTER_MAP[post.cluster_id]['short']} surge")
    top = counts.most_common(limit)
    return [{"keyword": keyword, "note": notes.get(keyword, "Detected keyword"), "count": count} for keyword, count in top]
