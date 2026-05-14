"""
Thesis-approved deterministic recommendation engine.

Recommendation inputs:
1. topic
2. sentiment
3. engagement score
4. predicted priority

The engine is intentionally rule-based and explainable. Topic selects the
response category, while sentiment, engagement score, and predicted priority
select the recommendation level.
"""

from __future__ import annotations

from dataclasses import dataclass


HIGH_ENGAGEMENT_THRESHOLD = 1000
MODERATE_ENGAGEMENT_THRESHOLD = 300

HIGH_PRIORITIES = {"HIGH", "CRITICAL"}
MEDIUM_OR_HIGH_PRIORITIES = {"MEDIUM", "HIGH", "CRITICAL"}

TOPIC_LABELS = {
    "cluster-a": "Relief Supply",
    "cluster-b": "Health and WASH",
    "cluster-c": "Evacuation Center Management",
    "cluster-d": "Logistics",
    "cluster-e": "Emergency Telecommunications",
    "cluster-f": "Education",
    "cluster-g": "Search and Rescue",
    "cluster-h": "Missing Persons",
}

TOPIC_RESPONSES = {
    "cluster-a": {
        "HIGH": "Immediate LGU Relief and DSWD Dispatch",
        "MEDIUM": "Relief Needs Validation and Prepositioning",
        "LOW": "Routine Relief Monitoring",
        "HIGH_TEXT": (
            "Coordinate with DSWD for immediate food pack deployment. Alert LGU and "
            "partner NGOs for supplementary relief distribution. Prioritize infants, "
            "elderly, and other vulnerable evacuees."
        ),
        "MEDIUM_TEXT": (
            "Coordinate NFI distribution with DSWD and Red Cross. Preposition "
            "blankets, clothing, and hygiene kits in the evacuation center. Monitor "
            "inventory for follow-up replenishment."
        ),
        "LOW_TEXT": (
            "Log successful relief distribution. Continue monitoring food and NFI "
            "levels. Assess remaining stock for the next distribution cycle."
        ),
    },
    "cluster-b": {
        "HIGH": "Immediate MDRRMO Health Response",
        "MEDIUM": "Barangay Health Validation and Supply Readiness",
        "LOW": "Routine Health Monitoring",
        "HIGH_TEXT": (
            "Deploy a rapid health response team immediately. Coordinate with DOH and "
            "RHU for disease surveillance, medical assessment, and sanitation "
            "intervention. Isolate symptomatic individuals and inspect water sources."
        ),
        "MEDIUM_TEXT": (
            "Deploy potable water supply and sanitation support to the evacuation "
            "center. Coordinate with WASH teams for toilet cleaning, desludging, and "
            "hygiene promotion. Monitor public health risks among evacuees."
        ),
        "LOW_TEXT": (
            "Log the stabilization of health conditions in the evacuation center. "
            "Continue routine monitoring, basic treatment, and health information "
            "dissemination. Maintain standby support for vulnerable groups."
        ),
    },
    "cluster-c": {
        "HIGH": "Immediate Evacuation Center Protection Response",
        "MEDIUM": "Evacuation Site Validation and Support",
        "LOW": "Routine Shelter Monitoring",
        "HIGH_TEXT": (
            "Activate secondary evacuation sites. Coordinate transport of overflow "
            "evacuees to alternate safe sites. Update camp capacity records and "
            "inform MDRRMO of congestion status."
        ),
        "MEDIUM_TEXT": (
            "Deploy camp management personnel, barangay peacekeeping staff, and "
            "social workers to restore order. Coordinate with PNP if crowd control "
            "support is needed. Re-establish camp rules and complaint handling "
            "procedures."
        ),
        "LOW_TEXT": (
            "Log improved camp organization and volunteer coordination. Continue "
            "monitoring protection concerns and service flow. Maintain regular camp "
            "management supervision."
        ),
    },
    "cluster-d": {
        "HIGH": "Immediate Logistics and Route Clearing Response",
        "MEDIUM": "Logistics Validation and Rerouting Readiness",
        "LOW": "Routine Logistics Monitoring",
        "HIGH_TEXT": (
            "Coordinate with MDRRMO and logistics teams to identify alternate "
            "delivery routes immediately. Deploy smaller transport units or boats if "
            "needed. Prioritize delivery of essential relief items to isolated areas."
        ),
        "MEDIUM_TEXT": (
            "Reassign available transport assets and coordinate with partner agencies "
            "for supplemental hauling support. Review dispatch scheduling and update "
            "expected delivery times to field responders. Monitor stock levels until "
            "delivery is completed."
        ),
        "LOW_TEXT": (
            "Log successful delivery and distribution of supplies. Continue routine "
            "monitoring of warehouse stock and field replenishment needs. Maintain "
            "delivery readiness for follow-up requests."
        ),
    },
    "cluster-e": {
        "HIGH": "Immediate Emergency Communications Restoration",
        "MEDIUM": "Communications Validation and Backup Readiness",
        "LOW": "Routine Communications Monitoring",
        "HIGH_TEXT": (
            "Coordinate with DICT for deployment of emergency satellite communication "
            "unit. Alert TELCO providers for priority signal restoration. Activate "
            "amateur radio network for backup communication."
        ),
        "MEDIUM_TEXT": (
            "Investigate MDRRMO hotline system status. Activate secondary communication "
            "channels and publish alternate hotlines immediately. Monitor unresolved "
            "communication complaints for escalation."
        ),
        "LOW_TEXT": (
            "Log restoration of communication services in the affected area. Continue "
            "monitoring for intermittent outages and nearby signal gaps. Maintain "
            "advisory updates until service becomes stable."
        ),
    },
    "cluster-f": {
        "HIGH": "Immediate Education Continuity Coordination",
        "MEDIUM": "School Disruption Validation and Coordination",
        "LOW": "Routine Education Monitoring",
        "HIGH_TEXT": (
            "Coordinate with DepEd and MDRRMO for urgent school use assessment and "
            "transition planning. Identify temporary learning spaces or alternate "
            "class arrangements. Issue an advisory on safe resumption timelines."
        ),
        "MEDIUM_TEXT": (
            "Coordinate with DepEd and partner organizations for replacement of "
            "damaged learning materials and classroom supplies. Conduct a rapid "
            "damage assessment and prioritize affected grade levels."
        ),
        "LOW_TEXT": (
            "Log school reopening and recovery status. Continue monitoring remaining "
            "educational material needs and student attendance. Maintain coordination "
            "with school administrators for follow-up concerns."
        ),
    },
    "cluster-g": {
        "HIGH": "Immediate MDRRMO Disaster Response",
        "MEDIUM": "Rescue Validation and Standby Deployment",
        "LOW": "Routine Rescue Monitoring",
        "HIGH_TEXT": (
            "Deploy Search and Rescue units immediately to the reported location. "
            "Coordinate with MDRRMO, BFP, and water rescue teams for extraction. "
            "Relay exact location details to the rescue operations commander."
        ),
        "MEDIUM_TEXT": (
            "Validate the incident and dispatch the nearest available rescue team. "
            "Prioritize assisted evacuation of vulnerable individuals. Coordinate "
            "with local health responders in case medical transport is needed."
        ),
        "LOW_TEXT": (
            "Log the completed rescue operation and document the number of "
            "individuals assisted. Continue monitoring the area for follow-up rescue "
            "needs. Coordinate with evacuation center personnel for status "
            "confirmation."
        ),
    },
    "cluster-h": {
        "HIGH": "Immediate Missing Persons Coordination",
        "MEDIUM": "Case Validation and Tracing Readiness",
        "LOW": "Routine Missing Persons Monitoring",
        "HIGH_TEXT": (
            "Coordinate with PNP, rescue personnel, and medico-legal teams for body "
            "retrieval and identification procedures. Notify MDRRMO for "
            "documentation and cross-reference with missing persons reports."
        ),
        "MEDIUM_TEXT": (
            "Register the missing person case with MDRRMO and PNP. Cross-reference "
            "evacuation center registries, hospital admissions, and available "
            "incident reports. Coordinate with barangay officials for localized "
            "tracing."
        ),
        "LOW_TEXT": (
            "Log the resolution of the missing person report. Update incident records "
            "and close the tracing case after verification. Continue monitoring for "
            "related or duplicate reports."
        ),
    },
}

DEFAULT_TOPIC = "general"
DEFAULT_TOPIC_LABEL = "General Incident"
DEFAULT_RESPONSES = {
    "HIGH": "Immediate LGU Assessment",
    "MEDIUM": "Field Validation and Monitoring",
    "LOW": "Routine Situation Monitoring",
    "HIGH_TEXT": "Immediate LGU assessment is recommended due to the combined distress indicators.",
    "MEDIUM_TEXT": "Validate the reported concern and maintain response readiness while monitoring for escalation.",
    "LOW_TEXT": "Continue routine monitoring until stronger urgency signals are observed.",
}


@dataclass(frozen=True)
class RecommendationResult:
    recommendation: str
    rule_id: str
    rationale: str
    inputs: dict

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "rule_id": self.rule_id,
            "rationale": self.rationale,
            "inputs": self.inputs,
        }


def normalize_topic(topic: str | None) -> str:
    key = (topic or "").strip().lower()
    if key in TOPIC_LABELS:
        return key
    for topic_id, label in TOPIC_LABELS.items():
        if key == label.lower():
            return topic_id
    return DEFAULT_TOPIC


def normalize_sentiment(sentiment: str | None) -> str:
    value = (sentiment or "").strip().lower()
    if value == "negative":
        return "Negative"
    if value == "positive":
        return "Positive"
    return "Neutral"


def normalize_priority(priority: str | None) -> str:
    value = (priority or "").strip().lower()
    mapping = {
        "critical": "CRITICAL",
        "high": "HIGH",
        "moderate": "MEDIUM",
        "medium": "MEDIUM",
        "monitoring": "LOW",
        "low": "LOW",
    }
    return mapping.get(value, "MEDIUM")


def sentiment_from_score(sentiment_score: int | float | None) -> str:
    score = int(sentiment_score or 0)
    if score >= 80:
        return "Negative"
    if score >= 60:
        return "Neutral"
    return "Positive"


def compute_engagement_score(
    post_count: int = 1,
    reactions: int = 0,
    comments: int = 0,
    shares: int = 0,
) -> int:
    count = max(int(post_count or 0), 1)
    return (
        (count * 10)
        + (int(reactions or 0) * 1)
        + (int(comments or 0) * 3)
        + (int(shares or 0) * 5)
    )


def _resolve_level(sentiment: str, engagement_score: int, priority: str) -> str:
    if priority in HIGH_PRIORITIES:
        return "HIGH"
    if priority == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def _rationale_for(level: str, topic_label: str, sentiment: str, engagement_score: int, priority: str) -> str:
    if level == "HIGH":
        return (
            f"{topic_label} topic received predicted priority {priority}, so the engine "
            f"used the HIGH recommendation mapped for that cluster. Supporting inputs "
            f"were sentiment={sentiment} and engagement_score={engagement_score}."
        )
    if level == "MEDIUM":
        return (
            f"{topic_label} topic received predicted priority {priority}, so the engine "
            f"used the MEDIUM recommendation mapped for that cluster. Supporting inputs "
            f"were sentiment={sentiment} and engagement_score={engagement_score}."
        )
    return (
        f"{topic_label} topic received predicted priority {priority}, so the engine "
        f"used the LOW recommendation mapped for that cluster. Supporting inputs were "
        f"sentiment={sentiment} and engagement_score={engagement_score}."
    )


def evaluate(topic: str, sentiment: str, engagement_score: int, priority: str) -> dict:
    topic_key = normalize_topic(topic)
    sentiment_label = normalize_sentiment(sentiment)
    priority_label = normalize_priority(priority)
    score = int(engagement_score or 0)

    profile = TOPIC_RESPONSES.get(topic_key, DEFAULT_RESPONSES)
    topic_label = TOPIC_LABELS.get(topic_key, DEFAULT_TOPIC_LABEL)
    level = _resolve_level(sentiment_label, score, priority_label)
    rule_id = f"REC-{topic_key.replace('cluster-', '').upper() if topic_key != DEFAULT_TOPIC else 'GEN'}-{level}"
    rationale = _rationale_for(level, topic_label, sentiment_label, score, priority_label)

    return RecommendationResult(
        recommendation=profile[f"{level}_TEXT"],
        rule_id=rule_id,
        rationale=rationale,
        inputs={
            "topic": topic_label,
            "sentiment": sentiment_label,
            "engagement_score": score,
            "priority": priority_label,
        },
    ).to_dict()


def evaluate_from_post(
    topic: str,
    priority: str,
    sentiment: str | None = None,
    sentiment_score: int | float | None = None,
    reactions: int = 0,
    comments: int = 0,
    shares: int = 0,
    post_count: int = 1,
) -> dict:
    sentiment_label = normalize_sentiment(sentiment) if sentiment else sentiment_from_score(sentiment_score)
    engagement_score = compute_engagement_score(
        post_count=post_count,
        reactions=reactions,
        comments=comments,
        shares=shares,
    )
    return evaluate(topic, sentiment_label, engagement_score, priority)


def list_rules() -> list[dict]:
    rules = []
    for topic_id, topic_label in TOPIC_LABELS.items():
        rules.extend([
            {
                "rule_id": f"REC-{topic_id.replace('cluster-', '').upper()}-HIGH",
                "topic": topic_label,
                "condition": (
                    "IF predicted priority IN {HIGH, CRITICAL}"
                ),
                "recommendation": TOPIC_RESPONSES[topic_id]["HIGH_TEXT"],
            },
            {
                "rule_id": f"REC-{topic_id.replace('cluster-', '').upper()}-MEDIUM",
                "topic": topic_label,
                "condition": (
                    "IF predicted priority = MEDIUM"
                ),
                "recommendation": TOPIC_RESPONSES[topic_id]["MEDIUM_TEXT"],
            },
            {
                "rule_id": f"REC-{topic_id.replace('cluster-', '').upper()}-LOW",
                "topic": topic_label,
                "condition": "IF predicted priority = LOW",
                "recommendation": TOPIC_RESPONSES[topic_id]["LOW_TEXT"],
            },
        ])
    return rules
