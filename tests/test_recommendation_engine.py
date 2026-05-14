from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from data import recommendation_payload_for
from models import Post
from services.rules.decision_engine import compute_engagement_score, evaluate


class RecommendationEngineTests(unittest.TestCase):
    def test_high_rule_uses_all_four_inputs(self):
        result = evaluate("cluster-g", "Negative", 1800, "High")
        self.assertEqual(result["rule_id"], "REC-G-HIGH")
        self.assertEqual(result["inputs"]["topic"], "Search and Rescue")
        self.assertEqual(result["inputs"]["sentiment"], "Negative")
        self.assertEqual(result["inputs"]["engagement_score"], 1800)
        self.assertEqual(result["inputs"]["priority"], "HIGH")
        self.assertIn("Deploy Search and Rescue units", result["recommendation"])

    def test_medium_rule_triggers_for_negative_sentiment_with_moderate_engagement(self):
        score = compute_engagement_score(post_count=1, reactions=250, comments=20, shares=15)
        result = evaluate("cluster-d", "Negative", score, "Medium")
        self.assertEqual(score, 395)
        self.assertEqual(result["rule_id"], "REC-D-MEDIUM")
        self.assertEqual(result["inputs"]["priority"], "MEDIUM")

    def test_high_priority_always_uses_high_cluster_mapping(self):
        result = evaluate("cluster-a", "Neutral", 2200, "Critical")
        self.assertEqual(result["rule_id"], "REC-A-HIGH")
        self.assertEqual(result["inputs"]["priority"], "CRITICAL")

    def test_medium_priority_does_not_fall_back_to_low_mapping(self):
        result = evaluate("cluster-c", "Positive", 20, "Medium")
        self.assertEqual(result["rule_id"], "REC-C-MEDIUM")
        self.assertEqual(result["inputs"]["priority"], "MEDIUM")

    def test_payload_helper_computes_thesis_engagement_formula(self):
        result = recommendation_payload_for(
            "cluster-b",
            "High",
            sentiment_score=92,
            reactions=600,
            comments=40,
            shares=60,
            post_count=1,
        )
        self.assertEqual(result["inputs"]["engagement_score"], 1030)
        self.assertEqual(result["rule_id"], "REC-B-HIGH")

    def test_post_api_dict_uses_live_recommendation_mapping(self):
        post = Post(
            id="demo-1",
            source="Facebook",
            page_source="Demo Page",
            caption="Relief delivered successfully.",
            source_url="https://example.com/post/1",
            cluster_id="cluster-a",
            priority="Low",
            sentiment_score=40,
            recommendation="stale high recommendation",
            reactions=5,
            likes=5,
            comments=1,
            shares=0,
            reposts=0,
            date=datetime.now(timezone.utc),
        )
        api_post = post.to_api_dict()
        self.assertEqual(
            api_post["recommendation"],
            "Log successful relief distribution. Continue monitoring food and NFI levels. Assess remaining stock for the next distribution cycle.",
        )
        self.assertEqual(api_post["recommendationDetails"]["rule_id"], "REC-A-LOW")


if __name__ == "__main__":
    unittest.main()
