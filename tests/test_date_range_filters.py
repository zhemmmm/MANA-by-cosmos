from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app import app
from data import seed_clusters
from models import Post, db


def _post(post_id: str, posted_at: datetime, scraped_at: datetime) -> Post:
    return Post(
        id=post_id,
        source="Facebook",
        page_source="Regression Source",
        author="Regression Source",
        caption="Flood relief goods needed at evacuation center.",
        source_url=f"https://example.test/{post_id}",
        external_id=post_id,
        reactions=10,
        shares=2,
        likes=10,
        reposts=0,
        comments=1,
        views=0,
        media_type="text",
        priority="High",
        sentiment_score=80,
        recommendation="Coordinate relief response.",
        status="Monitoring",
        cluster_id="cluster-a",
        is_relevant=True,
        date=posted_at,
        created_at=scraped_at,
        updated_at=scraped_at,
        keywords_json=json.dumps(["relief goods", "evacuation center"]),
        location="Philippines",
        severity_rank=3,
        raw_payload_json="{}",
    )


class DateRangeFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(
            TESTING=True,
            SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
            SQLALCHEMY_ENGINE_OPTIONS={},
        )

    def setUp(self):
        self.ctx = app.app_context()
        self.ctx.push()
        db.session.remove()
        db.drop_all()
        db.create_all()
        seed_clusters()

        scraped_now = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
        db.session.add_all(
            [
                _post(
                    "old-post-recent-scrape",
                    datetime(2025, 11, 1, 8, 0, tzinfo=timezone.utc),
                    scraped_now,
                ),
                _post(
                    "recent-post",
                    datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc),
                    scraped_now,
                ),
            ]
        )
        db.session.commit()
        self.client = app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_posts_date_range_uses_post_date_not_scrape_date(self):
        response = self.client.get("/api/posts?date_range=30d")
        self.assertEqual(response.status_code, 200)

        ids = {item["id"] for item in response.get_json()}
        self.assertIn("recent-post", ids)
        self.assertNotIn("old-post-recent-scrape", ids)

    def test_all_date_range_still_includes_all_post_dates(self):
        response = self.client.get("/api/posts?date_range=all")
        self.assertEqual(response.status_code, 200)

        ids = {item["id"] for item in response.get_json()}
        self.assertEqual(ids, {"recent-post", "old-post-recent-scrape"})

    def test_dashboard_summary_date_range_uses_post_date(self):
        response = self.client.get("/api/dashboard/summary?date_range=30d")
        self.assertEqual(response.status_code, 200)

        totals = response.get_json()["totals"]
        self.assertEqual(totals["totalPostsAnalyzed"], 1)
        self.assertEqual(totals["totalFacebookPosts"], 1)


if __name__ == "__main__":
    unittest.main()
